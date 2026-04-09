from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from vlm_anchor.utils import extract_first_number, normalize_numeric_text


try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None


def _to_pil(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, (str, Path)):
        return Image.open(image_like).convert("RGB")
    if hasattr(image_like, "convert"):
        return image_like.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_like)}")


def _to_base64_png(image_like: Any) -> str:
    image = _to_pil(image_like)
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@dataclass
class InferenceConfig:
    system_prompt: str
    user_template: str
    temperature: float
    top_p: float
    num_ctx: int
    max_new_tokens: int


@dataclass
class ImageTokenLayout:
    token_indices: np.ndarray
    grid_shape: tuple[int, int]


@dataclass
class TokenActivationRecord:
    sequence_position: int
    token_id: int
    token_piece: str
    image_scores: np.ndarray


@dataclass
class AttentionVisualizationConfig:
    method: str = "tam"
    span_source: str = "target_or_prediction"
    head_top_k: int = 4
    head_min_image_mass: float = 0.05
    layer_start_ratio: float = 0.5
    layer_count: int = 4
    layer_strategy: str = "mean"
    tam_filter_size: int = 3

    def __post_init__(self) -> None:
        self.method = str(self.method or "tam").strip().lower()
        self.span_source = str(self.span_source or "target_or_prediction").strip().lower()
        self.head_top_k = max(1, int(self.head_top_k))
        self.head_min_image_mass = max(0.0, float(self.head_min_image_mass))
        self.layer_start_ratio = min(1.0, max(0.0, float(self.layer_start_ratio)))
        self.layer_count = max(1, int(self.layer_count))
        self.layer_strategy = str(self.layer_strategy or "mean").strip().lower()
        self.tam_filter_size = max(1, int(self.tam_filter_size))
        if self.tam_filter_size % 2 == 0:
            self.tam_filter_size += 1
        if self.method not in {"tam", "attention"}:
            raise ValueError(f"Unsupported visualization method: {self.method}")
        if self.layer_strategy not in {"mean", "rollout"}:
            raise ValueError(f"Unsupported layer strategy: {self.layer_strategy}")

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "AttentionVisualizationConfig":
        if not cfg:
            return cls()

        attn_cfg = cfg.get("attention", cfg)
        head_cfg = attn_cfg.get("head_selection", {})
        layer_cfg = attn_cfg.get("layer_aggregation", {})
        tam_cfg = attn_cfg.get("tam", {})

        use_last_layer = bool(attn_cfg.get("use_last_layer", False))
        default_layer_count = 1 if use_last_layer else 4
        default_layer_start_ratio = 1.0 if use_last_layer else 0.5

        return cls(
            method=attn_cfg.get("method", "tam"),
            span_source=attn_cfg.get("span_source", "target_or_prediction"),
            head_top_k=head_cfg.get("top_k", attn_cfg.get("head_top_k", 4)),
            head_min_image_mass=head_cfg.get("min_image_mass", attn_cfg.get("head_min_image_mass", 0.05)),
            layer_start_ratio=layer_cfg.get("start_ratio", attn_cfg.get("layer_start_ratio", default_layer_start_ratio)),
            layer_count=layer_cfg.get("count", attn_cfg.get("layer_count", default_layer_count)),
            layer_strategy=layer_cfg.get("strategy", attn_cfg.get("layer_strategy", "mean")),
            tam_filter_size=tam_cfg.get("filter_size", attn_cfg.get("tam_filter_size", 3)),
        )


def _ceil_div(value: int, divisor: int) -> int:
    return int(ceil(value / divisor))


def _supports_attention_visualization(model_name: str) -> bool:
    lower_name = model_name.lower()
    if "llava" in lower_name:
        return False
    return any(family in lower_name for family in ("qwen", "gemma"))


class OllamaVisionRunner:
    supports_attention = False

    def __init__(self, model_name: str, inference_config: InferenceConfig):
        if ollama is None:
            raise ImportError("ollama python package is not installed")
        self.model_name = model_name
        self.cfg = inference_config

    def generate_number(self, question: str, images: list[Any]) -> dict[str, Any]:
        prompt = self.cfg.user_template.format(question=question)
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                    "images": [_to_base64_png(img) for img in images],
                },
            ],
            options={
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.max_new_tokens,
            },
        )
        raw = response["message"]["content"].strip()
        return {
            "raw_text": raw,
            "parsed_number": extract_first_number(raw),
            "backend": "ollama",
        }


class HFAttentionRunner:
    """
    Best-effort attention extraction for decoder-style VLMs.
    The returned map is a cross-modal answer-token-to-image-token approximation
    over the multimodal prompt prefix.
    """

    supports_attention = True

    def __init__(
        self,
        model_name: str,
        inference_config: InferenceConfig | None = None,
        device: str | None = None,
        attention_visualization_config: AttentionVisualizationConfig | None = None,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_name = model_name
        self.cfg = inference_config
        self.attention_cfg = attention_visualization_config or AttentionVisualizationConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.supports_attention = _supports_attention_visualization(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation="eager",
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        self.model.eval()
        self._output_projection_cache: dict[int, tuple[torch.Tensor, float]] = {}

    def _build_prompt(self, question: str, num_images: int) -> list[dict]:
        prompt = (
            self.cfg.user_template.replace("{question}", question)
            if self.cfg
            else f"Answer with exactly one Arabic numeral only. Question: {question}"
        )
        content = []
        for _ in range(num_images):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages: list[dict] = []
        if self.cfg and self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_inputs(
        self,
        question: str,
        images: list[Any],
        include_attention_metadata: bool = False,
    ) -> tuple[list[Image.Image], int, dict[str, Any], list[ImageTokenLayout] | None]:
        pil_images = [_to_pil(i) for i in images]
        messages = self._build_prompt(question=question, num_images=len(pil_images))
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=pil_images, text=text, return_tensors="pt")
        image_layouts = self._infer_image_token_layout(pil_images, inputs) if include_attention_metadata else None
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[-1]
        return pil_images, seq_len, inputs, image_layouts

    def _infer_image_token_layout(self, pil_images: list[Image.Image], inputs: dict[str, Any]) -> list[ImageTokenLayout]:
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            raise RuntimeError("The HF processor did not expose `mm_token_type_ids`, so image token positions are unknown.")

        mm_token_type_ids = mm_token_type_ids[0].detach().cpu().numpy()
        mm_positions = np.flatnonzero(mm_token_type_ids == 1)
        if mm_positions.size == 0:
            raise RuntimeError("The HF processor returned no multimodal image token positions.")

        token_counts: list[int]
        grid_shapes: list[tuple[int, int]]
        if "image_grid_thw" in inputs:
            token_counts, grid_shapes = self._compute_qwen_layout_specs(
                image_grid_thw=inputs["image_grid_thw"],
                merge_size=int(getattr(self.processor.image_processor, "merge_size", 1) or 1),
            )
        elif "image_position_ids" in inputs:
            image_inputs = self.processor.image_processor(pil_images)
            token_counts, grid_shapes = self._compute_gemma_layout_specs(
                num_soft_tokens_per_image=image_inputs["num_soft_tokens_per_image"],
                image_position_ids=image_inputs["image_position_ids"],
                pooling_kernel_size=int(
                    getattr(getattr(self.model.config, "vision_config", None), "pooling_kernel_size", 1) or 1
                ),
            )
        else:
            raise RuntimeError(
                "The HF processor did not expose supported image layout metadata. Refusing to create a guessed heatmap."
            )

        if sum(token_counts) != int(mm_positions.size):
            raise RuntimeError(
                "The inferred per-image token counts do not match the multimodal token positions: "
                f"{token_counts} vs {int(mm_positions.size)}."
            )

        layouts: list[ImageTokenLayout] = []
        cursor = 0
        for token_count, grid_shape in zip(token_counts, grid_shapes):
            token_indices = mm_positions[cursor : cursor + token_count]
            if token_indices.size != token_count:
                raise RuntimeError("Failed to slice multimodal image token positions for an input image.")
            layouts.append(ImageTokenLayout(token_indices=token_indices.copy(), grid_shape=grid_shape))
            cursor += token_count
        return layouts

    @staticmethod
    def _compute_qwen_layout_specs(
        image_grid_thw: torch.Tensor | np.ndarray,
        merge_size: int,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        token_counts: list[int] = []
        grid_shapes: list[tuple[int, int]] = []
        for grid in image_grid_thw:
            if hasattr(grid, "tolist"):
                t, h, w = [int(x) for x in grid.tolist()]
            else:
                t, h, w = [int(x) for x in grid]
            pooled_h = _ceil_div(h, max(1, merge_size))
            pooled_w = _ceil_div(w, max(1, merge_size))
            token_counts.append(int(t * pooled_h * pooled_w))
            grid_shapes.append((int(t * pooled_h), int(pooled_w)))
        return token_counts, grid_shapes

    @staticmethod
    def _compute_gemma_layout_specs(
        num_soft_tokens_per_image: list[int],
        image_position_ids: torch.Tensor | np.ndarray,
        pooling_kernel_size: int,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        token_counts = [int(x) for x in num_soft_tokens_per_image]
        grid_shapes: list[tuple[int, int]] = []
        for image_idx, token_count in enumerate(token_counts):
            positions = image_position_ids[image_idx]
            if hasattr(positions, "detach"):
                positions = positions.detach().cpu().numpy()
            valid = positions[(positions[:, 0] >= 0) & (positions[:, 1] >= 0)]
            if valid.size == 0:
                raise RuntimeError("Gemma image positions were empty after filtering padding.")

            max_x = int(valid[:, 0].max()) + 1
            max_y = int(valid[:, 1].max()) + 1
            pooled_w = _ceil_div(max_x, max(1, pooling_kernel_size))
            pooled_h = _ceil_div(max_y, max(1, pooling_kernel_size))
            if pooled_h * pooled_w != token_count:
                raise RuntimeError(
                    "Gemma image position metadata does not match the soft token count: "
                    f"{pooled_h}x{pooled_w} vs {token_count}."
                )
            grid_shapes.append((pooled_h, pooled_w))
        return token_counts, grid_shapes

    def _reshape_attention_values(
        self,
        values: np.ndarray,
        grid_shape: tuple[int, int],
        apply_tam_filter: bool = False,
    ) -> np.ndarray:
        height, width = grid_shape
        if len(values) != height * width:
            raise RuntimeError(
                "Image attention values do not match the inferred grid shape: "
                f"{len(values)} vs {height}x{width}."
            )
        heat = values.astype(np.float32).reshape(height, width)
        if apply_tam_filter:
            heat = self._rank_guassian_filter(heat, self.attention_cfg.tam_filter_size)
        if heat.max() > 0:
            heat = heat / heat.max()
        return heat

    @torch.no_grad()
    def _generate(
        self,
        question: str,
        images: list[Any],
        max_new_tokens: int = 4,
        output_attentions: bool = False,
        target_answer_text: str | None = None,
    ) -> dict[str, Any]:
        pil_images, seq_len, inputs, image_layouts = self._prepare_inputs(
            question=question,
            images=images,
            include_attention_metadata=output_attentions,
        )
        do_sample = bool(self.cfg and self.cfg.temperature > 0)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
            generate_kwargs["top_p"] = self.cfg.top_p
        if output_attentions:
            generate_kwargs["output_hidden_states"] = True
            if self.attention_cfg.method == "attention":
                generate_kwargs["output_attentions"] = True

        out = self.model.generate(
            **inputs,
            **generate_kwargs,
        )
        generated = out.sequences[:, seq_len:]
        generated_token_ids = generated[0].detach().cpu().tolist()
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        result = {
            "raw_text": decoded,
            "parsed_number": extract_first_number(decoded),
            "backend": "huggingface",
        }
        if not output_attentions:
            return result

        if not image_layouts:
            raise RuntimeError("Attention visualization was requested, but image token layout metadata was unavailable.")

        attention_outputs = self._build_attention_outputs(
            prompt_inputs=inputs,
            generation_output=out,
            generated_token_ids=generated_token_ids,
            prompt_seq_len=seq_len,
            image_layouts=image_layouts,
            predicted_answer_text=result["parsed_number"],
            target_answer_text=target_answer_text,
        )
        result["image_heatmaps"] = attention_outputs["image_heatmaps"]
        result["image_sizes"] = [img.size for img in pil_images]
        result["image_spans"] = [
            (int(layout.token_indices[0]), int(layout.token_indices[-1]) + 1) for layout in image_layouts
        ]
        result["attention_token_ids"] = attention_outputs["attention_token_ids"]
        result["attention_tokens"] = attention_outputs["attention_tokens"]
        result["attention_metadata"] = attention_outputs["attention_metadata"]
        return result

    @torch.no_grad()
    def generate_number(self, question: str, images: list[Any], max_new_tokens: int = 4) -> dict[str, Any]:
        return self._generate(question=question, images=images, max_new_tokens=max_new_tokens, output_attentions=False)

    @torch.no_grad()
    def generate_with_attention(
        self,
        question: str,
        images: list[Any],
        max_new_tokens: int = 4,
        target_answer_text: str | None = None,
    ) -> dict[str, Any]:
        if not self.supports_attention:
            raise RuntimeError(f"Attention visualization is not supported for model {self.model_name}.")
        return self._generate(
            question=question,
            images=images,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            target_answer_text=target_answer_text,
        )

    def _decode_token_ids(self, token_ids: list[int]) -> list[str]:
        try:
            return [str(tok) for tok in self.processor.tokenizer.convert_ids_to_tokens(token_ids)]
        except Exception:
            return [str(tok) for tok in token_ids]

    def _decode_token_piece(self, token_id: int) -> str:
        try:
            return str(
                self.processor.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except Exception:
            tokens = self._decode_token_ids([token_id])
            return tokens[0] if tokens else str(token_id)

    def _build_attention_outputs(
        self,
        generation_output: Any,
        prompt_inputs: dict[str, Any],
        generated_token_ids: list[int],
        prompt_seq_len: int,
        image_layouts: list[ImageTokenLayout],
        predicted_answer_text: str | None,
        target_answer_text: str | None = None,
    ) -> dict[str, Any]:
        if self.attention_cfg.method == "attention":
            return self._build_attention_outputs_from_attention(
                generation_output=generation_output,
                prompt_inputs=prompt_inputs,
                generated_token_ids=generated_token_ids,
                prompt_seq_len=prompt_seq_len,
                image_layouts=image_layouts,
                predicted_answer_text=predicted_answer_text,
                target_answer_text=target_answer_text,
            )

        try:
            return self._build_tam_outputs(
                generation_output=generation_output,
                prompt_inputs=prompt_inputs,
                generated_token_ids=generated_token_ids,
                prompt_seq_len=prompt_seq_len,
                image_layouts=image_layouts,
                predicted_answer_text=predicted_answer_text,
                target_answer_text=target_answer_text,
            )
        except Exception as tam_exc:
            try:
                attention_outputs = self._build_attention_outputs_from_attention(
                    generation_output=generation_output,
                    prompt_inputs=prompt_inputs,
                    generated_token_ids=generated_token_ids,
                    prompt_seq_len=prompt_seq_len,
                    image_layouts=image_layouts,
                    predicted_answer_text=predicted_answer_text,
                    target_answer_text=target_answer_text,
                )
                attention_outputs["attention_metadata"]["requested_visualization_method"] = "tam"
                attention_outputs["attention_metadata"]["tam_fallback_reason"] = str(tam_exc)
                return attention_outputs
            except Exception as attention_exc:
                raise RuntimeError(
                    "TAM visualization failed and the legacy attention fallback also failed: "
                    f"{tam_exc}; {attention_exc}"
                ) from attention_exc

    def _build_tam_outputs(
        self,
        generation_output: Any,
        prompt_inputs: dict[str, Any],
        generated_token_ids: list[int],
        prompt_seq_len: int,
        image_layouts: list[ImageTokenLayout],
        predicted_answer_text: str | None,
        target_answer_text: str | None = None,
    ) -> dict[str, Any]:
        answer_span = self._select_answer_token_span(
            generated_token_ids=generated_token_ids,
            target_answer_text=target_answer_text,
            predicted_answer_text=predicted_answer_text,
        )
        all_image_token_indices = np.concatenate([layout.token_indices for layout in image_layouts]).astype(np.int64, copy=False)
        step_hidden_states = self._extract_generation_step_hidden_states(generation_output)
        if len(step_hidden_states) < len(generated_token_ids):
            raise RuntimeError(
                "The generation backend returned fewer hidden-state rounds than generated tokens: "
                f"{len(step_hidden_states)} vs {len(generated_token_ids)}."
            )

        prompt_records = self._build_prompt_token_tam_records(
            prompt_input_ids=prompt_inputs["input_ids"][0].detach().cpu().tolist(),
            prompt_hidden_states=step_hidden_states[0],
            prompt_seq_len=prompt_seq_len,
            image_token_indices=all_image_token_indices,
        )
        generated_records = self._build_generated_token_tam_records(
            generated_token_ids=generated_token_ids,
            step_hidden_states=step_hidden_states,
            prompt_records=prompt_records,
            prompt_seq_len=prompt_seq_len,
            image_token_indices=all_image_token_indices,
        )
        answer_maps = [
            generated_records[rel_pos].image_scores.astype(np.float32, copy=False)
            for rel_pos in answer_span["relative_positions"]
            if rel_pos < len(generated_records)
        ]
        if not answer_maps:
            raise RuntimeError("No answer-token TAM map was available for the matched answer span.")

        aggregated_image_scores = np.stack(
            [self._normalize_array(image_scores) for image_scores in answer_maps],
            axis=0,
        ).mean(axis=0)

        metadata: dict[str, Any] = {
            "visualization_method": "tam",
            "span_source": answer_span["source"],
            "target_answer_text": target_answer_text or None,
            "predicted_answer_text": predicted_answer_text or None,
            "answer_relative_positions": [int(pos) for pos in answer_span["relative_positions"]],
            "tam_filter_size": int(self.attention_cfg.tam_filter_size),
            "tam_prompt_token_count": len(prompt_records),
            "tam_generated_token_count": len(generated_records),
        }
        image_heatmaps = self._split_image_scores_into_heatmaps(
            image_scores=aggregated_image_scores,
            image_layouts=image_layouts,
            apply_tam_filter=True,
        )

        return {
            "image_heatmaps": image_heatmaps,
            "attention_token_ids": answer_span["token_ids"],
            "attention_tokens": self._decode_token_ids(answer_span["token_ids"]),
            "attention_metadata": metadata,
        }

    def _build_attention_outputs_from_attention(
        self,
        generation_output: Any,
        prompt_inputs: dict[str, Any],
        generated_token_ids: list[int],
        prompt_seq_len: int,
        image_layouts: list[ImageTokenLayout],
        predicted_answer_text: str | None,
        target_answer_text: str | None = None,
    ) -> dict[str, Any]:
        answer_span = self._select_answer_token_span(
            generated_token_ids=generated_token_ids,
            target_answer_text=target_answer_text,
            predicted_answer_text=predicted_answer_text,
        )
        all_image_token_indices = np.concatenate([layout.token_indices for layout in image_layouts]).astype(np.int64, copy=False)
        metadata: dict[str, Any] = {
            "visualization_method": "attention",
            "span_source": answer_span["source"],
            "target_answer_text": target_answer_text or None,
            "predicted_answer_text": predicted_answer_text or None,
            "answer_relative_positions": [int(pos) for pos in answer_span["relative_positions"]],
        }
        answer_absolute_positions = [int(prompt_seq_len + pos) for pos in answer_span["relative_positions"]]

        try:
            prompt_attention, aggregation_metadata = self._build_full_sequence_attention_map(
                prompt_inputs=prompt_inputs,
                full_sequences=generation_output.sequences,
                prompt_seq_len=prompt_seq_len,
                answer_token_positions=answer_absolute_positions,
                image_token_indices=all_image_token_indices,
            )
            metadata["attention_backend"] = "full_sequence"
        except Exception as exc:
            prompt_attention, aggregation_metadata = self._extract_answer_token_attention(
                generation_output=generation_output,
                prompt_seq_len=prompt_seq_len,
                answer_relative_positions=answer_span["relative_positions"],
                image_token_indices=all_image_token_indices,
            )
            metadata["attention_backend"] = "generation_steps"
            metadata["fallback_reason"] = str(exc)

        metadata.update(aggregation_metadata)
        image_heatmaps: list[np.ndarray] = []
        for layout in image_layouts:
            image_heatmaps.append(
                self._reshape_attention_values(
                    prompt_attention[layout.token_indices],
                    grid_shape=layout.grid_shape,
                )
            )

        return {
            "image_heatmaps": image_heatmaps,
            "attention_token_ids": answer_span["token_ids"],
            "attention_tokens": self._decode_token_ids(answer_span["token_ids"]),
            "attention_metadata": metadata,
        }

    def _build_prompt_token_tam_records(
        self,
        prompt_input_ids: list[int],
        prompt_hidden_states: torch.Tensor,
        prompt_seq_len: int,
        image_token_indices: np.ndarray,
    ) -> list[TokenActivationRecord]:
        prompt_positions = self._select_prompt_text_positions(
            prompt_input_ids=prompt_input_ids,
            prompt_seq_len=prompt_seq_len,
            image_token_indices=image_token_indices,
        )
        prompt_records: list[TokenActivationRecord] = []
        for sequence_position in prompt_positions:
            token_id = int(prompt_input_ids[sequence_position])
            token_piece = self._decode_token_piece(token_id)
            scores = self._compute_prompt_class_scores(
                prompt_hidden_states=prompt_hidden_states,
                class_token_id=token_id,
            )
            image_scores = scores[image_token_indices]
            if prompt_records:
                prev_positions = np.array([record.sequence_position for record in prompt_records], dtype=np.int64)
                prev_scores = scores[prev_positions]
                image_scores = self._apply_tam_eci(
                    current_image_scores=image_scores,
                    current_token_piece=token_piece,
                    previous_records=prompt_records,
                    previous_text_scores=prev_scores,
                )
            prompt_records.append(
                TokenActivationRecord(
                    sequence_position=int(sequence_position),
                    token_id=token_id,
                    token_piece=token_piece,
                    image_scores=image_scores.astype(np.float32, copy=False),
                )
            )
        return prompt_records

    def _build_generated_token_tam_records(
        self,
        generated_token_ids: list[int],
        step_hidden_states: list[torch.Tensor],
        prompt_records: list[TokenActivationRecord],
        prompt_seq_len: int,
        image_token_indices: np.ndarray,
    ) -> list[TokenActivationRecord]:
        generated_records: list[TokenActivationRecord] = []
        for rel_pos, token_id in enumerate(generated_token_ids):
            token_piece = self._decode_token_piece(int(token_id))
            scores = self._compute_generation_class_scores(
                step_hidden_states=step_hidden_states,
                class_token_id=int(token_id),
                max_step_index=int(rel_pos),
            )
            image_scores = scores[image_token_indices]
            previous_records = prompt_records + generated_records
            if previous_records:
                prev_positions = np.array([record.sequence_position for record in previous_records], dtype=np.int64)
                prev_scores = scores[prev_positions]
                image_scores = self._apply_tam_eci(
                    current_image_scores=image_scores,
                    current_token_piece=token_piece,
                    previous_records=previous_records,
                    previous_text_scores=prev_scores,
                )
            generated_records.append(
                TokenActivationRecord(
                    sequence_position=int(prompt_seq_len + rel_pos),
                    token_id=int(token_id),
                    token_piece=token_piece,
                    image_scores=image_scores.astype(np.float32, copy=False),
                )
            )
        return generated_records

    def _select_prompt_text_positions(
        self,
        prompt_input_ids: list[int],
        prompt_seq_len: int,
        image_token_indices: np.ndarray,
    ) -> list[int]:
        image_index_set = {int(idx) for idx in image_token_indices.tolist()}
        special_token_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])
        return [
            idx
            for idx in range(prompt_seq_len)
            if idx not in image_index_set and int(prompt_input_ids[idx]) not in special_token_ids
        ]

    def _extract_generation_step_hidden_states(self, generation_output: Any) -> list[torch.Tensor]:
        step_hidden_states = getattr(generation_output, "hidden_states", None)
        if not step_hidden_states:
            raise RuntimeError("The HF generation backend did not return hidden states for TAM visualization.")

        last_hidden_states: list[torch.Tensor] = []
        for step_idx, step_hidden in enumerate(step_hidden_states):
            if not step_hidden:
                raise RuntimeError(f"Hidden-state step {step_idx} was empty.")
            last_hidden = step_hidden[-1]
            if isinstance(last_hidden, (list, tuple)):
                if not last_hidden:
                    raise RuntimeError(f"Hidden-state step {step_idx} exposed an empty final layer.")
                last_hidden = last_hidden[0]
            if last_hidden.ndim == 3:
                last_hidden = last_hidden[0]
            if last_hidden.ndim != 2:
                raise RuntimeError(f"Unsupported hidden-state tensor shape: {tuple(last_hidden.shape)}")
            last_hidden_states.append(last_hidden.detach().to(torch.float32).cpu())
        return last_hidden_states

    def _get_output_projection_vector(self, class_token_id: int) -> tuple[torch.Tensor, float]:
        cached = self._output_projection_cache.get(int(class_token_id))
        if cached is not None:
            return cached

        output_embeddings = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        if output_embeddings is None:
            output_embeddings = getattr(self.model, "lm_head", None)
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise RuntimeError("The HF model does not expose an output embedding matrix for TAM visualization.")

        weight_vector = output_embeddings.weight[int(class_token_id)].detach().to(torch.float32).cpu()
        bias = getattr(output_embeddings, "bias", None)
        bias_value = float(bias[int(class_token_id)].detach().cpu()) if bias is not None else 0.0
        cached = (weight_vector, bias_value)
        self._output_projection_cache[int(class_token_id)] = cached
        return cached

    def _compute_prompt_class_scores(
        self,
        prompt_hidden_states: torch.Tensor,
        class_token_id: int,
    ) -> np.ndarray:
        weight_vector, bias_value = self._get_output_projection_vector(int(class_token_id))
        scores = prompt_hidden_states @ weight_vector
        if bias_value:
            scores = scores + bias_value
        return scores.detach().cpu().numpy().astype(np.float32).clip(min=0.0)

    def _compute_generation_class_scores(
        self,
        step_hidden_states: list[torch.Tensor],
        class_token_id: int,
        max_step_index: int,
    ) -> np.ndarray:
        weight_vector, bias_value = self._get_output_projection_vector(int(class_token_id))
        per_step_scores: list[np.ndarray] = []
        for step_hidden in step_hidden_states[: max_step_index + 1]:
            step_scores = step_hidden @ weight_vector
            if bias_value:
                step_scores = step_scores + bias_value
            per_step_scores.append(step_scores.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(per_step_scores, axis=0).clip(min=0.0)

    def _apply_tam_eci(
        self,
        current_image_scores: np.ndarray,
        current_token_piece: str,
        previous_records: list[TokenActivationRecord],
        previous_text_scores: np.ndarray,
    ) -> np.ndarray:
        if not previous_records:
            return current_image_scores

        eligible_indices = [
            idx
            for idx, record in enumerate(previous_records)
            if record.token_piece != current_token_piece and float(previous_text_scores[idx]) > 0.0
        ]
        if not eligible_indices:
            return current_image_scores

        weights = previous_text_scores[eligible_indices].astype(np.float32, copy=False)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            return current_image_scores

        interference = np.stack([previous_records[idx].image_scores for idx in eligible_indices], axis=0)
        normalized_weights = weights / (weight_sum + 1e-8)
        interference_map = (interference * normalized_weights[:, None]).sum(axis=0)
        scaled_interference = self._least_squares_scale(current_image_scores, interference_map)
        return np.clip(current_image_scores - interference_map * scaled_interference, a_min=0.0, a_max=None)

    @staticmethod
    def _least_squares_scale(map1: np.ndarray, map2: np.ndarray) -> float:
        denominator = float(np.dot(map2, map2))
        if denominator <= 1e-8:
            return 0.0
        return float(np.dot(map1, map2) / denominator)

    @staticmethod
    def _normalize_array(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        min_value = float(values.min()) if values.size else 0.0
        max_value = float(values.max()) if values.size else 0.0
        if max_value - min_value <= 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return (values - min_value) / (max_value - min_value)

    def _split_image_scores_into_heatmaps(
        self,
        image_scores: np.ndarray,
        image_layouts: list[ImageTokenLayout],
        apply_tam_filter: bool,
    ) -> list[np.ndarray]:
        heatmaps: list[np.ndarray] = []
        cursor = 0
        for layout in image_layouts:
            token_count = int(layout.token_indices.size)
            values = image_scores[cursor : cursor + token_count]
            if len(values) != token_count:
                raise RuntimeError("Failed to slice per-image TAM scores from the concatenated image score vector.")
            heatmaps.append(
                self._reshape_attention_values(
                    values,
                    grid_shape=layout.grid_shape,
                    apply_tam_filter=apply_tam_filter,
                )
            )
            cursor += token_count
        return heatmaps

    @staticmethod
    def _rank_guassian_filter(image_scores: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return image_scores.astype(np.float32, copy=False)

        image_scores = image_scores.astype(np.float32, copy=False)
        pad_width = kernel_size // 2
        padded = np.pad(image_scores, pad_width=pad_width, mode="reflect")
        axis = np.arange(kernel_size**2, dtype=np.float32) - (kernel_size**2 // 2)
        filtered = np.zeros_like(image_scores, dtype=np.float32)

        for row in range(image_scores.shape[0]):
            for col in range(image_scores.shape[1]):
                window = padded[row : row + kernel_size, col : col + kernel_size].reshape(-1)
                sorted_window = np.sort(window)
                mean_value = float(sorted_window.mean())
                if mean_value <= 0:
                    filtered[row, col] = 0.0
                    continue
                sigma = max(float(sorted_window.std()) / mean_value, 1e-4)
                kernel = np.exp(-(axis**2) / (2.0 * sigma**2))
                kernel = kernel / max(float(kernel.sum()), 1e-8)
                filtered[row, col] = float((sorted_window * kernel).sum())
        return filtered

    def _select_answer_token_span(
        self,
        generated_token_ids: list[int],
        target_answer_text: str | None,
        predicted_answer_text: str | None,
    ) -> dict[str, Any]:
        special_token_ids = set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])
        kept_positions = [idx for idx, token_id in enumerate(generated_token_ids) if int(token_id) not in special_token_ids]
        if not kept_positions:
            raise RuntimeError("No generated non-special answer token was found for attention visualization.")

        kept_token_ids = [int(generated_token_ids[idx]) for idx in kept_positions]
        candidate_order: list[tuple[str, str | None]] = []
        if self.attention_cfg.span_source == "prediction_only":
            candidate_order.append(("predicted_answer", predicted_answer_text))
        elif self.attention_cfg.span_source == "target_only":
            candidate_order.extend([("target_answer", target_answer_text), ("predicted_answer", predicted_answer_text)])
        else:
            candidate_order.extend([("target_answer", target_answer_text), ("predicted_answer", predicted_answer_text)])

        for source_name, source_text in candidate_order:
            span = self._find_matching_generated_span(kept_token_ids, source_text)
            if not span:
                continue
            relative_positions = [kept_positions[idx] for idx in span]
            return {
                "source": source_name,
                "relative_positions": relative_positions,
                "token_ids": [int(generated_token_ids[idx]) for idx in relative_positions],
            }

        numeric_positions = [
            idx for idx in kept_positions if re.search(r"\d", normalize_numeric_text(self._decode_token_piece(int(generated_token_ids[idx]))))
        ]
        if numeric_positions:
            contiguous_groups = self._group_contiguous_indices(numeric_positions)
            chosen_group = max(contiguous_groups, key=lambda group: (len(group), group[-1]))
            return {
                "source": "numeric_fallback",
                "relative_positions": chosen_group,
                "token_ids": [int(generated_token_ids[idx]) for idx in chosen_group],
            }

        return {
            "source": "non_special_fallback",
            "relative_positions": [kept_positions[-1]],
            "token_ids": [int(generated_token_ids[kept_positions[-1]])],
        }

    def _find_matching_generated_span(self, token_ids: list[int], target_text: str | None) -> list[int] | None:
        target_norm = normalize_numeric_text(target_text)
        if not target_norm:
            return None

        pieces = [self._decode_token_piece(int(token_id)) for token_id in token_ids]
        best_span: list[int] | None = None
        best_key: tuple[int, int] | None = None
        for start_idx in range(len(token_ids)):
            combined = ""
            for end_idx in range(start_idx, len(token_ids)):
                combined += pieces[end_idx]
                combined_norm = normalize_numeric_text(combined)
                if not combined_norm:
                    continue
                if combined_norm == target_norm:
                    candidate = list(range(start_idx, end_idx + 1))
                    candidate_key = (len(candidate), -start_idx)
                    if best_key is None or candidate_key < best_key:
                        best_span = candidate
                        best_key = candidate_key
                compact = combined_norm.replace(" ", "")
                target_compact = target_norm.replace(" ", "")
                if len(compact) > len(target_compact) + 4 and target_compact not in compact:
                    break
        return best_span

    @staticmethod
    def _group_contiguous_indices(indices: list[int]) -> list[list[int]]:
        if not indices:
            return []
        groups: list[list[int]] = [[indices[0]]]
        for idx in indices[1:]:
            if idx == groups[-1][-1] + 1:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        return groups

    def _build_full_sequence_inputs(self, prompt_inputs: dict[str, Any], full_sequences: torch.Tensor) -> dict[str, Any]:
        prompt_seq_len = int(prompt_inputs["input_ids"].shape[-1])
        full_sequences = full_sequences.to(self.model.device)
        extra_len = int(full_sequences.shape[-1] - prompt_seq_len)
        if extra_len < 0:
            raise RuntimeError("The generated sequence is shorter than the prompt sequence.")

        model_inputs: dict[str, Any] = {"input_ids": full_sequences}
        for key, value in prompt_inputs.items():
            if key == "input_ids":
                continue
            if not torch.is_tensor(value):
                model_inputs[key] = value
                continue
            if value.ndim >= 2 and value.shape[0] == full_sequences.shape[0] and value.shape[-1] == prompt_seq_len:
                if key == "attention_mask":
                    extension = torch.ones((*value.shape[:-1], extra_len), dtype=value.dtype, device=value.device)
                    model_inputs[key] = torch.cat([value, extension], dim=-1)
                elif key == "position_ids":
                    model_inputs[key] = self._extend_position_ids(value, extra_len)
                elif key == "cache_position":
                    continue
                else:
                    extension = torch.zeros((*value.shape[:-1], extra_len), dtype=value.dtype, device=value.device)
                    model_inputs[key] = torch.cat([value, extension], dim=-1)
            else:
                model_inputs[key] = value
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(full_sequences, dtype=torch.long, device=full_sequences.device)
        return model_inputs

    @staticmethod
    def _extend_position_ids(position_ids: torch.Tensor, extra_len: int) -> torch.Tensor:
        if extra_len <= 0:
            return position_ids
        increments = torch.arange(1, extra_len + 1, device=position_ids.device, dtype=position_ids.dtype)
        view_shape = (1,) * (position_ids.ndim - 1) + (extra_len,)
        extension = position_ids[..., -1:] + increments.view(view_shape)
        return torch.cat([position_ids, extension], dim=-1)

    def _build_full_sequence_attention_map(
        self,
        prompt_inputs: dict[str, Any],
        full_sequences: torch.Tensor,
        prompt_seq_len: int,
        answer_token_positions: list[int],
        image_token_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        full_inputs = self._build_full_sequence_inputs(prompt_inputs=prompt_inputs, full_sequences=full_sequences)
        outputs = self.model(
            **full_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        layer_attentions = getattr(outputs, "attentions", None) or getattr(outputs, "decoder_attentions", None)
        if not layer_attentions:
            raise RuntimeError("No full-sequence attention tensors were returned by the HF forward pass.")

        answer_positions = [int(pos) for pos in answer_token_positions]
        return self._aggregate_prompt_attention_from_layers(
            layer_attentions=layer_attentions,
            query_positions=answer_positions,
            prompt_seq_len=prompt_seq_len,
            image_token_indices=image_token_indices,
            allow_rollout=True,
        )

    def _extract_answer_token_attention(
        self,
        generation_output: Any,
        prompt_seq_len: int,
        answer_relative_positions: list[int],
        image_token_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        step_attentions = getattr(generation_output, "attentions", None) or getattr(
            generation_output,
            "decoder_attentions",
            None,
        )
        if not step_attentions:
            raise RuntimeError("No attention tensors were returned by the HF generation backend.")

        step_offset = 1 if self._get_step_query_length(step_attentions[0]) > 1 else 0
        selected_step_indices = [
            step_offset + rel_pos for rel_pos in answer_relative_positions if step_offset + rel_pos < len(step_attentions)
        ]
        if not selected_step_indices:
            raise RuntimeError(
                "No answer-span token has an available generation-step attention query. "
                "This usually happens when the model emitted only one token and stopped immediately."
            )

        return self._aggregate_prompt_attention_from_generation_steps(
            step_attentions=step_attentions,
            step_indices=selected_step_indices,
            prompt_seq_len=prompt_seq_len,
            image_token_indices=image_token_indices,
        )

    def _aggregate_prompt_attention_from_generation_steps(
        self,
        step_attentions: Any,
        step_indices: list[int],
        prompt_seq_len: int,
        image_token_indices: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        num_layers = len(step_attentions[step_indices[0]])
        layer_indices = self._select_layer_indices(num_layers)
        if not layer_indices:
            raise RuntimeError("No attention layer was selected for generation-step visualization.")

        per_layer_maps: list[torch.Tensor] = []
        layer_metadata: list[dict[str, Any]] = []
        for layer_idx in layer_indices:
            per_step_head_maps: list[torch.Tensor] = []
            for step_idx in step_indices:
                layer_tensor = self._extract_layer_attention_tensor(step_attentions[step_idx][layer_idx])
                if layer_tensor.shape[-2] < 1:
                    raise RuntimeError(f"HF generation step {step_idx} exposed no query position for layer {layer_idx}.")
                available_prompt_len = min(prompt_seq_len, int(layer_tensor.shape[-1]))
                per_step_head_maps.append(layer_tensor[:, -1, :available_prompt_len])
            head_maps = torch.stack(per_step_head_maps, dim=0).mean(dim=0)
            selected_heads, head_stats = self._select_image_centric_heads(
                head_maps=head_maps,
                image_token_indices=image_token_indices,
            )
            per_layer_maps.append(
                self._pad_prompt_attention(
                    head_maps[selected_heads].mean(dim=0),
                    prompt_seq_len=prompt_seq_len,
                )
            )
            layer_metadata.append(
                {
                    "layer_index": int(layer_idx),
                    "selected_heads": [int(head_idx) for head_idx in selected_heads],
                    "head_scores": [float(head_stats["scores"][head_idx]) for head_idx in selected_heads],
                    "head_image_mass": [float(head_stats["image_mass"][head_idx]) for head_idx in selected_heads],
                    "head_localization": [float(head_stats["localization"][head_idx]) for head_idx in selected_heads],
                }
            )

        prompt_attention = torch.stack(per_layer_maps, dim=0).mean(dim=0).to(torch.float32)
        prompt_attention = prompt_attention.detach().cpu().numpy().astype(np.float32)
        if prompt_attention.max() > 0:
            prompt_attention = prompt_attention / prompt_attention.max()
        return prompt_attention, {
            "layer_indices": [int(idx) for idx in layer_indices],
            "layer_strategy": "mean",
            "step_indices": [int(idx) for idx in step_indices],
            "selected_layers": layer_metadata,
        }

    def _aggregate_prompt_attention_from_layers(
        self,
        layer_attentions: Any,
        query_positions: list[int],
        prompt_seq_len: int,
        image_token_indices: np.ndarray,
        allow_rollout: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        layer_indices = self._select_layer_indices(len(layer_attentions))
        if not layer_indices:
            raise RuntimeError("No attention layer was selected for visualization.")

        per_layer_maps: list[torch.Tensor] = []
        rollout_matrices: list[torch.Tensor] = []
        layer_metadata: list[dict[str, Any]] = []
        effective_strategy = self.attention_cfg.layer_strategy if allow_rollout else "mean"

        for layer_idx in layer_indices:
            layer_tensor = self._extract_layer_attention_tensor(layer_attentions[layer_idx])
            head_maps = self._compute_head_query_maps(
                layer_tensor=layer_tensor,
                query_positions=query_positions,
                prompt_seq_len=prompt_seq_len,
            )
            selected_heads, head_stats = self._select_image_centric_heads(
                head_maps=head_maps,
                image_token_indices=image_token_indices,
            )
            per_layer_maps.append(
                self._pad_prompt_attention(
                    head_maps[selected_heads].mean(dim=0),
                    prompt_seq_len=prompt_seq_len,
                )
            )
            layer_metadata.append(
                {
                    "layer_index": int(layer_idx),
                    "selected_heads": [int(head_idx) for head_idx in selected_heads],
                    "head_scores": [float(head_stats["scores"][head_idx]) for head_idx in selected_heads],
                    "head_image_mass": [float(head_stats["image_mass"][head_idx]) for head_idx in selected_heads],
                    "head_localization": [float(head_stats["localization"][head_idx]) for head_idx in selected_heads],
                }
            )

            if effective_strategy == "rollout":
                layer_matrix = layer_tensor[selected_heads].mean(dim=0)
                rollout_matrices.append(self._prepare_rollout_matrix(layer_matrix))

        if effective_strategy == "rollout":
            rollout = torch.eye(
                rollout_matrices[0].shape[-1],
                dtype=rollout_matrices[0].dtype,
                device=rollout_matrices[0].device,
            )
            for layer_matrix in rollout_matrices:
                rollout = layer_matrix @ rollout
            prompt_attention = rollout[query_positions].mean(dim=0)[:prompt_seq_len]
        else:
            prompt_attention = torch.stack(per_layer_maps, dim=0).mean(dim=0)

        prompt_attention = prompt_attention.to(torch.float32).detach().cpu().numpy().astype(np.float32)
        if prompt_attention.max() > 0:
            prompt_attention = prompt_attention / prompt_attention.max()
        return prompt_attention, {
            "layer_indices": [int(idx) for idx in layer_indices],
            "layer_strategy": effective_strategy,
            "query_positions": [int(idx) for idx in query_positions],
            "selected_layers": layer_metadata,
        }

    def _select_layer_indices(self, num_layers: int) -> list[int]:
        if num_layers <= 0:
            return []
        start_idx = min(num_layers - 1, int(np.floor((num_layers - 1) * self.attention_cfg.layer_start_ratio)))
        available = list(range(start_idx, num_layers))
        if len(available) <= self.attention_cfg.layer_count:
            return available

        sampled = np.linspace(start_idx, num_layers - 1, num=self.attention_cfg.layer_count)
        selected: list[int] = []
        for idx in sampled:
            rounded = int(round(float(idx)))
            if rounded not in selected:
                selected.append(rounded)
        for idx in available:
            if len(selected) >= self.attention_cfg.layer_count:
                break
            if idx not in selected:
                selected.append(idx)
        return sorted(selected[: self.attention_cfg.layer_count])

    def _extract_layer_attention_tensor(self, layer_attention: Any) -> torch.Tensor:
        if isinstance(layer_attention, (list, tuple)):
            if not layer_attention:
                raise RuntimeError("An HF attention layer was empty.")
            layer_attention = layer_attention[0]
        if layer_attention.ndim == 4:
            return layer_attention[0]
        if layer_attention.ndim == 3:
            return layer_attention
        raise RuntimeError(f"Unsupported attention tensor shape: {tuple(layer_attention.shape)}")

    def _compute_head_query_maps(
        self,
        layer_tensor: torch.Tensor,
        query_positions: list[int],
        prompt_seq_len: int,
    ) -> torch.Tensor:
        max_query_idx = max(query_positions)
        if max_query_idx >= layer_tensor.shape[-2]:
            raise RuntimeError(
                "The requested answer-token query index exceeds the available attention query length: "
                f"{max_query_idx} vs {layer_tensor.shape[-2]}."
            )
        query_index = torch.as_tensor(query_positions, device=layer_tensor.device, dtype=torch.long)
        query_rows = layer_tensor.index_select(dim=-2, index=query_index)
        return query_rows[..., :prompt_seq_len].mean(dim=1)

    def _select_image_centric_heads(
        self,
        head_maps: torch.Tensor,
        image_token_indices: np.ndarray,
    ) -> tuple[list[int], dict[str, np.ndarray]]:
        if image_token_indices.size == 0:
            raise RuntimeError("No image token indices were available for head selection.")
        valid_image_token_indices = image_token_indices[
            (image_token_indices >= 0) & (image_token_indices < int(head_maps.shape[-1]))
        ]
        if valid_image_token_indices.size == 0:
            raise RuntimeError(
                "No image token index fell inside the available attention key range: "
                f"{int(image_token_indices.min())}-{int(image_token_indices.max())} vs {int(head_maps.shape[-1])}."
            )

        image_index = torch.as_tensor(valid_image_token_indices, device=head_maps.device, dtype=torch.long)
        image_maps = head_maps.index_select(dim=-1, index=image_index)
        image_mass = image_maps.sum(dim=-1).float()
        image_token_count = image_maps.shape[-1]

        if image_token_count == 1:
            localization = torch.ones_like(image_mass, dtype=torch.float32)
        else:
            probs = image_maps / image_maps.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
            localization = (1.0 - (entropy / float(np.log(image_token_count)))).float()
        scores = (image_mass * localization).float()

        ranked_heads = torch.argsort(scores, descending=True).detach().cpu().tolist()
        eligible = [
            int(head_idx)
            for head_idx in ranked_heads
            if float(image_mass[head_idx].detach().cpu()) >= self.attention_cfg.head_min_image_mass
        ]
        top_k = min(self.attention_cfg.head_top_k, head_maps.shape[0])
        selected_heads = eligible[:top_k] if eligible else ranked_heads[:top_k]
        return selected_heads, {
            "scores": scores.detach().cpu().to(torch.float32).numpy(),
            "image_mass": image_mass.detach().cpu().to(torch.float32).numpy(),
            "localization": localization.detach().cpu().to(torch.float32).numpy(),
        }

    def _prepare_rollout_matrix(self, layer_matrix: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(layer_matrix.shape[-1], dtype=layer_matrix.dtype, device=layer_matrix.device)
        rollout_matrix = layer_matrix + identity
        rollout_matrix = rollout_matrix / rollout_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return rollout_matrix

    def _pad_prompt_attention(self, prompt_attention: torch.Tensor, prompt_seq_len: int) -> torch.Tensor:
        prompt_attention = prompt_attention.to(torch.float32)
        current_len = int(prompt_attention.shape[-1])
        if current_len == prompt_seq_len:
            return prompt_attention
        if current_len > prompt_seq_len:
            return prompt_attention[:prompt_seq_len]
        padded = torch.zeros(prompt_seq_len, dtype=prompt_attention.dtype, device=prompt_attention.device)
        padded[:current_len] = prompt_attention
        return padded

    def _get_step_query_length(self, step_attention: Any) -> int:
        last_layer_attn = step_attention[-1]
        if isinstance(last_layer_attn, (list, tuple)):
            if not last_layer_attn:
                raise RuntimeError("The final HF attention layer is empty.")
            last_layer_attn = last_layer_attn[0]
        return int(last_layer_attn.shape[-2])

    def _extract_step_last_query_attention(self, step_attention: Any) -> torch.Tensor:
        last_layer_attn = step_attention[-1]
        if isinstance(last_layer_attn, (list, tuple)):
            if not last_layer_attn:
                raise RuntimeError("The final HF attention layer is empty.")
            last_layer_attn = last_layer_attn[0]

        if last_layer_attn.ndim == 4:
            return last_layer_attn[0, :, -1, :]
        if last_layer_attn.ndim == 3:
            return last_layer_attn[:, -1, :]

        raise RuntimeError(f"Unsupported attention tensor shape: {tuple(last_layer_attn.shape)}")

def build_model_runner(
    model_name: str,
    inference_config: InferenceConfig | None = None,
    device: str | None = None,
    attention_visualization_config: AttentionVisualizationConfig | None = None,
) -> Any:
    return HFAttentionRunner(
        model_name,
        inference_config=inference_config,
        device=device,
        attention_visualization_config=attention_visualization_config,
    )
