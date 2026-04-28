from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_anchor.utils import extract_first_number


FASTVLM_IMAGE_TOKEN_INDEX = -200


def _to_pil(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, (str, Path)):
        return Image.open(image_like).convert("RGB")
    if hasattr(image_like, "convert"):
        return image_like.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_like)}")


@dataclass
class InferenceConfig:
    system_prompt: str
    user_template: str
    temperature: float
    top_p: float
    max_new_tokens: int


class _BaseRunner:
    """Shared generation + token-info extraction for all VLM runners."""

    model_name: str
    cfg: InferenceConfig | None
    device: str

    def _resolve_device(self, device: str | None) -> str:
        return device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.device.startswith("cuda") else torch.float32

    def _format_user_text(self, question: str) -> str:
        if self.cfg:
            return self.cfg.user_template.replace("{question}", question)
        return f"Answer with exactly one Arabic numeral only. Question: {question}"

    def _build_generate_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        do_sample = bool(self.cfg and self.cfg.temperature > 0)
        kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if do_sample:
            kw["temperature"] = self.cfg.temperature
            kw["top_p"] = self.cfg.top_p
        return kw

    THINKING_MARKER = "</think>"

    def _summarize_generation(
        self,
        *,
        generated_ids: torch.LongTensor,
        scores: tuple[torch.Tensor, ...],
        tokenizer: Any,
        decoded: str,
    ) -> dict[str, Any]:
        gen_ids = generated_ids[0].tolist()
        token_info: list[dict[str, Any]] = []
        for step, step_logits in enumerate(scores or ()):
            if step >= len(gen_ids):
                break
            token_id = int(gen_ids[step])
            logits_vec = step_logits[0].float()
            probs = torch.softmax(logits_vec, dim=-1)
            token_info.append(
                {
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                    "logit": float(logits_vec[token_id].item()),
                    "probability": float(probs[token_id].item()),
                }
            )

        # Reasoning-mode VLMs emit `<think>...</think>\n\n<final>` — keep the
        # full trace in raw_text but parse the answer from the post-trace tail
        # so trace-internal numerals (e.g. "let me count: 1, 2, 3") don't get
        # picked up as the prediction. For non-thinking models the marker is
        # absent, post_trace == decoded, and behaviour is unchanged.
        marker = self.THINKING_MARKER
        marker_idx = decoded.rfind(marker)
        post_trace = decoded[marker_idx + len(marker):] if marker_idx >= 0 else decoded
        answer_number = extract_first_number(post_trace)

        # Match answer token by reverse iteration so we lock onto the final
        # answer token rather than an earlier same-valued numeral inside the
        # trace. BPE tokenizers don't allow safe text-position arithmetic, so
        # we trust generation order: the final answer is the latest match.
        answer_token = next(
            (
                t for t in reversed(token_info)
                if answer_number and extract_first_number(t["token_text"]) == answer_number
            ),
            None,
        )

        return {
            "raw_text": decoded,
            "parsed_number": answer_number,
            "backend": "huggingface",
            "token_info": token_info,
            "answer_token_logit": answer_token["logit"] if answer_token else None,
            "answer_token_probability": answer_token["probability"] if answer_token else None,
            "answer_token_id": answer_token["token_id"] if answer_token else None,
            "answer_token_text": answer_token["token_text"] if answer_token else None,
            # Reasoning-mode bookkeeping. `thinking_marker_present` is False
            # whenever max_new_tokens cut off the trace before `</think>`,
            # so post-trace parsing fell back to the full decoded string —
            # the prediction may then be a trace-internal numeral. Always
            # filter on this in any thinking-mode analysis.
            "thinking_marker_present": marker_idx >= 0,
            "n_generated_tokens": len(gen_ids),
        }


class HFAttentionRunner(_BaseRunner):
    """Default path: AutoProcessor + AutoModelForImageTextToText with chat template."""

    def __init__(
        self,
        model_name: str,
        inference_config: InferenceConfig | None = None,
        device: str | None = None,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "sdpa"
        self.model.eval()

    def _build_prompt(self, question: str, num_images: int) -> list[dict]:
        prompt = self._format_user_text(question)
        content: list[dict] = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": prompt})
        messages: list[dict] = []
        if self.cfg and self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_inputs(self, question: str, images: list[Any]) -> tuple[int, dict[str, Any]]:
        pil_images = [_to_pil(i) for i in images]
        messages = self._build_prompt(question=question, num_images=len(pil_images))
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=pil_images, text=text, return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[-1]
        return seq_len, inputs

    @torch.no_grad()
    def generate_number(self, question: str, images: list[Any], max_new_tokens: int = 4) -> dict[str, Any]:
        seq_len, inputs = self._prepare_inputs(question=question, images=images)
        generate_kwargs = self._build_generate_kwargs(max_new_tokens)
        out = self.model.generate(**inputs, **generate_kwargs)
        generated = out.sequences[:, seq_len:]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return self._summarize_generation(
            generated_ids=generated,
            scores=getattr(out, "scores", ()) or (),
            tokenizer=tokenizer,
            decoded=decoded,
        )


class FastVLMRunner(_BaseRunner):
    """Adapter for apple/FastVLM-7B (custom LlavaQwen2ForCausalLM via trust_remote_code)."""

    def __init__(
        self,
        model_name: str,
        inference_config: InferenceConfig | None = None,
        device: str | None = None,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
        )
        self.model.eval()
        self.image_processor = self.model.get_vision_tower().image_processor

    def _render_chat(self, question: str, num_images: int) -> str:
        user_text = self._format_user_text(question)
        image_tokens = "\n".join(["<image>"] * num_images)
        user_content = f"{image_tokens}\n{user_text}" if num_images else user_text
        messages: list[dict] = []
        if self.cfg and self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _splice_image_tokens(self, rendered: str, num_images: int) -> torch.LongTensor:
        """Tokenize text around each <image> placeholder and splice IMAGE_TOKEN_INDEX inline."""
        parts = rendered.split("<image>")
        if len(parts) - 1 != num_images:
            raise ValueError(
                f"Chat template produced {len(parts) - 1} <image> placeholders but {num_images} images were provided"
            )

        def tok(text: str) -> list[int]:
            if not text:
                return []
            return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

        ids: list[int] = tok(parts[0])
        for rest in parts[1:]:
            ids.append(FASTVLM_IMAGE_TOKEN_INDEX)
            ids.extend(tok(rest))
        return torch.tensor([ids], dtype=torch.long)

    @torch.no_grad()
    def generate_number(self, question: str, images: list[Any], max_new_tokens: int = 4) -> dict[str, Any]:
        pil_images = [_to_pil(i) for i in images]
        rendered = self._render_chat(question, num_images=len(pil_images))
        input_ids = self._splice_image_tokens(rendered, num_images=len(pil_images)).to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        pixel_values = self.image_processor(images=pil_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.model.device, dtype=self.model.dtype)

        generate_kwargs = self._build_generate_kwargs(max_new_tokens)
        out = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            **generate_kwargs,
        )
        # Custom generate path uses inputs_embeds internally -> out.sequences contains only new tokens.
        generated = out.sequences
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return self._summarize_generation(
            generated_ids=generated,
            scores=getattr(out, "scores", ()) or (),
            tokenizer=self.tokenizer,
            decoded=decoded,
        )


class ConvLLaVARunner(_BaseRunner):
    """Adapter for ConvLLaVA/ConvLLaVA-sft-* (liuhaotian-style LLaVA without HF port).

    Loads the LLaMA backbone, the ConvNeXt vision tower, and the MLP projector from the
    consolidated checkpoint at `model_name`, then runs inference by projecting vision
    features into the embedding space and concatenating them with the tokenized prompt.
    """

    VICUNA_SYSTEM = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    )

    def __init__(
        self,
        model_name: str,
        inference_config: InferenceConfig | None = None,
        device: str | None = None,
        vision_tower_name: str | None = None,
    ):
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            CLIPImageProcessor,
            ConvNextModel,
            LlamaForCausalLM,
        )

        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()

        ckpt_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        vt_name = vision_tower_name or getattr(ckpt_cfg, "mm_vision_tower", None)
        if vt_name is None:
            raise ValueError(f"{model_name} config missing mm_vision_tower")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # LLaMA backbone — loaded from the consolidated checkpoint; extra vision/projector
        # weights get filtered out below when we assemble the state dict for its submodules.
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Vision tower: ConvNextModel exposes `last_hidden_state` as (B, C, H, W) before
        # the pooled layernorm — the same signal the original LLaVA code consumes.
        self.vision_tower = ConvNextModel.from_pretrained(vt_name, dtype=dtype).to(self.device)
        self.vision_tower.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(vt_name)

        # MLP projector: GELU sandwiched between two linears; shapes are mm_hidden_size -> hidden_size.
        import torch.nn as nn

        hidden = ckpt_cfg.hidden_size
        mm_hidden = ckpt_cfg.mm_hidden_size
        projector = nn.Sequential(
            nn.Linear(mm_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        ).to(device=self.device, dtype=dtype)
        self._load_vision_and_projector(projector)
        self.mm_projector = projector
        self.mm_projector.eval()

    def _load_vision_and_projector(self, projector: Any) -> None:
        """Pull `model.mm_projector.*` and `model.vision_tower.vision_tower.*` out of the
        consolidated LlavaLlamaForCausalLM checkpoint and copy them onto the submodules."""
        from pathlib import Path as _Path
        from huggingface_hub import snapshot_download

        snapshot = _Path(snapshot_download(self.model_name, allow_patterns=["*.bin", "*.safetensors", "*.json"]))
        proj_sd: dict[str, torch.Tensor] = {}
        vt_sd: dict[str, torch.Tensor] = {}
        for shard in sorted(list(snapshot.glob("*.bin")) + list(snapshot.glob("*.safetensors"))):
            if shard.suffix == ".safetensors":
                from safetensors.torch import load_file

                sd = load_file(str(shard))
            else:
                sd = torch.load(shard, map_location="cpu", weights_only=True)
            for k, v in sd.items():
                if k.startswith("model.mm_projector."):
                    proj_sd[k[len("model.mm_projector.") :]] = v
                elif k.startswith("model.vision_tower.vision_tower."):
                    vt_sd[k[len("model.vision_tower.vision_tower.") :]] = v

        missing = projector.load_state_dict(proj_sd, strict=False)
        if missing.missing_keys:
            raise RuntimeError(f"Projector missing keys: {missing.missing_keys}")
        vt_res = self.vision_tower.load_state_dict(vt_sd, strict=False)
        if vt_res.missing_keys:
            raise RuntimeError(f"Vision tower missing keys: {vt_res.missing_keys[:8]}...")

    def _encode_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        px = self.image_processor(images=pil_images, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.vision_tower.dtype
        )
        feats = self.vision_tower(px).last_hidden_state  # (N, C, H, W) — pre-pool, no norm
        n, c, h, w = feats.shape
        feats = feats.reshape(n, c, h * w).permute(0, 2, 1)  # (N, H*W, C)
        return self.mm_projector(feats)  # (N, H*W, hidden)

    def _build_prompt(self, question: str, num_images: int) -> str:
        system = (self.cfg.system_prompt.strip() if self.cfg and self.cfg.system_prompt else "") or self.VICUNA_SYSTEM
        user_text = self._format_user_text(question)
        image_tokens = "\n".join(["<image>"] * num_images)
        user = f"{image_tokens}\n{user_text}" if num_images else user_text
        return f"{system}\n\nUSER: {user}\nASSISTANT:"

    def _tokenize_with_image_placeholders(self, prompt: str, num_images: int) -> list[list[int]]:
        parts = prompt.split("<image>")
        if len(parts) - 1 != num_images:
            raise ValueError(
                f"Prompt has {len(parts) - 1} <image> placeholders but {num_images} images were provided"
            )
        chunks: list[list[int]] = []
        for idx, text in enumerate(parts):
            add_bos = idx == 0
            ids = self.tokenizer(text, add_special_tokens=add_bos, return_tensors="pt").input_ids[0].tolist()
            chunks.append(ids)
        return chunks

    @torch.no_grad()
    def generate_number(self, question: str, images: list[Any], max_new_tokens: int = 4) -> dict[str, Any]:
        pil_images = [_to_pil(i) for i in images]
        prompt = self._build_prompt(question, num_images=len(pil_images))
        text_chunks = self._tokenize_with_image_placeholders(prompt, num_images=len(pil_images))
        image_embeds = self._encode_images(pil_images) if pil_images else None  # (N, T, hidden)

        embed_layer = self.model.get_input_embeddings()
        parts: list[torch.Tensor] = []
        for idx, chunk in enumerate(text_chunks):
            chunk_ids = torch.tensor(chunk, device=self.device, dtype=torch.long)
            parts.append(embed_layer(chunk_ids))
            if idx < len(text_chunks) - 1 and image_embeds is not None:
                parts.append(image_embeds[idx])
        inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0).to(self.model.dtype)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        generate_kwargs = self._build_generate_kwargs(max_new_tokens)
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        generated = out.sequences  # inputs_embeds path -> only new tokens
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return self._summarize_generation(
            generated_ids=generated,
            scores=getattr(out, "scores", ()) or (),
            tokenizer=self.tokenizer,
            decoded=decoded,
        )


def build_runner(
    hf_model: str,
    inference_config: InferenceConfig | None = None,
    device: str | None = None,
) -> _BaseRunner:
    """Dispatch to the right runner based on the HF model id."""
    lower = hf_model.lower()
    if "fastvlm" in lower or "llava-qwen" in lower:
        return FastVLMRunner(hf_model, inference_config=inference_config, device=device)
    if "convllava" in lower:
        return ConvLLaVARunner(hf_model, inference_config=inference_config, device=device)
    return HFAttentionRunner(hf_model, inference_config=inference_config, device=device)
