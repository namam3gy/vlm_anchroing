from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm_anchor.utils import extract_first_number


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


class HFAttentionRunner:
    def __init__(
        self,
        model_name: str,
        inference_config: InferenceConfig | None = None,
        device: str | None = None,
    ):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_name = model_name
        self.cfg = inference_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
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
        prompt = (
            self.cfg.user_template.replace("{question}", question)
            if self.cfg
            else f"Answer with exactly one Arabic numeral only. Question: {question}"
        )
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
        do_sample = bool(self.cfg and self.cfg.temperature > 0)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
            generate_kwargs["top_p"] = self.cfg.top_p

        out = self.model.generate(**inputs, **generate_kwargs)
        generated = out.sequences[:, seq_len:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return {
            "raw_text": decoded,
            "parsed_number": extract_first_number(decoded),
            "backend": "huggingface",
        }
