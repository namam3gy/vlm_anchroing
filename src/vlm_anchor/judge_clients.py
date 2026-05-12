"""Closed-API VLM judge clients for the anchoring pilot."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from google import genai
from PIL import Image

from vlm_anchor.utils import extract_first_number


def parse_score(raw: str) -> int | None:
    text = (raw or "").strip()
    digit = extract_first_number(text)
    if not digit:
        return None
    try:
        value = int(digit)
    except (ValueError, TypeError):
        return None
    if value < 1 or value > 5:
        return None
    return value


@dataclass(frozen=True)
class JudgeResponse:
    score: int | None
    raw: str


def _image_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    blob = path.read_bytes()
    b64 = base64.b64encode(blob).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


class JudgeClient(Protocol):
    def score(self, images: list[Path], prompt: str) -> JudgeResponse: ...


class OpenAIJudgeClient:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_output_tokens: int = 8,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def score(self, images: list[Path], prompt: str) -> JudgeResponse:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(Path(img))},
            })
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return JudgeResponse(score=parse_score(raw), raw=raw)


class GeminiJudgeClient:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_output_tokens: int = 8,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )

    def score(self, images: list[Path], prompt: str) -> JudgeResponse:
        contents: list = [prompt]
        for img in images:
            contents.append(Image.open(img).convert("RGB"))
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
        )
        raw = (getattr(response, "text", "") or "").strip()
        return JudgeResponse(score=parse_score(raw), raw=raw)
