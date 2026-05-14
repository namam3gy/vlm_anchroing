"""Closed-API VLM judge clients for the anchoring pilot."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
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


def parse_scores_labeled(raw: str, dim_labels: list[str]) -> dict[str, int | None]:
    """Parse multi-line labeled output like 'Helpfulness: 4\\nVisual Faithfulness: 3'.

    For each label, finds the FIRST '<label>:<sep>?<int>' match (case-insensitive,
    tolerant of asterisks/markdown emphasis around the label and trailing punctuation).
    Returns dict {label: int | None}; None when the label is missing or the integer
    is out of [1, 5].
    """
    text = raw or ""
    out: dict[str, int | None] = {}
    for label in dim_labels:
        # Escape label and allow optional **bold** wrappers + 1-3 char separators
        esc = re.escape(label)
        pattern = re.compile(
            rf"\**\s*{esc}\s*\**\s*[:\-=]\s*\**\s*(\d+)",
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            out[label] = None
            continue
        try:
            v = int(m.group(1))
        except (ValueError, TypeError):
            out[label] = None
            continue
        out[label] = v if 1 <= v <= 5 else None
    return out


@dataclass(frozen=True)
class JudgeResponse:
    score: int | None
    raw: str
    scores: dict[str, int | None] = field(default_factory=dict)


def _image_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    blob = path.read_bytes()
    b64 = base64.b64encode(blob).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


class JudgeClient(Protocol):
    def score(self, images: list[Path], prompt: str, dim_labels: list[str] | None = None) -> JudgeResponse: ...


def _wrap_response(raw: str, dim_labels: list[str] | None) -> JudgeResponse:
    if dim_labels:
        scores = parse_scores_labeled(raw, dim_labels)
        single = scores.get(dim_labels[0])
    else:
        scores = {}
        single = parse_score(raw)
    return JudgeResponse(score=single, raw=raw, scores=scores)


class OpenAIJudgeClient:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_output_tokens: int = 8,
        temperature: float = 0.0,
        base_url: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def score(self, images: list[Path], prompt: str, dim_labels: list[str] | None = None) -> JudgeResponse:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_uri(Path(img))},
            })
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        completion = self._client.chat.completions.create(**kwargs)
        raw = (completion.choices[0].message.content or "").strip()
        return _wrap_response(raw, dim_labels)


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

    def score(self, images: list[Path], prompt: str, dim_labels: list[str] | None = None) -> JudgeResponse:
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
        return _wrap_response(raw, dim_labels)
