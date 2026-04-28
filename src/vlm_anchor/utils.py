from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved.resolve()
    if base_dir is not None:
        return (Path(base_dir).expanduser() / resolved).resolve()
    return resolved.resolve()


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else str(value)
                    if isinstance(value, Path)
                    else value
                    for key, value in row.items()
                }
            )


def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"[^a-z0-9\-\s\.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_first_number(text: str | None) -> str:
    text = normalize_numeric_text(text)
    if not text:
        return ""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        value = match.group(0)
        if value.endswith(".0"):
            value = value[:-2]
        return value

    tokens = text.split()
    total = 0
    current = 0
    matched = False
    for tok in tokens:
        if tok not in _NUM_WORDS:
            continue
        matched = True
        val = _NUM_WORDS[tok]
        if tok == "hundred":
            current = max(1, current) * val
        else:
            current += val
    total += current
    if matched:
        return str(total)
    return text.split()[0] if text.split() else ""


def extract_last_number(text: str | None) -> str:
    """Return the *last* numeric span in ``text``.

    Mirrors :func:`extract_first_number` but matches the final occurrence
    rather than the first. Reasoning-mode VLMs emit a chain of thought
    followed by a final answer — the answer lives at the *end* of the
    output, while the trace contains many irrelevant numeric spans.
    """
    text = normalize_numeric_text(text)
    if not text:
        return ""
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    if matches:
        value = matches[-1].group(0)
        if value.endswith(".0"):
            value = value[:-2]
        return value

    tokens = text.split()
    total = 0
    current = 0
    matched = False
    for tok in tokens:
        if tok not in _NUM_WORDS:
            continue
        matched = True
        val = _NUM_WORDS[tok]
        if tok == "hundred":
            current = max(1, current) * val
        else:
            current += val
    total += current
    if matched:
        return str(total)
    return tokens[-1] if tokens else ""


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
