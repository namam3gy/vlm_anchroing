"""Pilot manifest types + paired-bootstrap CI helper.

Each PilotSample bundles one VLFeedback record's (prompt, image,
chosen_response) along with paths to the digit-1 anchor image and its
Telea-masked counterpart for the m-arm. Yields per-arm dicts with
`images` lists ordered so that the original target image is ALWAYS
first — the judge prompt explicitly says "the FIRST image".
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PilotSample:
    sample_id: str
    prompt: str
    image_path: Path
    chosen_response: str
    anchor_image_path: Path
    anchor_image_masked_path: Path


def iter_pilot_arms(sample: PilotSample) -> Iterator[dict]:
    yield {"arm": "b", "images": [sample.image_path]}
    yield {"arm": "a", "images": [sample.image_path, sample.anchor_image_path]}
    yield {"arm": "m", "images": [sample.image_path, sample.anchor_image_masked_path]}


def write_manifest_jsonl(samples: list[PilotSample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps({
                "sample_id": s.sample_id,
                "prompt": s.prompt,
                "image_path": str(s.image_path),
                "chosen_response": s.chosen_response,
                "anchor_image_path": str(s.anchor_image_path),
                "anchor_image_masked_path": str(s.anchor_image_masked_path),
            }) + "\n")


def load_manifest_jsonl(path: Path) -> list[PilotSample]:
    out: list[PilotSample] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out.append(PilotSample(
                sample_id=r["sample_id"],
                prompt=r["prompt"],
                image_path=Path(r["image_path"]),
                chosen_response=r["chosen_response"],
                anchor_image_path=Path(r["anchor_image_path"]),
                anchor_image_masked_path=Path(r["anchor_image_masked_path"]),
            ))
    return out


@dataclass(frozen=True)
class PairedCI:
    point: float
    lo: float
    hi: float
    n_pairs: int


def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    alpha: float,
    rng_seed: int,
) -> PairedCI:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    diffs = y[mask] - x[mask]
    n = diffs.size
    if n == 0:
        return PairedCI(point=float("nan"), lo=float("nan"), hi=float("nan"), n_pairs=0)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(low=0, high=n, size=(n_resamples, n))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedCI(point=float(diffs.mean()), lo=float(lo), hi=float(hi), n_pairs=int(n))
