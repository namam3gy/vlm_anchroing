"""Build the judge-pilot manifest from VLFeedback or VL-RewardBench.

Dispatches on `dataset.kind` (`vlfeedback` or `vlrewardbench`).

For each sampled record:
  1. Pick a target image and prompt
  2. Pick ONE response uniformly at random (seed-fixed) from the available pool
     - VLFeedback: 4 model completions per record
     - VL-RewardBench: 2 model responses per record
  3. Cache the target image to `<images_dir>/<sample_id>.png`
  4. Append to the manifest

Note on field names: PilotSample.chosen_response retains its original name for
backward compatibility with the v1 (chosen-by-mean-rating) manifests; under the
v2 random selector it semantically holds whichever response the RNG picked.

Usage:
    uv run python scripts/build_judge_pilot_dataset.py --config <cfg>.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from vlm_anchor.judge_pilot_data import PilotSample, write_manifest_jsonl
from vlm_anchor.vlfeedback_loader import (
    derive_chosen_completion_index,
    random_completion_index,
)
from vlm_anchor.vlrewardbench_loader import (
    chosen_response_index,
    random_response_index,
)


def _normalize_completions(rec: dict) -> list[dict]:
    """Convert HF Sequence(struct) struct-of-arrays back to list-of-dicts."""
    raw = rec.get("completions") or {}
    if isinstance(raw, dict):
        n = len(raw.get("annotations", []))
        return [{key: raw[key][i] for key in raw} for i in range(n)]
    return list(raw)


def _build_vlfeedback_sample(rec: dict, rng: random.Random, selector: str) -> tuple[str, str, str] | None:
    """Return (sample_id, prompt, response) or None to skip this record."""
    completions = _normalize_completions(rec)
    if selector == "random":
        idx = random_completion_index(completions, rng)
    elif selector == "chosen":
        idx = derive_chosen_completion_index(completions)
    else:
        raise ValueError(f"Unknown selector: {selector!r}")
    if idx is None:
        return None
    response = (completions[idx].get("response") or "").strip()
    if not response:
        return None
    return rec["id"], str(rec["prompt"]).strip(), response


def _build_vlrewardbench_sample(rec: dict, rng: random.Random, selector: str) -> tuple[str, str, str] | None:
    """Return (sample_id, prompt, response) or None to skip this record."""
    responses = list(rec.get("response") or [])
    if selector == "random":
        idx = random_response_index(responses, rng)
    elif selector == "chosen":
        ranking = list(rec.get("human_ranking") or [])
        idx = chosen_response_index(responses, ranking)
    else:
        raise ValueError(f"VL-RewardBench unknown selector: {selector!r}")
    if idx is None:
        return None
    response = (responses[idx] or "").strip()
    if not response:
        return None
    return rec["id"], str(rec["query"]).strip(), response


_BUILDERS = {
    "vlfeedback": _build_vlfeedback_sample,
    "vlrewardbench": _build_vlrewardbench_sample,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds_cfg = cfg["dataset"]
    anchor = cfg["anchor"]

    kind = ds_cfg.get("kind", "vlfeedback")
    if kind not in _BUILDERS:
        raise ValueError(f"Unknown dataset.kind: {kind!r}")
    builder = _BUILDERS[kind]
    selector = ds_cfg.get("response_selector", "random")

    images_dir = Path(ds_cfg["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ds_cfg['hf_repo']} (split={ds_cfg['hf_split']}, kind={kind}, selector={selector}) ...")
    ds = load_dataset(ds_cfg["hf_repo"], split=ds_cfg["hf_split"])
    n_total = len(ds)
    print(f"Loaded {n_total} records")

    rng = random.Random(int(ds_cfg["seed"]))
    indices = list(range(n_total))
    rng.shuffle(indices)

    anchor_path = Path(anchor["digit_image"]).resolve()
    anchor_masked_path = Path(anchor["digit_image_masked"]).resolve()

    samples: list[PilotSample] = []
    target_count = int(ds_cfg["sample_count"])
    cursor = 0
    while len(samples) < target_count and cursor < len(indices):
        idx = indices[cursor]
        cursor += 1
        rec = ds[idx]
        built = builder(rec, rng, selector)
        if built is None:
            continue
        sample_id, prompt, response = built

        image_path = images_dir / f"{sample_id}.png"
        if not image_path.exists():
            rec["image"].convert("RGB").save(image_path, format="PNG")

        samples.append(PilotSample(
            sample_id=sample_id,
            prompt=prompt,
            image_path=image_path.resolve(),
            chosen_response=response,  # field name retained; semantically v2 = random pick
            anchor_image_path=anchor_path,
            anchor_image_masked_path=anchor_masked_path,
        ))

    if len(samples) < target_count:
        raise RuntimeError(f"Only {len(samples)} usable samples found after scanning {cursor} records")

    out_path = Path(ds_cfg["manifest_out"])
    write_manifest_jsonl(samples, out_path)
    print(f"Wrote {len(samples)} samples to {out_path}")
    print(f"Cached {len(samples)} images under {images_dir}")


if __name__ == "__main__":
    main()
