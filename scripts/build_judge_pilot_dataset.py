"""Build the judge-pilot manifest from VLFeedback.

One-shot:
  1. Stream VLFeedback (uses HF datasets cache; ~80K records, ~10 GB on disk
     because of embedded images — the streaming-then-cache pattern below
     downloads the full snapshot once).
  2. Sample 100 random records (seed-fixed).
  3. For each, derive `chosen_response` = response of the completion with the
     highest mean of 3 GPT-4V dim ratings (skipping unparseable).
  4. Cache each target image to `inputs/judge_pilot/images/<sample_id>.png`.
  5. Write `inputs/judge_pilot/manifest.jsonl`.

Usage:
    uv run python scripts/build_judge_pilot_dataset.py --config configs/judge_pilot.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from vlm_anchor.judge_pilot_data import PilotSample, write_manifest_jsonl
from vlm_anchor.vlfeedback_loader import derive_chosen_completion_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds_cfg = cfg["dataset"]
    anchor = cfg["anchor"]

    images_dir = Path(ds_cfg["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ds_cfg['hf_repo']} (split={ds_cfg['hf_split']}) ...")
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
        completions = list(rec.get("completions") or [])
        chosen_idx = derive_chosen_completion_index(completions)
        if chosen_idx is None:
            continue
        chosen = completions[chosen_idx]
        chosen_response = (chosen.get("response") or "").strip()
        if not chosen_response:
            continue

        sample_id = rec["id"]
        image_path = images_dir / f"{sample_id}.png"
        if not image_path.exists():
            rec["image"].convert("RGB").save(image_path, format="PNG")

        samples.append(PilotSample(
            sample_id=sample_id,
            prompt=str(rec["prompt"]).strip(),
            image_path=image_path.resolve(),
            chosen_response=chosen_response,
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
