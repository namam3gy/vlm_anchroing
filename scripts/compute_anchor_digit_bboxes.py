"""E1-patch infra — compute the digit-pixel bounding box for every anchor
image by diffing the original `inputs/irrelevant_number/<v>.png` against
its masked counterpart `inputs/irrelevant_number_masked/<v>.png`.

The masked version inpaints the digit pixels only; everything else is
identical. The bbox of the absolute pixel-difference therefore is the
bounding box of the digit region. We persist the bboxes to JSON so that
`extract_attention_mass.py` can map them to vision-encoder patch
indices and slice attention accordingly.

Output: ``inputs/irrelevant_number_bboxes.json`` —

    {
      "<value>": {
        "image_size": [W, H],
        "bbox_xyxy": [x0, y0, x1, y1],   # pixel coordinates
        "fraction": float,                # bbox area / image area
        "diff_pixels": int                # n nonzero diff pixels
      }, ...
    }

Idempotent. Skips values for which both files don't exist or diff is
below threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def compute_bbox(anchor_path: Path, masked_path: Path, threshold: int = 8) -> dict | None:
    """Return bbox dict or None if diff is below threshold / shapes mismatch."""
    a = np.asarray(Image.open(anchor_path).convert("RGB"))
    m = np.asarray(Image.open(masked_path).convert("RGB"))
    if a.shape != m.shape:
        return None
    diff = np.abs(a.astype(np.int16) - m.astype(np.int16)).sum(axis=-1)
    mask = diff > threshold
    n = int(mask.sum())
    if n == 0:
        return None
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    H, W = mask.shape
    area = (x1 - x0) * (y1 - y0)
    return {
        "image_size": [W, H],
        "bbox_xyxy": [x0, y0, x1, y1],
        "fraction": area / (W * H),
        "diff_pixels": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", type=Path,
                        default=REPO_ROOT / "inputs" / "irrelevant_number")
    parser.add_argument("--masked-dir", type=Path,
                        default=REPO_ROOT / "inputs" / "irrelevant_number_masked")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "inputs" / "irrelevant_number_bboxes.json")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Per-pixel sum-abs-diff threshold (default 8).")
    args = parser.parse_args()

    anchors = {p.stem: p for p in sorted(args.anchor_dir.glob("*.png"))}
    masked = {p.stem: p for p in sorted(args.masked_dir.glob("*.png"))}

    out: dict[str, dict] = {}
    skipped = []
    for value in sorted(set(anchors) & set(masked), key=lambda s: int(s) if s.isdigit() else float("inf")):
        bbox = compute_bbox(anchors[value], masked[value], threshold=args.threshold)
        if bbox is None:
            skipped.append(value)
            continue
        out[value] = bbox

    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} | n={len(out)} bboxes | skipped={len(skipped)}")
    if skipped[:5]:
        print(f"  first skipped: {skipped[:5]}")
    if out:
        sample = next(iter(out.items()))
        print(f"  sample: {sample[0]} → bbox={sample[1]['bbox_xyxy']} "
              f"({sample[1]['fraction']*100:.2f}% of {sample[1]['image_size']})")


if __name__ == "__main__":
    main()
