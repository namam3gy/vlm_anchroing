"""Generate digit-masked variants of FLUX-rendered anchor images.

For each anchor image at ``inputs/irrelevant_number/{value}.png`` the script
runs easyocr to find the bounding box of the rendered digit, computes the
mean RGB color of pixels OUTSIDE that bounding box, and fills the bbox
region with that mean color. The result preserves all non-digit pixels
exactly while erasing the digit into a "background-tinted" rectangle.

Usage:
    uv run python scripts/generate_anchor_masked_images.py \\
        --numbers 0,5,100,5000,10000 --preview

When ``--preview`` is supplied, output goes to
``inputs/irrelevant_number_masked_preview/`` and a side-by-side compare PNG
(``{value}_compare.png``) is written next to the masked image so the user can
inspect the result. Without ``--preview``, output goes to
``inputs/irrelevant_number_masked/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "inputs" / "irrelevant_number"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "inputs" / "irrelevant_number_masked"
PREVIEW_OUTPUT_DIR = PROJECT_ROOT / "inputs" / "irrelevant_number_masked_preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers",
        type=str,
        default=None,
        help="Comma-separated subset of values to process. Defaults to all PNGs in inputs/irrelevant_number/.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write to inputs/irrelevant_number_masked_preview/ and emit side-by-side compare PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masked images.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for easyocr (default CPU; 128 images is fast on CPU).",
    )
    return parser.parse_args()


def discover_values(numbers_arg: str | None) -> list[int]:
    if numbers_arg:
        return [int(n.strip()) for n in numbers_arg.split(",") if n.strip()]
    values: list[int] = []
    for p in sorted(SOURCE_DIR.glob("*.png")):
        try:
            values.append(int(p.stem))
        except ValueError:
            continue
    return sorted(values)


def axis_aligned_bbox(quad: Iterable[Iterable[float]]) -> tuple[int, int, int, int]:
    pts = [(float(p[0]), float(p[1])) for p in quad]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


def text_distance(detected: str, expected: str) -> tuple[int, int]:
    """Sort key: prefer exact match on expected digit string, then closeness."""
    # Strip whitespace and any non-digit garnish for comparison.
    cleaned = "".join(ch for ch in detected if ch.isdigit())
    if cleaned == expected:
        return (0, 0)
    if expected in cleaned:
        return (1, abs(len(cleaned) - len(expected)))
    if cleaned and cleaned in expected:
        return (2, abs(len(cleaned) - len(expected)))
    return (3, abs(len(cleaned) - len(expected)) + 100)


def pick_best_detection(
    detections: list[tuple], expected_value: int
) -> tuple | None:
    """From easyocr output, choose the detection whose text best matches the expected digit."""
    expected_str = str(expected_value)
    candidates: list[tuple] = []
    for det in detections:
        bbox, text, conf = det
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        candidates.append(det)
    if not candidates:
        return None
    candidates.sort(
        key=lambda d: (text_distance(d[1], expected_str), -float(d[2]))
    )
    return candidates[0]


def mask_image(
    img_array: np.ndarray, bbox_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    """Replace the bbox region with the mean RGB of pixels outside the bbox."""
    h, w, _ = img_array.shape
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    inside_mask = np.zeros((h, w), dtype=bool)
    inside_mask[y0:y1, x0:x1] = True
    outside_pixels = img_array[~inside_mask]
    mean_color = outside_pixels.mean(axis=0).astype(np.uint8)
    out = img_array.copy()
    out[y0:y1, x0:x1] = mean_color
    return out


def write_compare(orig_path: Path, masked_path: Path, compare_path: Path) -> None:
    orig = Image.open(orig_path).convert("RGB")
    masked = Image.open(masked_path).convert("RGB")
    gap = 10
    width = orig.width + masked.width + gap
    height = max(orig.height, masked.height)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(orig, (0, 0))
    canvas.paste(masked, (orig.width + gap, 0))
    canvas.save(compare_path)


def main() -> int:
    args = parse_args()

    if not SOURCE_DIR.exists():
        print(f"ERROR: source dir does not exist: {SOURCE_DIR}", file=sys.stderr)
        return 2

    output_dir = PREVIEW_OUTPUT_DIR if args.preview else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    values = discover_values(args.numbers)
    if not values:
        print("ERROR: no values to process", file=sys.stderr)
        return 2

    # Lazy import — easyocr pulls torch on first import.
    import easyocr  # type: ignore

    reader = easyocr.Reader(["en"], gpu=args.gpu, verbose=False)

    successes: list[int] = []
    failures: list[tuple[int, str]] = []

    for value in values:
        src = SOURCE_DIR / f"{value}.png"
        if not src.exists():
            failures.append((value, "source image missing"))
            continue
        dst = output_dir / f"{value}.png"
        if dst.exists() and not args.overwrite:
            print(f"skip {value}: {dst} already exists (use --overwrite)")
            successes.append(value)
            continue

        img = np.array(Image.open(src).convert("RGB"))
        detections = reader.readtext(img)
        best = pick_best_detection(detections, value)
        if best is None:
            failures.append((value, "no digit-like detection"))
            print(
                f"FAIL {value}: no digit-like detection in OCR output",
                file=sys.stderr,
            )
            continue

        bbox_quad, text, conf = best
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits != str(value):
            print(
                f"WARN {value}: best OCR text={text!r} (digits={digits!r}) "
                f"does not exact-match expected {value} (conf={conf:.3f}); using anyway",
                file=sys.stderr,
            )

        bbox_xyxy = axis_aligned_bbox(bbox_quad)
        masked = mask_image(img, bbox_xyxy)
        Image.fromarray(masked).save(dst)
        print(
            f"OK   {value}: text={text!r} conf={conf:.3f} bbox={bbox_xyxy} "
            f"-> {dst}"
        )
        successes.append(value)

        if args.preview:
            compare_path = output_dir / f"{value}_compare.png"
            write_compare(src, dst, compare_path)
            print(f"     wrote compare {compare_path}")

    print(
        f"\nDone. success={len(successes)} fail={len(failures)} "
        f"out={output_dir}"
    )
    if failures:
        for v, reason in failures:
            print(f"  FAIL {v}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
