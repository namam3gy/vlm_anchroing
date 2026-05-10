"""Build site/data/demo.json + site/assets/img/ from outputs/ predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Display id -> outputs/ subdirectory name
MAIN_PANEL: dict[str, str] = {
    "llava-interleave-7b": "llava-next-interleaved-7b",
    "llava-onevision-7b":  "llava-onevision-qwen2-7b-ov",
    "qwen2.5-vl-7b":       "qwen2.5-vl-7b-instruct",
    "qwen3-vl-8b":         "qwen3-vl-8b-instruct",
    "gemma-3-27b":         "gemma3-27b-it",
}
MAIN_PANEL_LABELS: dict[str, str] = {
    "llava-interleave-7b": "LLaVA-Interleave 7B (main)",
    "llava-onevision-7b":  "LLaVA-OneVision 7B",
    "qwen2.5-vl-7b":       "Qwen2.5-VL 7B",
    "qwen3-vl-8b":         "Qwen3-VL 8B",
    "gemma-3-27b":         "Gemma-3 27B",
}
FORBIDDEN_OUTPUT_SUBTREE = "before_C_form"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--inputs-root", type=Path, default=PROJECT_ROOT / "inputs")
    p.add_argument("--site-root", type=Path, default=PROJECT_ROOT / "site")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--anchor-stratum", default="S1",
                   help="Stratum suffix used for the anchor (a) and masked (m) conditions.")
    p.add_argument("--max-image-px", type=int, default=768,
                   help="Long-edge pixel cap for copied images.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"build_demo_data: outputs={args.outputs_root} site={args.site_root}", file=sys.stderr)
    # Pipeline implemented in subsequent tasks.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
