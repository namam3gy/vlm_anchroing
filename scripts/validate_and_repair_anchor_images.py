"""Validate rendered anchor digit images via Qwen2.5-VL OCR and regenerate mismatches.

For each image inputs/irrelevant_number/<N>.png in the supplied integer list, ask
Qwen2.5-VL to read the digits, parse the first number, and flag any image whose
read-out does not match <N>. Mismatches are regenerated with a different seed via
scripts/generate_irrelevant_number_images.py and re-validated, up to a fixed
retry budget.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image

from vlm_anchor.models import build_runner
from vlm_anchor.utils import extract_first_number

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "inputs/irrelevant_number"
GENERATOR = ROOT / "scripts/generate_irrelevant_number_images.py"

OCR_PROMPT = (
    "Read the number written in this image. Reply with only the digits, "
    "nothing else, no units, no words."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--numbers",
        required=True,
        help="Comma-separated list of integers to validate (filenames without .png).",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--retries", type=int, default=3, help="Per-image regeneration attempts.")
    p.add_argument("--seed-base", type=int, default=1729, help="Initial seed_base; retries use 1729+10000*k.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_ocr(runner, expected_numbers: list[int]) -> dict[int, tuple[str, str]]:
    """Return {number: (raw_decoded, parsed)} for each expected number."""
    results: dict[int, tuple[str, str]] = {}
    for n in expected_numbers:
        path = ANCHOR_DIR / f"{n}.png"
        if not path.exists():
            results[n] = ("<missing>", "")
            continue
        with Image.open(path) as im:
            im_rgb = im.convert("RGB")
            out = runner.generate_number(
                question=OCR_PROMPT, images=[im_rgb], max_new_tokens=12
            )
        decoded = (out.get("raw_text") or "").strip()
        parsed = out.get("parsed_number") or extract_first_number(decoded)
        results[n] = (decoded, parsed)
    return results


def regenerate(numbers: list[int], seed_base: int) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(GENERATOR),
        "--numbers",
        ",".join(str(n) for n in numbers),
        "--seed-base",
        str(seed_base),
        "--overwrite",
    ]
    print(f"[regen] seed_base={seed_base} numbers={numbers}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    args = parse_args()
    targets = sorted({int(x) for x in args.numbers.split(",") if x.strip()})

    print(f"[load] {args.model}", flush=True)
    runner = build_runner(args.model, device=args.device)

    pending = list(targets)
    final_mismatch: list[tuple[int, str, str]] = []

    for attempt in range(args.retries + 1):
        if not pending:
            break
        print(f"\n[ocr] attempt {attempt} | n={len(pending)}", flush=True)
        results = run_ocr(runner, pending)
        ok: list[int] = []
        bad: list[int] = []
        for n in pending:
            decoded, parsed = results[n]
            try:
                parsed_int = int(parsed) if parsed else None
            except ValueError:
                parsed_int = None
            if parsed_int == n:
                ok.append(n)
            else:
                bad.append(n)
                if attempt == 0 or attempt == args.retries:
                    print(f"  bad {n}: decoded={decoded!r} parsed={parsed!r}", flush=True)

        print(f"[ocr] attempt {attempt} ok={len(ok)} bad={len(bad)}", flush=True)

        if not bad:
            pending = []
            break

        if attempt == args.retries:
            for n in bad:
                final_mismatch.append((n, *results[n]))
            pending = []
            break

        new_seed_base = args.seed_base + 10000 * (attempt + 1)
        regenerate(bad, seed_base=new_seed_base)
        pending = bad

    print()
    if final_mismatch:
        print(f"[result] FINAL_MISMATCH n={len(final_mismatch)}")
        for n, decoded, parsed in final_mismatch:
            print(f"  {n}: decoded={decoded!r} parsed={parsed!r}")
        sys.exit(1)
    print(f"[result] all {len(targets)} images verified clean.")


if __name__ == "__main__":
    main()
