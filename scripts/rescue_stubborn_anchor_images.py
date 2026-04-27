"""Rescue stubborn anchor digit images.

Loads FLUX and Qwen2.5-VL in the same Python process, then for each stubborn
number iterates over (scene_offset, seed) combinations until the rendered image
OCR-decodes to the intended digits. Avoids reloading the OCR model per attempt.
"""

from __future__ import annotations

import argparse
import io
from itertools import product
from pathlib import Path

import torch
from PIL import Image

from vlm_anchor.models import build_runner
from vlm_anchor.utils import extract_first_number

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_irrelevant_number_images import (  # type: ignore
    DEFAULT_NEGATIVE_PROMPT,
    SCENE_TEMPLATES,
    build_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "inputs/irrelevant_number"

OCR_PROMPT = (
    "Read the number written in this image. Reply with only the digits, "
    "nothing else, no units, no words."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numbers", required=True, help="CSV of stubborn integers, e.g. '300,1700'.")
    p.add_argument("--ocr-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--flux-model", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=0.0)
    p.add_argument("--max-attempts", type=int, default=24, help="Per-number total attempts.")
    return p.parse_args()


def ocr_check(runner, path: Path, expected: int) -> tuple[bool, str, str]:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        out = runner.generate_number(question=OCR_PROMPT, images=[rgb], max_new_tokens=12)
    decoded = (out.get("raw_text") or "").strip()
    parsed = out.get("parsed_number") or extract_first_number(decoded)
    try:
        return int(parsed) == expected, decoded, parsed
    except (TypeError, ValueError):
        return False, decoded, parsed


def main() -> None:
    args = parse_args()
    targets = sorted({int(x) for x in args.numbers.split(",") if x.strip()})

    print("[load] FLUX", flush=True)
    from diffusers import FluxPipeline

    flux = FluxPipeline.from_pretrained(args.flux_model, torch_dtype=torch.bfloat16)
    flux = flux.to("cuda")
    flux.set_progress_bar_config(disable=True)

    print(f"[load] OCR ({args.ocr_model})", flush=True)
    ocr = build_runner(args.ocr_model, device="cuda")

    n_scenes = len(SCENE_TEMPLATES)
    failed: list[int] = []

    for n in targets:
        natural_offset = (n - 1) % n_scenes
        attempts = 0
        success = False
        # Iterate seeds and scene offsets, skipping the original (offset=0, default seeds) and prior retries.
        # Walk scene offsets 1..n_scenes-1 first (mod), each with a few seeds.
        candidate_seeds = [41729, 51729, 61729, 71729]
        for offset, seed_base in product(range(1, n_scenes), candidate_seeds):
            if attempts >= args.max_attempts:
                break
            attempts += 1
            scene = SCENE_TEMPLATES[(n - 1 + offset) % n_scenes]
            prompt = build_prompt(number=n, seed_base=seed_base, scene_offset=offset)
            seed = seed_base + n
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = flux(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=256,
                generator=generator,
            ).images[0]
            target = ANCHOR_DIR / f"{n}.png"
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            with Image.open(io.BytesIO(buf.getvalue())) as im:
                im.convert("RGB").save(target, format="PNG")

            ok, decoded, parsed = ocr_check(ocr, target, n)
            print(
                f"[try] n={n} attempt={attempts} offset={offset} (scene={scene.name}) "
                f"seed_base={seed_base} -> decoded={decoded!r} parsed={parsed!r} ok={ok}",
                flush=True,
            )
            if ok:
                success = True
                break
        if not success:
            failed.append(n)
            print(f"[fail] n={n} after {attempts} attempts", flush=True)

    print()
    if failed:
        print(f"[result] STILL_FAILED n={len(failed)}: {failed}")
        sys.exit(1)
    print(f"[result] all {len(targets)} stubborn images rescued.")


if __name__ == "__main__":
    main()
