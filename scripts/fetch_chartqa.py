#!/usr/bin/env python3
"""Fetch the ChartQA *test* split and migrate it to the local VQA-like layout.

Only the ``test`` split is supported (per project scope); the script errors
on other splits unless explicitly overridden. Output layout:

    <output_dir>/
        images/000000000001.png
        questions.jsonl
        summary.json
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image as PILImage


DATASET = "HuggingFaceM4/ChartQA"
SPLIT = "test"


def load_chartqa(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Run `uv sync` first.") from exc
    return load_dataset(DATASET, split=split)


def cast_image_without_decode(dataset):
    from datasets import Image

    return dataset.cast_column("image", Image(decode=False))


def extension_from_format(image_format: str | None) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tiff"}
    if not image_format:
        return ".png"
    return mapping.get(image_format.upper(), ".png")


def derive_extension(image_asset: Dict[str, Any]) -> str:
    source_path = image_asset.get("path")
    if source_path:
        suffix = Path(source_path).suffix.lower()
        if suffix:
            return suffix
    image_bytes = image_asset.get("bytes")
    if image_bytes:
        with PILImage.open(io.BytesIO(bytes(image_bytes))) as image:
            return extension_from_format(image.format)
    return ".png"


def save_image_asset(image_asset: Dict[str, Any], target: Path) -> None:
    if target.exists():
        return
    source_path = image_asset.get("path")
    if source_path:
        source = Path(source_path)
        if source.exists():
            shutil.copy2(source, target)
            return
    image_bytes = image_asset.get("bytes")
    if image_bytes:
        target.write_bytes(bytes(image_bytes))
        return
    raise RuntimeError(f"Unable to save image asset for {target.name}: missing path and bytes")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


HUMAN_OR_MACHINE_LABELS = {0: "human", 1: "machine"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ChartQA test split and emit questions.jsonl + images.")
    parser.add_argument("--output-dir", default="inputs/chartqa_test")
    parser.add_argument("--split", default=SPLIT, help="Default: test. Other splits rejected unless --allow-other-splits.")
    parser.add_argument("--allow-other-splits", action="store_true", help="Permit splits other than test (not recommended).")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if args.split != SPLIT and not args.allow_other_splits:
        parser.error(f"ChartQA fetcher is restricted to the '{SPLIT}' split. Pass --allow-other-splits to override.")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading dataset={DATASET} split={args.split}")
    dataset = load_chartqa(args.split)
    print(f"[load] loaded {dataset.num_rows} rows")

    total_rows = dataset.num_rows
    if args.max_samples is not None:
        total_rows = min(total_rows, args.max_samples)
        dataset = dataset.select(range(total_rows))

    metadata_ds = dataset.remove_columns(["image"])
    question_records: List[Dict[str, Any]] = []

    for idx, row in enumerate(metadata_ds):
        labels = row.get("label") or []
        primary_answer = str(labels[0]) if labels else ""
        human_or_machine = row.get("human_or_machine")
        question_records.append(
            {
                "question_id": idx,
                "image_id": idx,
                "question": row.get("query", ""),
                "question_type": "chartqa",
                "answer_type": "text",
                "multiple_choice_answer": primary_answer,
                "answers": [
                    {"answer": str(label), "answer_confidence": "yes", "answer_id": i + 1}
                    for i, label in enumerate(labels)
                ],
                "human_or_machine": HUMAN_OR_MACHINE_LABELS.get(int(human_or_machine), str(human_or_machine))
                if human_or_machine is not None
                else None,
                "image_file": "",
            }
        )

    raw_image_dataset = cast_image_without_decode(dataset)
    filename_for_index: Dict[int, str] = {}
    image_assets: Dict[int, Dict[str, Any]] = {}

    print(f"[images] resolving filenames for {raw_image_dataset.num_rows} images")
    for idx, row in enumerate(raw_image_dataset):
        asset = row["image"]
        filename = f"{idx:012d}{derive_extension(asset)}"
        filename_for_index[idx] = filename
        image_assets[idx] = asset

    for record in question_records:
        record["image_file"] = f"images/{filename_for_index[record['image_id']]}"

    if args.skip_images:
        print(f"[images] skipped saving {len(image_assets)} images")
    else:
        progress_every = max(50, len(image_assets) // 10) if len(image_assets) > 50 else len(image_assets)
        print(f"[images] saving {len(image_assets)} images with workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    save_image_asset, image_assets[idx], image_dir / filename_for_index[idx]
                ): idx
                for idx in filename_for_index
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                if progress_every and (completed % progress_every == 0 or completed == len(futures)):
                    print(f"[images] saved={completed}/{len(futures)}")

    jsonl_path = output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in question_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        output_dir / "summary.json",
        {
            "dataset": DATASET,
            "split": args.split,
            "num_questions": len(question_records),
            "num_unique_images": len(filename_for_index),
            "questions_file": "questions.jsonl",
            "images_dir": "images",
            "images_downloaded": not args.skip_images,
            "max_samples": args.max_samples,
        },
    )

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(f"[done] images in {image_dir}")


if __name__ == "__main__":
    main()
