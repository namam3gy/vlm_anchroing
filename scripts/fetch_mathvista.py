#!/usr/bin/env python3
"""Fetch MathVista (AI4Math/MathVista) and migrate to the local VQA-like layout.

Defaults to the ``testmini`` split (1k rows) since the full ``test`` split has
no public ground-truth answers. Output:

    <output_dir>/
        images/<pid>.<ext>
        questions.jsonl
        summary.json
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image as PILImage


DATASET = "AI4Math/MathVista"
SPLIT = "testmini"
IMAGE_COLUMN = "decoded_image"


def load_mathvista(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Run `uv sync` first.") from exc
    return load_dataset(DATASET, split=split)


def cast_image_without_decode(dataset):
    from datasets import Image

    return dataset.cast_column(IMAGE_COLUMN, Image(decode=False))


def extension_from_format(image_format: str | None) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tiff"}
    if not image_format:
        return ".jpg"
    return mapping.get(image_format.upper(), ".jpg")


def derive_extension(image_asset: Dict[str, Any], fallback_path: str | None) -> str:
    source_path = image_asset.get("path")
    if source_path:
        suffix = Path(source_path).suffix.lower()
        if suffix:
            return suffix
    if fallback_path:
        suffix = Path(fallback_path).suffix.lower()
        if suffix:
            return suffix
    image_bytes = image_asset.get("bytes")
    if image_bytes:
        with PILImage.open(io.BytesIO(bytes(image_bytes))) as image:
            return extension_from_format(image.format)
    return ".jpg"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MathVista and emit questions.jsonl + images.")
    parser.add_argument("--output-dir", default="inputs/mathvista_testmini")
    parser.add_argument("--split", default=SPLIT, help="Dataset split: testmini (default), test, or default.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be > 0")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading dataset={DATASET} split={args.split}")
    dataset = load_mathvista(args.split)
    print(f"[load] loaded {dataset.num_rows} rows")

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, dataset.num_rows)))

    metadata_ds = dataset.remove_columns([IMAGE_COLUMN])
    question_records: List[Dict[str, Any]] = []
    answer_type_counter: Counter[str] = Counter()
    question_type_counter: Counter[str] = Counter()

    for idx, row in enumerate(metadata_ds):
        pid = str(row.get("pid") or idx)
        answer_text = "" if row.get("answer") is None else str(row["answer"])
        question_type = row.get("question_type") or ""
        answer_type = row.get("answer_type") or ""
        answer_type_counter[answer_type] += 1
        question_type_counter[question_type] += 1

        choices = row.get("choices")
        answers: List[Dict[str, Any]]
        if choices:
            answers = [
                {"answer": str(choice), "answer_confidence": "yes", "answer_id": i + 1}
                for i, choice in enumerate(choices)
            ]
        else:
            answers = [{"answer": answer_text, "answer_confidence": "yes", "answer_id": 1}]

        question_records.append(
            {
                "question_id": pid,
                "image_id": pid,
                "question": row.get("question", ""),
                "question_type": question_type,
                "answer_type": answer_type,
                "multiple_choice_answer": answer_text,
                "answers": answers,
                "choices": list(choices) if choices else None,
                "unit": row.get("unit"),
                "precision": row.get("precision"),
                "metadata": row.get("metadata"),
                "query": row.get("query", ""),
                "source_image": row.get("image"),
                "image_file": "",
            }
        )

    raw_image_dataset = cast_image_without_decode(dataset)
    filename_for_pid: Dict[str, str] = {}
    image_assets: Dict[str, Dict[str, Any]] = {}

    print(f"[images] resolving filenames for {raw_image_dataset.num_rows} images")
    for record, row in zip(question_records, raw_image_dataset):
        asset = row[IMAGE_COLUMN]
        pid = record["image_id"]
        filename = f"{pid}{derive_extension(asset, record.get('source_image'))}"
        filename_for_pid[pid] = filename
        image_assets[pid] = asset

    for record in question_records:
        record["image_file"] = f"images/{filename_for_pid[record['image_id']]}"

    if args.skip_images:
        print(f"[images] skipped saving {len(image_assets)} images")
    else:
        progress_every = max(50, len(image_assets) // 10) if len(image_assets) > 50 else len(image_assets)
        print(f"[images] saving {len(image_assets)} images with workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    save_image_asset, image_assets[pid], image_dir / filename_for_pid[pid]
                ): pid
                for pid in filename_for_pid
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
            "num_unique_images": len(filename_for_pid),
            "questions_file": "questions.jsonl",
            "images_dir": "images",
            "images_downloaded": not args.skip_images,
            "max_samples": args.max_samples,
            "question_type_counts": dict(question_type_counter),
            "answer_type_counts": dict(answer_type_counter),
        },
    )

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(f"[done] images in {image_dir}")


if __name__ == "__main__":
    main()
