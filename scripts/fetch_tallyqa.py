#!/usr/bin/env python3
"""Fetch the TallyQA test split and migrate it to the local VQAv2-like layout.

Output structure mirrors ``inputs/vqav2_number_val`` so the rest of the
pipeline (loader, config) can share conventions:

    <output_dir>/
        images/000000000001.jpg
        questions.jsonl
        summary.json

Each row of the upstream dataset carries an ``image`` plus a list of QA pairs
(``qa``). We flatten it so every question becomes one JSONL record.
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


DATASET = "vikhyatk/tallyqa-test"
SPLIT = "test"


def load_tallyqa(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Run `uv sync` first.") from exc
    return load_dataset(DATASET, split=split)


def cast_image_without_decode(dataset):
    from datasets import Image

    return dataset.cast_column("image", Image(decode=False))


def parse_int_answer(answer: object) -> int | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, float):
        return int(answer) if answer.is_integer() else None
    text = str(answer).strip()
    if not text:
        return None
    if text.startswith("-"):
        return int(text) if text[1:].isdigit() else None
    return int(text) if text.isdigit() else None


def extension_from_format(image_format: str | None) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tiff"}
    if not image_format:
        return ".jpg"
    return mapping.get(image_format.upper(), ".jpg")


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
    parser = argparse.ArgumentParser(description="Fetch TallyQA test split and emit questions.jsonl + images.")
    parser.add_argument("--output-dir", default="inputs/tallyqa_test")
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of source images to process.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap number of flattened QA records.")
    parser.add_argument(
        "--answer-range",
        type=int,
        default=None,
        help="If set, only keep QA pairs whose integer answer is in [0, answer_range].",
    )
    parser.add_argument("--only-simple", action="store_true", help="Keep only rows with is_simple=True.")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be > 0")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading dataset={DATASET} split={args.split}")
    dataset = load_tallyqa(args.split)
    print(f"[load] loaded {dataset.num_rows} rows")

    metadata_ds = dataset.remove_columns(["image"])
    if args.max_images is not None:
        metadata_ds = metadata_ds.select(range(min(args.max_images, metadata_ds.num_rows)))

    question_records: List[Dict[str, Any]] = []
    answer_counter: Counter[int] = Counter()
    selected_image_indices: List[int] = []
    image_filenames: List[str] = []

    progress_every = max(200, metadata_ds.num_rows // 20) if metadata_ds.num_rows else 1
    for src_idx, row in enumerate(metadata_ds):
        image_index = src_idx
        for qa_idx, qa in enumerate(row.get("qa") or []):
            answer_text = str(qa.get("answer", "")).strip()
            answer_int = parse_int_answer(answer_text)
            if args.answer_range is not None:
                if answer_int is None or not (0 <= answer_int <= args.answer_range):
                    continue
            if args.only_simple and not qa.get("is_simple", False):
                continue

            question_records.append(
                {
                    "question_id": int(image_index) * 100 + int(qa_idx),
                    "image_id": int(image_index),
                    "question": qa.get("question", ""),
                    "question_type": "how many",
                    "answer_type": "number" if answer_int is not None else "other",
                    "multiple_choice_answer": answer_text,
                    "answers": [{"answer": answer_text, "answer_confidence": "yes", "answer_id": 1}],
                    "is_simple": bool(qa.get("is_simple", False)),
                    "data_source": qa.get("data_source", ""),
                    "image_file": "",
                }
            )
            if answer_int is not None:
                answer_counter[answer_int] += 1
            if args.max_samples is not None and len(question_records) >= args.max_samples:
                break

        if not selected_image_indices or selected_image_indices[-1] != image_index:
            selected_image_indices.append(image_index)

        if (src_idx + 1) % progress_every == 0 or src_idx + 1 == metadata_ds.num_rows:
            print(
                f"[scan] processed={src_idx + 1}/{metadata_ds.num_rows} "
                f"records={len(question_records)} unique_images={len(selected_image_indices)}"
            )

        if args.max_samples is not None and len(question_records) >= args.max_samples:
            break

    if not question_records:
        raise RuntimeError("No QA rows matched the requested filters.")

    referenced_image_indices = sorted({record["image_id"] for record in question_records})
    raw_image_dataset = cast_image_without_decode(dataset)
    image_asset_dataset = raw_image_dataset.select(referenced_image_indices)
    image_assets: Dict[int, Dict[str, Any]] = {}
    filename_for_image: Dict[int, str] = {}

    for image_index, row in zip(referenced_image_indices, image_asset_dataset):
        asset = row["image"]
        filename = f"{image_index:012d}{derive_extension(asset)}"
        image_assets[image_index] = asset
        filename_for_image[image_index] = filename
        image_filenames.append(filename)

    for record in question_records:
        record["image_file"] = f"images/{filename_for_image[record['image_id']]}"

    if args.skip_images:
        print(f"[images] skipped saving {len(image_assets)} images")
    else:
        progress_every = max(50, len(image_assets) // 10) if len(image_assets) > 50 else len(image_assets)
        print(f"[images] saving {len(image_assets)} images with workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    save_image_asset, image_assets[idx], image_dir / filename_for_image[idx]
                ): idx
                for idx in referenced_image_indices
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
            "num_unique_images": len(filename_for_image),
            "questions_file": "questions.jsonl",
            "images_dir": "images",
            "images_downloaded": not args.skip_images,
            "max_images": args.max_images,
            "max_samples": args.max_samples,
            "answer_range": args.answer_range,
            "only_simple": args.only_simple,
            "answer_counts": {str(k): answer_counter[k] for k in sorted(answer_counter)},
        },
    )

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(f"[done] images in {image_dir}")


if __name__ == "__main__":
    main()
