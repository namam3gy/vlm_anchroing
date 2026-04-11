#!/usr/bin/env python3
import argparse
import io
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image as PILImage


DATASET = "lmms-lab/VQAv2"
DATASET_CACHE_SLUG = "datasets--lmms-lab--vq_av2"
CONFIG = "default"
SPLIT = "validation"


def find_local_arrow_paths(split: str, datasets_cache_dir: str | None = None) -> List[Path]:
    cache_roots: List[Path] = []
    if datasets_cache_dir:
        cache_roots.append(Path(datasets_cache_dir).expanduser())

    env_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if env_cache_dir:
        env_path = Path(env_cache_dir).expanduser()
        if env_path not in cache_roots:
            cache_roots.append(env_path)

    for cache_root in cache_roots:
        pattern = f"{DATASET_CACHE_SLUG}/default/0.0.0/*/{DATASET_CACHE_SLUG}-{split}-*.arrow"
        arrow_paths = sorted(cache_root.glob(pattern))
        if arrow_paths:
            return arrow_paths

    return []


def load_vqav2(split: str, datasets_cache_dir: str | None = None):
    try:
        from datasets import Dataset, concatenate_datasets, load_dataset
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Run `uv sync` before executing this script.") from exc

    arrow_paths = find_local_arrow_paths(split, datasets_cache_dir=datasets_cache_dir)
    if arrow_paths:
        print(f"[load] using local arrow cache with {len(arrow_paths)} shard(s) from {arrow_paths[0].parent}")
        shard_datasets = [Dataset.from_file(str(path)) for path in arrow_paths]
        if len(shard_datasets) == 1:
            return shard_datasets[0]
        return concatenate_datasets(shard_datasets)

    print("[load] local arrow cache not found, falling back to Hugging Face dataset builder")
    return load_dataset(DATASET, name=CONFIG, split=split)


def cast_image_column_without_decode(dataset):
    try:
        from datasets import Image
    except ImportError as exc:
        raise RuntimeError("The `datasets` package is required. Run `uv sync` before executing this script.") from exc

    return dataset.cast_column("image", Image(decode=False))


def parse_ground_truth(answer: object) -> int | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer
    if isinstance(answer, float):
        if answer.is_integer():
            return int(answer)
        return None

    text = str(answer).strip()
    if not text:
        return None
    if text.startswith("-"):
        digits = text[1:]
        if digits.isdigit():
            return int(text)
        return None
    if text.isdigit():
        return int(text)
    return None


def format_answer_progress(answer_counts: Dict[int, int]) -> str:
    return ", ".join(f"{answer}:{answer_counts[answer]}" for answer in sorted(answer_counts))


def select_balanced_indices(
    dataset,
    answer_range: int,
    samples_per_answer: int,
    max_rows: int | None = None,
) -> Tuple[List[int], Dict[int, int], int, int, bool]:
    target_answers = list(range(answer_range + 1))
    answer_counts = {answer: 0 for answer in target_answers}
    selected_indices: List[int] = []
    total_rows = int(dataset.num_rows)
    scan_limit = min(total_rows, max_rows) if max_rows is not None else total_rows
    progress_every = max(1000, scan_limit // 20) if scan_limit else 1

    answer_types = dataset["answer_type"]
    ground_truths = dataset["multiple_choice_answer"]

    for idx in range(scan_limit):
        if answer_types[idx] == "number":
            ground_truth = parse_ground_truth(ground_truths[idx])
            if ground_truth is not None and 0 <= ground_truth <= answer_range:
                if answer_counts[ground_truth] < samples_per_answer:
                    selected_indices.append(idx)
                    answer_counts[ground_truth] += 1

        scanned_rows = idx + 1
        if scanned_rows % progress_every == 0 or scanned_rows == scan_limit:
            print(
                f"[index] scanned={scanned_rows}/{scan_limit} "
                f"selected={len(selected_indices)} counts={format_answer_progress(answer_counts)}"
            )

        if all(answer_counts[answer] >= samples_per_answer for answer in target_answers):
            print(
                f"[index] target reached after scanning {scanned_rows}/{scan_limit} rows "
                f"counts={format_answer_progress(answer_counts)}"
            )
            return selected_indices, answer_counts, scanned_rows, total_rows, True

    print(
        f"[index] finished scanning {scan_limit}/{total_rows} rows "
        f"counts={format_answer_progress(answer_counts)}"
    )
    return selected_indices, answer_counts, scan_limit, total_rows, False


def extension_from_format(image_format: str | None) -> str:
    if not image_format:
        return ".jpg"
    normalized = image_format.upper()
    if normalized == "JPEG":
        return ".jpg"
    if normalized == "PNG":
        return ".png"
    if normalized == "WEBP":
        return ".webp"
    if normalized == "BMP":
        return ".bmp"
    if normalized == "GIF":
        return ".gif"
    if normalized == "TIFF":
        return ".tiff"
    return ".jpg"


def derive_extension_from_image_asset(image_asset: Dict[str, Any]) -> str:
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

    raise RuntimeError(f"Unable to save image asset for {target.name}: missing both path and bytes")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch VQAv2 validation rows with answer_type=number and save questions + images."
    )
    parser.add_argument(
        "--output-dir",
        default="inputs/vqav2_number_val",
        help="Directory under inputs/ to store filtered questions and images.",
    )
    parser.add_argument("--split", default=SPLIT, help="Dataset split to load. Default: validation.")
    parser.add_argument(
        "--datasets-cache-dir",
        default=None,
        help="Optional local HF datasets cache root. If set, cached arrow shards are used before any remote load.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Concurrent image save workers.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on scanned validation rows.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on saved samples after balanced selection.",
    )
    parser.add_argument(
        "--answer-range",
        type=int,
        default=9,
        help="Only keep samples whose integer ground truth is between 0 and this value, inclusive.",
    )
    parser.add_argument(
        "--samples-per-answer",
        type=int,
        default=30,
        help="Maximum number of samples to keep for each answer in the inclusive answer range.",
    )
    parser.add_argument("--skip-images", action="store_true", help="Only save question metadata without saving images.")
    args = parser.parse_args()

    if args.answer_range < 0:
        parser.error("--answer-range must be >= 0")
    if args.samples_per_answer <= 0:
        parser.error("--samples-per-answer must be > 0")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be > 0")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading dataset={DATASET} config={CONFIG} split={args.split}")
    dataset = load_vqav2(args.split, datasets_cache_dir=args.datasets_cache_dir)
    print(f"[load] loaded {dataset.num_rows} rows")

    selected_indices, scan_counts, scanned_rows, total_rows, target_reached = select_balanced_indices(
        dataset,
        answer_range=args.answer_range,
        samples_per_answer=args.samples_per_answer,
        max_rows=args.max_rows,
    )

    if args.max_samples is not None and len(selected_indices) > args.max_samples:
        print(
            f"[index] truncating selected rows from {len(selected_indices)} to max_samples={args.max_samples}. "
            "This may make the final answer distribution unbalanced."
        )
        selected_indices = selected_indices[: args.max_samples]

    if not selected_indices:
        raise RuntimeError("No rows matched the requested answer range and per-answer sampling constraints.")

    print(f"[select] selected_rows={len(selected_indices)} scanned_rows={scanned_rows}/{total_rows}")

    metadata_dataset = dataset.remove_columns(["image"]).select(selected_indices)
    question_records: List[Dict[str, object]] = []
    first_index_for_image: Dict[int, int] = {}
    final_answer_counts: Counter[int] = Counter()
    record_progress_every = max(100, len(selected_indices) // 10) if len(selected_indices) > 100 else len(selected_indices)

    for record_idx, (source_idx, row) in enumerate(zip(selected_indices, metadata_dataset), start=1):
        image_id = int(row["image_id"])
        ground_truth = parse_ground_truth(row["multiple_choice_answer"])
        if ground_truth is not None:
            final_answer_counts[ground_truth] += 1
        if image_id not in first_index_for_image:
            first_index_for_image[image_id] = source_idx

        question_records.append(
            {
                "question_id": int(row["question_id"]),
                "image_id": image_id,
                "question": row["question"],
                "question_type": row["question_type"],
                "answer_type": row["answer_type"],
                "multiple_choice_answer": row["multiple_choice_answer"],
                "answers": row["answers"],
                "image_file": "",
            }
        )

        if record_progress_every and (record_idx % record_progress_every == 0 or record_idx == len(selected_indices)):
            print(
                f"[records] built={record_idx}/{len(selected_indices)} "
                f"unique_images={len(first_index_for_image)}"
            )

    raw_image_dataset = cast_image_column_without_decode(dataset)
    image_ids_in_order = list(first_index_for_image)
    image_source_indices = [first_index_for_image[image_id] for image_id in image_ids_in_order]
    image_asset_dataset = raw_image_dataset.select(image_source_indices)
    image_assets: Dict[int, Dict[str, Any]] = {}
    image_filenames: Dict[int, str] = {}

    print(f"[images] resolving filenames for {len(image_ids_in_order)} unique images")
    for image_id, row in zip(image_ids_in_order, image_asset_dataset):
        image_asset = row["image"]
        filename = f"{image_id:012d}{derive_extension_from_image_asset(image_asset)}"
        image_assets[image_id] = image_asset
        image_filenames[image_id] = filename

    for record in question_records:
        record["image_file"] = f"images/{image_filenames[int(record['image_id'])]}"

    downloaded_count = 0
    if args.skip_images:
        print(f"[images] skipped saving {len(image_assets)} images")
    else:
        image_progress_every = max(25, len(image_assets) // 10) if len(image_assets) > 25 else len(image_assets)
        print(f"[images] saving {len(image_assets)} images with workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(save_image_asset, image_assets[image_id], image_dir / image_filenames[image_id]): image_id
                for image_id in image_ids_in_order
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                downloaded_count += 1
                if image_progress_every and (completed % image_progress_every == 0 or completed == len(futures)):
                    print(f"[images] saved={completed}/{len(futures)}")

    answer_summary = {
        str(answer): final_answer_counts.get(answer, 0)
        for answer in range(args.answer_range + 1)
    }
    missing_answers = {
        str(answer): args.samples_per_answer - final_answer_counts.get(answer, 0)
        for answer in range(args.answer_range + 1)
        if final_answer_counts.get(answer, 0) < args.samples_per_answer
    }
    if missing_answers:
        print(f"[warn] some answers did not reach the requested quota: {missing_answers}")

    print(
        f"[summary] collected {len(question_records)} questions across {len(image_filenames)} unique images "
        f"answer_counts={answer_summary}"
    )

    jsonl_path = output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in question_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        output_dir / "summary.json",
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": args.split,
            "filter": {
                "answer_type": "number",
                "ground_truth_min": 0,
                "ground_truth_max": args.answer_range,
            },
            "num_questions": len(question_records),
            "num_unique_images": len(image_filenames),
            "questions_file": "questions.jsonl",
            "images_dir": "images",
            "images_downloaded": not args.skip_images,
            "max_rows": args.max_rows,
            "max_samples": args.max_samples,
            "samples_per_answer": args.samples_per_answer,
            "answer_counts": answer_summary,
            "scanned_rows": scanned_rows,
            "scan_counts": {str(answer): scan_counts[answer] for answer in sorted(scan_counts)},
            "target_reached": target_reached,
        },
    )

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(f"[done] images in {image_dir}")


if __name__ == "__main__":
    main()
