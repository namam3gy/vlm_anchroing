#!/usr/bin/env python3
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

import requests


ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "lmms-lab/VQAv2"
CONFIG = "default"
SPLIT = "validation"
PAGE_SIZE = 100


def request_json(session: requests.Session, params: Dict[str, object], retries: int = 8) -> Dict[str, object]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(ROWS_URL, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"[rows] request failed on attempt {attempt}/{retries}: {exc}. retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch rows after {retries} attempts") from last_error


def iter_row_pages(session: requests.Session, max_rows: int | None = None) -> Iterable[Tuple[List[Dict[str, object]], int]]:
    offset = 0
    total = None
    seen = 0

    while total is None or offset < total:
        if max_rows is not None and seen >= max_rows:
            break
        payload = request_json(
            session,
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            },
        )
        rows = payload["rows"]
        total = payload["num_rows_total"]

        if not rows:
            break

        page_rows: List[Dict[str, object]] = []
        print(f"[rows] fetched offset={offset} count={len(rows)} total={total}")
        for wrapped_row in rows:
            row = wrapped_row["row"]
            seen += 1
            page_rows.append(row)
            if max_rows is not None and seen >= max_rows:
                break

        yield page_rows, total
        offset += len(rows)


def derive_extension(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix
    content_type = parse_qs(urlparse(url).query).get("response-content-type", [""])[0].lower()
    if "png" in content_type:
        return ".png"
    return ".jpg"


def download_one(image_id: int, url: str, image_dir: Path, retries: int = 8) -> Tuple[int, str]:
    ext = derive_extension(url)
    filename = f"{image_id:012d}{ext}"
    target = image_dir / filename
    if target.exists():
        return image_id, filename

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                tmp_path = target.with_suffix(target.suffix + ".part")
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                tmp_path.rename(target)
            return image_id, filename
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"[image] download failed image_id={image_id} attempt={attempt}/{retries}: {exc}. retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download image_id={image_id}") from last_error


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch VQAv2 validation rows with answer_type=number and save questions + images.")
    parser.add_argument("--output-dir", default="inputs/vqav2_number_val", help="Directory under inputs/ to store filtered questions and images.")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent image download workers.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on scanned validation rows.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on saved number-type question samples.")
    parser.add_argument("--skip-images", action="store_true", help="Only save question metadata without downloading images.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    question_records: List[Dict[str, object]] = []
    downloaded: Dict[int, str] = {}
    downloaded_count = 0
    limit_reached = False

    for page_rows, total_rows in iter_row_pages(session, max_rows=args.max_rows):
        pending_images: Dict[int, str] = {}
        for row in page_rows:
            if row["answer_type"] != "number":
                continue
            if args.max_samples is not None and len(question_records) >= args.max_samples:
                limit_reached = True
                break

            image_id = int(row["image_id"])
            image_url = row["image"]["src"]
            image_file = downloaded.get(image_id)
            if image_file is None:
                image_file = f"{image_id:012d}{derive_extension(image_url)}"
                downloaded[image_id] = image_file
                if not args.skip_images:
                    pending_images[image_id] = image_url

            question_records.append(
                {
                    "question_id": int(row["question_id"]),
                    "image_id": image_id,
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "answer_type": row["answer_type"],
                    "multiple_choice_answer": row["multiple_choice_answer"],
                    "answers": row["answers"],
                    "image_file": f"images/{image_file}",
                }
            )

        if pending_images:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(download_one, image_id, image_url, image_dir): image_id
                    for image_id, image_url in pending_images.items()
                }
                for future in as_completed(futures):
                    image_id, filename = future.result()
                    downloaded[image_id] = filename
                    downloaded_count += 1
            print(
                f"[image] downloaded batch={len(pending_images)} total_downloaded={downloaded_count} "
                f"unique_images={len(downloaded)} scanned_total={total_rows}"
            )
        if limit_reached:
            break

    print(f"[summary] collected {len(question_records)} number questions across {len(downloaded)} unique images")

    jsonl_path = output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in question_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        output_dir / "summary.json",
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "filter": {"answer_type": "number"},
            "num_questions": len(question_records),
            "num_unique_images": len(downloaded),
            "questions_file": "questions.jsonl",
            "images_dir": "images",
            "images_downloaded": not args.skip_images,
            "max_rows": args.max_rows,
            "max_samples": args.max_samples,
        },
    )

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")
    print(f"[done] images in {image_dir}")


if __name__ == "__main__":
    main()
