#!/usr/bin/env python3
"""Convert the local InfographicVQA *validation* parquet shards into our
standard VQAv2-like layout.

Assumes the raw lmms-lab/DocVQA InfographicVQA config has been downloaded to
``inputs/infographicvqa_val/InfographicVQA/validation-*.parquet``.

Filters applied at fetch time:
  - first answer parses as a positive integer in [1, max_gt]
  - non-integer answers (floats, text, multi-span text) are dropped
  - test split is not consumed (labels are private)

Output layout (snapshot files coexist with the raw InfographicVQA/ subdir):

    <output_dir>/
        images/000000000000.png
        questions.jsonl
        summary.json
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image as PILImage


def parse_positive_int(answer: object, max_value: int) -> int | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 1 <= answer <= max_value else None
    if isinstance(answer, float):
        if not answer.is_integer():
            return None
        v = int(answer)
        return v if 1 <= v <= max_value else None
    text = str(answer).strip().replace(",", "").replace("%", "")
    if not text:
        return None
    try:
        f = float(text)
    except ValueError:
        return None
    if not f.is_integer():
        return None
    v = int(f)
    return v if 1 <= v <= max_value else None


def gt_bin(v: int) -> str:
    if v <= 8: return "(0,8]"
    if v <= 20: return "(8,20]"
    if v <= 100: return "(20,100]"
    if v <= 1000: return "(100,1k]"
    return "(1k,10k]"


def derive_extension(image_format: str | None) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tiff"}
    if not image_format:
        return ".png"
    return mapping.get(image_format.upper(), ".png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert InfographicVQA validation parquet to inputs/infographicvqa_val snapshot.",
    )
    parser.add_argument(
        "--raw-dir",
        default="inputs/infographicvqa_val/InfographicVQA",
        help="Directory holding validation-*.parquet shards.",
    )
    parser.add_argument("--output-dir", default="inputs/infographicvqa_val")
    parser.add_argument("--max-gt", type=int, default=10000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    raw_dir = Path(args.raw_dir)
    shards = sorted(raw_dir.glob("validation-*.parquet"))
    if not shards:
        parser.error(f"no validation-*.parquet under {raw_dir}")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()
    bin_counter: Counter = Counter()
    answer_type_kept: Counter = Counter()

    next_idx = 0
    for shard in shards:
        print(f"[load] {shard.name}")
        t = pq.read_table(shard)
        cols = ["questionId", "question", "answers", "answer_type", "image"]
        rows = t.select(cols).to_pylist()
        for row in rows:
            ans_list = row.get("answers") or []
            if not ans_list:
                drop_reasons["no_answer"] += 1
                continue
            v = parse_positive_int(ans_list[0], args.max_gt)
            if v is None:
                drop_reasons["non_int_or_oob"] += 1
                continue

            asset = row.get("image") or {}
            image_bytes = asset.get("bytes")
            if not image_bytes:
                drop_reasons["no_image_bytes"] += 1
                continue

            ext = ".png"
            try:
                with PILImage.open(io.BytesIO(bytes(image_bytes))) as im:
                    ext = derive_extension(im.format)
            except Exception:
                drop_reasons["unreadable_image"] += 1
                continue

            img_id = next_idx
            next_idx += 1
            filename = f"{img_id:012d}{ext}"
            target = image_dir / filename
            if not args.skip_images and not target.exists():
                target.write_bytes(bytes(image_bytes))

            ans_text = str(v)
            atypes = list(row.get("answer_type") or [])
            for a in atypes:
                answer_type_kept[a] += 1
            records.append(
                {
                    "question_id": int(row.get("questionId") or img_id),
                    "image_id": img_id,
                    "question": row.get("question") or "",
                    "question_type": "infographicvqa",
                    "answer_type": "number",
                    "multiple_choice_answer": ans_text,
                    "answers": [{"answer": ans_text, "answer_confidence": "yes", "answer_id": 1}],
                    "infovqa_answer_types_raw": atypes,
                    "image_file": f"images/{filename}",
                }
            )
            bin_counter[gt_bin(v)] += 1

            if args.max_samples is not None and len(records) >= args.max_samples:
                break
        if args.max_samples is not None and len(records) >= args.max_samples:
            break

    print(f"[filter] kept={len(records):,} dropped={sum(drop_reasons.values()):,} ({dict(drop_reasons)})")
    print(f"[filter] gt-bin: {dict(bin_counter)}")
    print(f"[filter] answer_types kept: {dict(answer_type_kept)}")

    jsonl_path = output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as h:
        for rec in records:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "InfographicVQA (lmms-lab/DocVQA mirror)",
        "split": "validation",
        "num_questions": len(records),
        "num_unique_images": len(records),
        "questions_file": "questions.jsonl",
        "images_dir": "images",
        "images_downloaded": not args.skip_images,
        "max_gt": args.max_gt,
        "max_samples": args.max_samples,
        "gt_bin_distribution": dict(bin_counter),
        "answer_types_kept": dict(answer_type_kept),
        "dropped": dict(drop_reasons),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
