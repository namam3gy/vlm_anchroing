#!/usr/bin/env python3
"""Convert the local PlotQA *test* split into our standard VQAv2-like layout.

Assumes the raw release has already been unzipped to ``inputs/plotqa_raw/PlotQA/test/``
(qa_pairs_V1.json, png.tar.gz, annotations.json). Only V1 is consumed; V2 is
ignored to keep the question distribution template-controlled.

Filters applied at fetch time:
  - template in {data_retrieval, min_max, arithmetic}
  - answer is a positive integer in [1, max_gt]
  - structural (yes/no), comparison, and compound templates are dropped
  - non-integer arithmetic results (floats like "1.32") are dropped

Output layout (matches inputs/chartqa_test/, etc.):

    <output_dir>/
        images/000000000000.png
        questions.jsonl
        summary.json
"""
from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


KEPT_TEMPLATES = {"data_retrieval", "min_max", "arithmetic"}


def parse_positive_int(answer: object, max_value: int) -> int | None:
    """Return int(answer) if it is a whole positive integer in [1, max_value]; else None."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 1 <= answer <= max_value else None
    if isinstance(answer, float):
        if not answer.is_integer():
            return None
        v = int(answer)
        return v if 1 <= v <= max_value else None
    text = str(answer).strip()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PlotQA test (V1) raw to inputs/plotqa_test snapshot.",
    )
    parser.add_argument("--raw-dir", default="inputs/plotqa_raw/PlotQA/test")
    parser.add_argument("--output-dir", default="inputs/plotqa_test")
    parser.add_argument("--max-gt", type=int, default=10000, help="Max gt to keep (anchor cap).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total kept Q-A records (post-filter). Stratified across gt-bins if --stratified.",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="If set, sample evenly across gt bins (0,8] (8,20] (20,100] (100,1k] (1k,10k].",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    qa_path = raw_dir / "qa_pairs_V1.json"
    tar_path = raw_dir / "png.tar.gz"
    if not qa_path.exists():
        parser.error(f"missing {qa_path}")
    if not tar_path.exists() and not args.skip_images:
        parser.error(f"missing {tar_path}")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {qa_path}")
    with qa_path.open() as f:
        qa = json.load(f)["qa_pairs"]
    print(f"[load] {len(qa):,} raw V1 Q-A pairs")

    # Filter.
    kept: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()
    template_counts: Counter = Counter()
    for r in qa:
        tmpl = r.get("template")
        if tmpl not in KEPT_TEMPLATES:
            drop_reasons["template"] += 1
            continue
        v = parse_positive_int(r.get("answer"), args.max_gt)
        if v is None:
            drop_reasons["non_int_or_oob"] += 1
            continue
        kept.append({**r, "_gt_int": v})
        template_counts[tmpl] += 1
    print(
        f"[filter] kept={len(kept):,} dropped={sum(drop_reasons.values()):,} "
        f"({dict(drop_reasons)})"
    )
    print(f"[filter] templates: {dict(template_counts)}")

    # Optional sampling cap.
    rng = random.Random(args.seed)
    if args.max_samples is not None and args.max_samples < len(kept):
        if args.stratified:
            buckets: defaultdict[str, list] = defaultdict(list)
            for r in kept:
                buckets[gt_bin(r["_gt_int"])].append(r)
            per_bucket = max(1, args.max_samples // len(buckets))
            sampled: List[Dict[str, Any]] = []
            for b, items in buckets.items():
                rng.shuffle(items)
                sampled.extend(items[:per_bucket])
            rng.shuffle(sampled)
            kept = sampled[: args.max_samples]
            print(f"[sample] stratified into {len(buckets)} bins, ~{per_bucket}/bin → {len(kept):,}")
        else:
            rng.shuffle(kept)
            kept = kept[: args.max_samples]
            print(f"[sample] random subset → {len(kept):,}")

    # Build records and collect needed image_indices.
    records: List[Dict[str, Any]] = []
    needed_images: set[int] = set()
    bin_counter: Counter = Counter()
    for r in kept:
        v = r["_gt_int"]
        img_idx = int(r["image_index"])
        needed_images.add(img_idx)
        bin_counter[gt_bin(v)] += 1
        ans_text = str(v)
        records.append(
            {
                "question_id": int(r["question_id"]),
                "image_id": img_idx,
                "question": r["question_string"],
                "question_type": "plotqa_" + str(r["template"]),
                "answer_type": "number",
                "multiple_choice_answer": ans_text,
                "answers": [{"answer": ans_text, "answer_confidence": "yes", "answer_id": 1}],
                "plotqa_template": r["template"],
                "plotqa_qid": r.get("qid"),
                "image_file": f"images/{img_idx:012d}.png",
            }
        )
    print(f"[records] {len(records):,} records referencing {len(needed_images):,} images")
    print(f"[records] gt-bin distribution: {dict(bin_counter)}")

    # Extract images from tar.gz (single streaming pass).
    if args.skip_images:
        print(f"[images] skipped extraction of {len(needed_images):,} images")
    else:
        wanted_names = {f"png/{i}.png": i for i in needed_images}
        extracted = 0
        progress_every = max(50, len(needed_images) // 10)
        print(f"[images] streaming {tar_path} for {len(needed_images):,} images")
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf:
                if member.name not in wanted_names:
                    continue
                idx = wanted_names[member.name]
                target = image_dir / f"{idx:012d}.png"
                if target.exists():
                    extracted += 1
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                target.write_bytes(fobj.read())
                extracted += 1
                if extracted % progress_every == 0:
                    print(f"[images] extracted={extracted:,}/{len(needed_images):,}")
                if extracted >= len(needed_images):
                    break
        print(f"[images] extracted={extracted:,}/{len(needed_images):,}")

    jsonl_path = output_dir / "questions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as h:
        for rec in records:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "PlotQA",
        "split": "test",
        "version": "V1",
        "num_questions": len(records),
        "num_unique_images": len(needed_images),
        "questions_file": "questions.jsonl",
        "images_dir": "images",
        "images_downloaded": not args.skip_images,
        "max_gt": args.max_gt,
        "max_samples": args.max_samples,
        "stratified": bool(args.stratified),
        "templates_kept": sorted(KEPT_TEMPLATES),
        "gt_bin_distribution": dict(bin_counter),
        "template_counts_pre_sample": dict(template_counts),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {jsonl_path}")
    print(f"[done] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
