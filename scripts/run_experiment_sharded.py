"""Multi-GPU shard fan-out driver for run_experiment.py.

Splits a sample list into K shards (round-robin), launches K child processes
each pinned to one GPU via ``CUDA_VISIBLE_DEVICES``, waits for completion,
then concatenates per-shard predictions and recomputes a unified summary.

Output layout::

    outputs/<exp>/<model>/<timestamp>/
        predictions.jsonl
        predictions.csv
        summary.json
        _shards/
            shard0/predictions.jsonl + summary.json
            shard1/...
            shard2/...

Greedy decoding (temperature == 0) makes the merged set byte-identical to a
single-process run on the same model + samples.

Example::

    uv run python scripts/run_experiment_sharded.py \
        --config configs/experiment_e5e_tallyqa_full.yaml \
        --model llava-onevision-qwen2-7b-ov \
        --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Make `vlm_anchor.*` importable when this script runs as `python scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_anchor.metrics import summarize_experiment  # noqa: E402
from vlm_anchor.utils import dump_csv, dump_json, dump_jsonl, ensure_dir, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, help="Single model name from config.")
    p.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids, e.g. '0,1,2'. K = number of shards.",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output-root",
        default=None,
        help="Override output_root from config (rare).",
    )
    p.add_argument(
        "--timestamp",
        default=None,
        help="Force a specific timestamp dirname. Default = now().",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    num_shards = len(gpus)
    if num_shards < 2:
        raise SystemExit(
            "Sharded driver requires at least 2 GPUs (use run_experiment.py directly for K=1)"
        )

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    output_root = Path(args.output_root or cfg["output_root"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    experiment_root = ensure_dir(output_root / cfg_path.stem)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    final_dir = ensure_dir(experiment_root / args.model / timestamp)
    shards_root = ensure_dir(final_dir / "_shards")

    print(f"[driver] config={cfg_path.name} model={args.model} gpus={gpus} ts={timestamp}")
    print(f"[driver] final dir: {final_dir}")

    procs: list[tuple[int, str, subprocess.Popen]] = []
    for i, gpu in enumerate(gpus):
        shard_dir = ensure_dir(shards_root / f"shard{i}")
        log_path = shard_dir / "stdout.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_experiment.py",
            "--config",
            str(cfg_path),
            "--models",
            args.model,
            "--shard-idx",
            str(i),
            "--num-shards",
            str(num_shards),
            "--output-dir",
            str(shard_dir),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[driver] launch shard {i} on GPU {gpu} -> {log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((i, gpu, proc))

    failed: list[int] = []
    for i, gpu, proc in procs:
        rc = proc.wait()
        print(f"[driver] shard {i} (GPU {gpu}) returned {rc}")
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"[driver] shard(s) failed: {failed}; logs in {shards_root}")

    print("[driver] merging shards ...")
    merged: list[dict] = []
    for i in range(num_shards):
        shard_pred = shards_root / f"shard{i}" / "predictions.jsonl"
        if not shard_pred.exists():
            raise SystemExit(f"[driver] missing shard predictions: {shard_pred}")
        with shard_pred.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    merged.append(json.loads(line))

    if not merged:
        raise SystemExit("[driver] merged record set is empty")

    # Sort by (sample_instance_index, condition) so the unified file has a
    # deterministic order independent of shard arrival.
    cond_order = {
        "target_only": 0,
        "target_plus_irrelevant_number": 1,
        "target_plus_irrelevant_number_S0": 1,
        "target_plus_irrelevant_number_S1": 1,
        "target_plus_irrelevant_number_S2": 1,
        "target_plus_irrelevant_number_masked": 2,
        "target_plus_irrelevant_number_masked_S0": 2,
        "target_plus_irrelevant_number_masked_S1": 2,
        "target_plus_irrelevant_number_masked_S2": 2,
        "target_plus_irrelevant_neutral": 3,
    }
    merged.sort(
        key=lambda r: (
            r.get("sample_instance_index") if r.get("sample_instance_index") is not None else 10**9,
            cond_order.get(r.get("condition", ""), 99),
        )
    )

    dump_jsonl(merged, final_dir / "predictions.jsonl")
    dump_csv(merged, final_dir / "predictions.csv")
    summary = summarize_experiment(merged)
    dump_json(summary, final_dir / "summary.json")
    print(f"[driver] wrote {len(merged)} merged records to {final_dir}")
    print(f"[driver] summary: {json.dumps(summary, ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
