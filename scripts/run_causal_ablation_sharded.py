"""Multi-GPU shard fan-out driver for `causal_anchor_ablation.py`.

Causal ablation iterates samples × conditions × ablation modes, generating
under attention-mask hooks. With greedy decoding the sample slice is the
only thing affected by sharding; merge = concat predictions.jsonl.

Output layout::

    outputs/causal_ablation/<model>/<timestamp>/
        predictions.jsonl                # canonical, post-merge
        _shards/shard{i}/predictions.jsonl

Example::

    uv run python scripts/run_causal_ablation_sharded.py \
        --model llava-onevision-qwen2-7b-ov \
        --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
        --peak-layer 14 \
        --config configs/experiment_e7_plotqa_full.yaml \
        --susceptibility-csv docs/insights/_data/susceptibility_plotqa_onevision.csv \
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--peak-layer", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--susceptibility-csv", required=True)
    p.add_argument("--top-decile-n", type=int, default=100)
    p.add_argument("--bottom-decile-n", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--modes", type=str,
        default="baseline,ablate_peak,ablate_peak_window,ablate_lower_half,ablate_upper_half,ablate_all",
    )
    p.add_argument("--gpus", required=True, help="Comma-separated GPU ids.")
    p.add_argument("--timestamp", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    K = len(gpus)
    if K < 2:
        raise SystemExit("Sharded causal driver requires K >= 2 GPUs")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    final_dir = (PROJECT_ROOT / "outputs" / "causal_ablation"
                 / args.model / timestamp)
    final_dir.mkdir(parents=True, exist_ok=True)
    shards_root = final_dir / "_shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    print(f"[driver] causal_ablation sharded: model={args.model} K={K} ts={timestamp}")

    procs: list[tuple[int, str, subprocess.Popen]] = []
    for i, gpu in enumerate(gpus):
        shard_dir = shards_root / f"shard{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "stdout.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            "uv", "run", "python", "scripts/causal_anchor_ablation.py",
            "--model", args.model,
            "--hf-model", args.hf_model,
            "--peak-layer", str(args.peak_layer),
            "--config", args.config,
            "--susceptibility-csv", args.susceptibility_csv,
            "--top-decile-n", str(args.top_decile_n),
            "--bottom-decile-n", str(args.bottom_decile_n),
            "--max-new-tokens", str(args.max_new_tokens),
            "--seed", str(args.seed),
            "--modes", args.modes,
            "--shard-idx", str(i),
            "--num-shards", str(K),
            "--output-dir", str(shard_dir),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[driver] launch shard {i} on GPU {gpu} -> {log_path}")
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs.append((i, gpu, proc))

    failed: list[int] = []
    for i, gpu, proc in procs:
        rc = proc.wait()
        print(f"[driver] shard {i} (GPU {gpu}) returned {rc}")
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"[driver] shard(s) failed: {failed}")

    print("[driver] merging shard predictions ...")
    final_pred = final_dir / "predictions.jsonl"
    n_total = 0
    with final_pred.open("w", encoding="utf-8") as out_f:
        for i in range(K):
            shard_pred = shards_root / f"shard{i}" / "predictions.jsonl"
            if not shard_pred.exists():
                print(f"[driver] WARN: missing shard predictions: {shard_pred}")
                continue
            with shard_pred.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.rstrip("\n")
                    if line:
                        out_f.write(line + "\n")
                        n_total += 1
    print(f"[driver] wrote {n_total} merged records to {final_pred}")


if __name__ == "__main__":
    main()
