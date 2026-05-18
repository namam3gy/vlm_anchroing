"""Promote a shard-run output into the H1 cross_model_cross_dataset tree.

After `run_experiment_sharded.py` finishes, it writes the merged
predictions to:
  outputs/paper2/_shard_runs/<exp_stem>/<model>/<timestamp>/
      predictions.{jsonl,csv}
      summary.json
      _shards/shard{0..K}/...

This script picks up the most-recent run for the requested
(dataset, model) pair, copies the merged predictions to the canonical
H1 cell path, and touches `_done.marker`:

  outputs/paper2/cross_model_cross_dataset/predictions/<dataset>/<model>/
      predictions.{jsonl,csv}
      summary.json
      _done.marker

Usage:
    .venv/bin/python scripts/_h1_promote_shard.py \\
        --dataset plotqa --model qwen2.5-vl-32b-instruct
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARD_RUNS = REPO / "outputs" / "paper2" / "_shard_runs"
H1_PREDS   = REPO / "outputs" / "paper2" / "cross_model_cross_dataset" / "predictions"

# Slug → config stem mapping (matches launch_h1_baseline.py DATASETS).
DATASET_CONFIG_STEM = {
    "mathvista":      "experiment_e5e_mathvista_full",
    "chartqa":        "experiment_e5e_chartqa_full",
    "infographicvqa": "experiment_e7_infographicvqa_full",
    "plotqa":         "experiment_e7_plotqa_full",
    "tallyqa":        "experiment_e5e_tallyqa_full",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG_STEM))
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    exp_stem = DATASET_CONFIG_STEM[args.dataset]
    src_root = SHARD_RUNS / exp_stem / args.model
    if not src_root.exists():
        sys.exit(f"No shard run found at {src_root}")

    runs = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not runs:
        sys.exit(f"No timestamped runs under {src_root}")
    src = runs[-1]  # most recent
    print(f"src: {src}")

    pred_csv = src / "predictions.csv"
    pred_jsonl = src / "predictions.jsonl"
    summary = src / "summary.json"
    for f in (pred_csv, pred_jsonl, summary):
        if not f.exists():
            sys.exit(f"missing {f}")

    dst = H1_PREDS / args.dataset / args.model
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_csv,   dst / "predictions.csv")
    shutil.copy2(pred_jsonl, dst / "predictions.jsonl")
    shutil.copy2(summary,    dst / "summary.json")
    (dst / "_done.marker").touch()
    print(f"dst: {dst}")
    print(f"  predictions.csv:    {(dst / 'predictions.csv').stat().st_size:,} bytes")
    print(f"  predictions.jsonl:  {(dst / 'predictions.jsonl').stat().st_size:,} bytes")
    print(f"  summary.json:       OK")
    print(f"  _done.marker:       touched")


if __name__ == "__main__":
    main()
