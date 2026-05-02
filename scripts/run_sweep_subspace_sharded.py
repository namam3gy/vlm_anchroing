"""Multi-GPU shard fan-out driver for `e6_steering_vector.py --phase sweep-subspace`.

Sweep-subspace iterates (cell × sid × cond) and writes one record per
generation to predictions.jsonl. With greedy decoding the union of K
disjoint sid-slices is byte-identical to a single-process run; merge is a
trivial concatenation.

Output layout::

    outputs/e6_steering/<model>/sweep_subspace_<dataset_tag>_<scope>/
        predictions.jsonl                # canonical, post-merge
        _shards/shard{i}/predictions.jsonl

Example::

    uv run python scripts/run_sweep_subspace_sharded.py \
        --config configs/experiment_e5e_tallyqa_full.yaml \
        --model llava-onevision-qwen2-7b-ov \
        --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
        --predictions-path outputs/experiment_e5e_tallyqa_full/.../predictions.jsonl \
        --dataset-tag tallyqa \
        --subspace-path outputs/e6_steering/<model>/_subspace/subspace_*.pt \
        --subspace-scope plotqa_infovqa_pooled_n5k \
        --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
        --max-samples 5000 \
        --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--predictions-path", required=True)
    p.add_argument("--dataset-tag", required=True)
    p.add_argument("--subspace-path", required=True)
    p.add_argument("--subspace-scope", required=True)
    p.add_argument("--sweep-layers", default="31")
    p.add_argument("--sweep-ks", default="4")
    p.add_argument("--sweep-alphas", default="1.0")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--gpus", required=True, help="Comma-separated GPU ids.")
    return p.parse_args()


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (PROJECT_ROOT / pp).resolve()
    return pp


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    K = len(gpus)
    if K < 2:
        raise SystemExit("Sharded sweep driver requires K >= 2 GPUs")

    pred_path = _resolve_path(args.predictions_path)
    subspace_path = _resolve_path(args.subspace_path)

    final_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
                 / f"sweep_subspace_{args.dataset_tag}_{args.subspace_scope}")
    final_dir.mkdir(parents=True, exist_ok=True)
    shards_root = final_dir / "_shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    print(f"[driver] sweep-subspace sharded: tag={args.dataset_tag} K={K} "
          f"final_dir={final_dir}")

    procs: list[tuple[int, str, subprocess.Popen]] = []
    for i, gpu in enumerate(gpus):
        shard_dir = shards_root / f"shard{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "stdout.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            "uv", "run", "python", "scripts/e6_steering_vector.py",
            "--phase", "sweep-subspace",
            "--model", args.model,
            "--hf-model", args.hf_model,
            "--e5c-run-dir", str(pred_path.parent),
            "--predictions-path", str(pred_path),
            "--dataset-tag", args.dataset_tag,
            "--subspace-path", str(subspace_path),
            "--subspace-scope", args.subspace_scope,
            "--sweep-layers", args.sweep_layers,
            "--sweep-ks", args.sweep_ks,
            "--sweep-alphas", args.sweep_alphas,
            "--max-samples", str(args.max_samples),
            "--max-new-tokens", str(args.max_new_tokens),
            "--config", args.config,
            "--shard-idx", str(i),
            "--num-shards", str(K),
            "--output-dir", str(shard_dir),
        ]
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
    seen_keys: set[tuple] = set()
    with final_pred.open("w", encoding="utf-8") as out_f:
        for i in range(K):
            shard_pred = shards_root / f"shard{i}" / "predictions.jsonl"
            if not shard_pred.exists():
                print(f"[driver] WARN: missing shard predictions: {shard_pred}")
                continue
            with shard_pred.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Defensive dedup on (sid, cond, L, K, alpha) — round-robin
                    # slicing should preclude dups, but resume + retry could
                    # re-emit a record under a partial-shard scenario.
                    key = (rec.get("sample_instance_id"), rec.get("condition"),
                           int(rec.get("cell_layer", -1)),
                           int(rec.get("subspace_K", -1)),
                           float(rec.get("cell_alpha", 0.0)))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out_f.write(line + "\n")
                    n_total += 1
    print(f"[driver] wrote {n_total} merged records to {final_pred}")


if __name__ == "__main__":
    main()
