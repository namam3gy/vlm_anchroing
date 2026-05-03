"""Multi-GPU shard fan-out driver for `e6_steering_vector.py --phase calibrate-subspace`.

Calibrate-subspace collects per-pair residual diffs (a-S1 minus m-S1) from a
predictions.jsonl, then feeds them to SVD to produce a top-K basis. With K
GPUs we slice eligible sids round-robin so each shard captures `ceil(max_pairs/K)`
diffs in parallel, then concatenate D_wrong / D_all tensors and run SVD once
on the merged set.

Important caveat: unlike `run_experiment_sharded.py` (greedy LM decoding,
byte-identical), calibrate sharding is **statistically equivalent, not
byte-identical**. The set of sids actually contributing to D may differ
slightly because the per-shard cap interacts with round-robin ordering. The
SVD basis V should still match the single-process basis up to rotation
(principal angles ≈ 0). Use `scripts/check_subspace_alignment.py` to verify.

Output layout::

    outputs/e6_steering/<model>/calibration_<dataset_tag>/
        D_wrong.pt + D_all.pt + v.pt + v_meta.json   (canonical, post-merge)
        _shards/shard{i}/D_wrong.pt + D_all.pt + shard_meta.json

Example::

    uv run python scripts/run_calibrate_subspace_sharded.py \
        --config configs/experiment_e7_plotqa_full.yaml \
        --model llava-onevision-qwen2-7b-ov \
        --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
        --predictions-path outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/<ts>/predictions.jsonl \
        --dataset-tag plotqa \
        --max-calibrate-pairs 2500 \
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

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--hf-model", required=True)
    p.add_argument("--predictions-path", required=True)
    p.add_argument("--dataset-tag", required=True)
    p.add_argument("--max-calibrate-pairs", type=int, required=True)
    p.add_argument("--gpus", required=True, help="Comma-separated GPU ids.")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override final calibration_<tag> directory. Default = "
             "outputs/e6_steering/<model>/calibration_<dataset_tag>",
    )
    return p.parse_args()


def _resolve_predictions_path(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (PROJECT_ROOT / pp).resolve()
    return pp


def _e5c_run_dir(predictions_path: Path) -> Path:
    return predictions_path.parent


def _save_merged(out_dir: Path, D_wrong: torch.Tensor, D_all: torch.Tensor,
                 args: argparse.Namespace, total_skipped: int,
                 total_wall: float) -> None:
    """Match _save_D_and_v invariants: D_wrong.pt, D_all.pt, v.pt, v_meta.json."""
    n_layers = int(D_all.shape[1])
    torch.save(D_all, out_dir / "D_all.pt")
    torch.save(D_wrong, out_dir / "D_wrong.pt")
    v_all = D_all.mean(0)
    v_wrong = D_wrong.mean(0) if D_wrong.shape[0] > 0 else torch.zeros_like(v_all)
    torch.save(torch.stack([v_wrong, v_all]), out_dir / "v.pt")
    sidecar = {
        "model": args.model, "hf_model": args.hf_model,
        "dataset_tag": args.dataset_tag,
        "n_wrong": int(D_wrong.shape[0]),
        "n_all": int(D_all.shape[0]),
        "n_skipped": total_skipped,
        "n_layers": n_layers,
        "d_model": int(D_all.shape[-1]),
        "D_wrong_shape": list(D_wrong.shape),
        "D_all_shape": list(D_all.shape),
        "v_index_0": "v_wrong", "v_index_1": "v_all",
        "wall_seconds": total_wall,
        "sharded": True,
        "num_shards": len(args.gpus.split(",")),
    }
    (out_dir / "v_meta.json").write_text(json.dumps(sidecar, indent=2))


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    K = len(gpus)
    if K < 2:
        raise SystemExit("Sharded calibrate driver requires K >= 2 GPUs")

    pred_path = _resolve_predictions_path(args.predictions_path)
    if args.out_dir is not None:
        final_dir = Path(args.out_dir)
        if not final_dir.is_absolute():
            final_dir = (PROJECT_ROOT / final_dir).resolve()
    else:
        final_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
                     / f"calibration_{args.dataset_tag}")
    final_dir.mkdir(parents=True, exist_ok=True)
    shards_root = final_dir / "_shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    print(f"[driver] calibrate-subspace sharded: tag={args.dataset_tag} K={K} "
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
            "--phase", "calibrate-subspace",
            "--model", args.model,
            "--hf-model", args.hf_model,
            "--e5c-run-dir", str(_e5c_run_dir(pred_path)),
            "--predictions-path", str(pred_path),
            "--dataset-tag", args.dataset_tag,
            "--max-calibrate-pairs", str(args.max_calibrate_pairs),
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

    print("[driver] merging shard tensors ...")
    Ds_wrong: list[torch.Tensor] = []
    Ds_all: list[torch.Tensor] = []
    total_skipped = 0
    total_wall = 0.0
    for i in range(K):
        shard_dir = shards_root / f"shard{i}"
        meta = json.loads((shard_dir / "shard_meta.json").read_text())
        total_skipped += meta.get("n_skipped", 0)
        total_wall = max(total_wall, float(meta.get("wall_seconds", 0)))
        if meta.get("empty"):
            continue
        Dw = torch.load(shard_dir / "D_wrong.pt", weights_only=False)
        Da = torch.load(shard_dir / "D_all.pt", weights_only=False)
        if Dw.numel() > 0:
            Ds_wrong.append(Dw)
        if Da.numel() > 0:
            Ds_all.append(Da)

    if not Ds_all:
        raise SystemExit("[driver] no calibration pairs collected across shards")

    D_all = torch.cat(Ds_all, dim=0)
    D_wrong = (torch.cat(Ds_wrong, dim=0) if Ds_wrong
               else torch.zeros(0, D_all.shape[1], D_all.shape[2], dtype=D_all.dtype))
    print(f"[driver] merged D_wrong={tuple(D_wrong.shape)} "
          f"D_all={tuple(D_all.shape)}")

    _save_merged(final_dir, D_wrong, D_all, args, total_skipped, total_wall)
    print(f"[driver] wrote merged calibration to {final_dir}")


if __name__ == "__main__":
    main()
