"""Compare two SVD subspaces via principal angles.

Used to verify that the sharded calibrate-subspace driver produces a top-K
basis aligned (up to rotation) with a single-process baseline. With greedy
decoding the underlying D-matrix differs only by a few correct-base rows
(round-robin slicing + per-shard cap interactions), so the top-K subspace
should coincide up to rotation: principal-angle cosines all ≈ 1.

Usage::

    uv run python scripts/check_subspace_alignment.py \
        --baseline outputs/e6_steering/<model>/calibration_infographicvqa/v.pt \
        --candidate outputs/e6_steering/<model>/calibration_infographicvqa_sharded_smoke/v.pt \
        --K 8

Reports per-layer mean cosine of principal angles between the top-K
right-singular subspaces of D_all (loaded from the sibling D_all.pt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="Path to baseline calibration directory or D_all.pt.")
    p.add_argument("--candidate", required=True,
                   help="Path to candidate (sharded) calibration directory or D_all.pt.")
    p.add_argument("--K", type=int, default=8,
                   help="Top-K basis size to compare (default 8).")
    p.add_argument("--threshold", type=float, default=0.99,
                   help="Min cosine of principal angle to pass alignment (default 0.99).")
    return p.parse_args()


def _load_D_all(path_arg: str) -> torch.Tensor:
    p = Path(path_arg)
    if p.is_dir():
        p = p / "D_all.pt"
    return torch.load(p, weights_only=False)


def _per_layer_topK_basis(D: torch.Tensor, K: int) -> torch.Tensor:
    """SVD per layer; return (n_layers, K, d) right-singular basis."""
    n, n_layers, d = D.shape
    bases = torch.empty(n_layers, K, d, dtype=D.dtype)
    for L in range(n_layers):
        # rows of D[:, L, :] are the per-pair diffs at layer L
        U, S, Vh = torch.linalg.svd(D[:, L, :], full_matrices=False)
        bases[L] = Vh[:K]
    return bases


def main() -> None:
    args = parse_args()
    D_base = _load_D_all(args.baseline).float()
    D_cand = _load_D_all(args.candidate).float()
    print(f"baseline D_all  shape: {tuple(D_base.shape)}")
    print(f"candidate D_all shape: {tuple(D_cand.shape)}")
    if D_base.shape[1:] != D_cand.shape[1:]:
        raise SystemExit("D_all per-layer/per-d shape mismatch — incompatible models")

    K = args.K
    V_base = _per_layer_topK_basis(D_base, K)
    V_cand = _per_layer_topK_basis(D_cand, K)

    n_layers = V_base.shape[0]
    cosines = torch.empty(n_layers, K)
    for L in range(n_layers):
        # Principal angles: cosines = singular values of V_base @ V_cand^T
        M = V_base[L] @ V_cand[L].T  # (K, K)
        # SVD of M: singular values = cosines of principal angles
        sv = torch.linalg.svdvals(M)
        cosines[L] = sv

    # Per-layer summary: min cosine across K (the "worst" angle).
    per_layer_min = cosines.min(dim=1).values
    overall_min = per_layer_min.min().item()
    overall_mean = cosines.mean().item()
    print(f"\nPer-layer min cos(principal angle), K={K}:")
    for L in range(n_layers):
        flag = "OK" if per_layer_min[L] >= args.threshold else "FAIL"
        print(f"  L{L:2d}  min={per_layer_min[L].item():.6f}  "
              f"mean={cosines[L].mean().item():.6f}  [{flag}]")
    print(f"\nOverall min cosine: {overall_min:.6f}")
    print(f"Overall mean cosine: {overall_mean:.6f}")
    print(f"Threshold: {args.threshold}")
    if overall_min >= args.threshold:
        print("PASS — sharded subspace aligned with baseline up to rotation.")
    else:
        print("FAIL — subspaces diverge; sharding changed the basis materially.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
