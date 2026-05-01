"""e6_compute_subspace.py — per-layer SVD from calibrate-subspace D matrices.

Reads D_wrong.pt from calibration_<tag>/ directories, pools by scope,
runs thin SVD per layer, saves top-K right singular vectors.

Output:
  outputs/e6_steering/<model>/_subspace/subspace_<scope>_K<K>.pt
    shape: (n_layers, K, d_model)  dtype: float32
  outputs/e6_steering/<model>/_subspace/singular_values_<scope>.csv
    columns: layer, sv_0..sv_{K-1}

Usage:
  uv run python scripts/e6_compute_subspace.py \\
      --model llava-next-interleaved-7b --scope pooled

  # single dataset (no GPU needed — pure CPU SVD)
  uv run python scripts/e6_compute_subspace.py \\
      --model llava-next-interleaved-7b --scope tally
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scope", default="pooled",
                    help="'pooled' (cats all tags) or a single tag like 'vqa'/'tally'/'chartqa'.")
    ap.add_argument("--K-max", type=int, default=16,
                    help="Number of singular vectors to retain per layer.")
    ap.add_argument("--tags", default=None,
                    help="Comma-separated calibration tags. Multi-entry → pool. "
                         "Single-entry → load that one tag's D_wrong.pt. "
                         "Omit to default to: scope=='pooled' → vqa,tally,chartqa "
                         "(legacy); else → [scope].")
    return ap.parse_args()


def _load_D_wrong(model: str, tag: str) -> torch.Tensor | None:
    p = (PROJECT_ROOT / "outputs" / "e6_steering" / model
         / f"calibration_{tag}" / "D_wrong.pt")
    if not p.exists():
        print(f"[warn] {p} not found — skipping tag '{tag}'")
        return None
    D = torch.load(p, weights_only=True).float()
    print(f"[load] {tag}: D_wrong shape={tuple(D.shape)}")
    return D


def main():
    args = _parse_args()
    base = PROJECT_ROOT / "outputs" / "e6_steering" / args.model

    if args.tags is not None:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    elif args.scope == "pooled":
        tags = ["vqa", "tally", "chartqa"]
    else:
        tags = [args.scope]
    pool_mode = len(tags) > 1
    if pool_mode:
        Ds = [d for t in tags if (d := _load_D_wrong(args.model, t)) is not None]
        if not Ds:
            raise RuntimeError(f"no D_wrong.pt found for tags={tags}")
        D = torch.cat(Ds, dim=0)
    else:
        D = _load_D_wrong(args.model, tags[0])
        if D is None:
            raise RuntimeError(f"D_wrong.pt not found for tag '{tags[0]}'")

    n_total, n_layers, d_model = D.shape
    K = min(args.K_max, n_total - 1, d_model)
    print(f"[svd] D pooled shape={tuple(D.shape)}; computing K={K} per layer")

    V_all = torch.zeros(n_layers, K, d_model, dtype=torch.float32)
    sv_rows: list[dict] = []

    for L in range(n_layers):
        D_L = D[:, L, :]  # (n_total, d_model)
        # Thin SVD; Vh rows are right singular vectors (top row = largest sv)
        _, S, Vh = torch.linalg.svd(D_L, full_matrices=False)
        V_all[L, :K, :] = Vh[:K, :]
        svs = S[:K].tolist()
        row: dict = {"layer": L}
        for k, sv in enumerate(svs):
            row[f"sv_{k}"] = f"{sv:.6f}"
        sv_rows.append(row)
        if L % 8 == 0 or L == n_layers - 1:
            s1 = S[1].item() if len(S) > 1 else 0.0
            print(f"  layer {L:3d}: S[0]={S[0].item():.4f}  S[1]={s1:.4f}")

    out_dir = base / "_subspace"
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / f"subspace_{args.scope}_K{K}.pt"
    torch.save(V_all, pt_path)
    print(f"[save] {pt_path}  shape={tuple(V_all.shape)}")

    csv_path = out_dir / f"singular_values_{args.scope}.csv"
    fieldnames = ["layer"] + [f"sv_{k}" for k in range(K)]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sv_rows)
    print(f"[save] {csv_path}")
    print(f"[done] subspace for scope='{args.scope}' written to {out_dir}")


if __name__ == "__main__":
    main()
