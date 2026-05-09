"""analyze_e6_multilayer_redundancy.py — H-A test for P0-2 graceful-degradation.

Hypothesis H-A: §5.2 multi-layer redundancy (anchor signal redundantly written by
multiple attention layers) PREDICTS the §6.4 high-effective-rank residual at L=26.
Each attention layer writes anchor information into the residual along a slightly
different direction; these directions accumulate at L=26 → naturally high effective
rank, K=8 captures the multi-layer-redundant direction stack.

Test:
  - V_K[L=26] = 8 × d_model = top-8 right singular vectors of cumulative
    D[:, L=26, :] (already on disk under outputs/e6_steering/<model>/_subspace/).
  - u_L_inc = top-1 right singular vector of INCREMENTAL ΔD[:, L, :] =
    D[:, L, :] - D[:, L-1, :] (anchor-specific contribution of layer L).
  - Subspace alignment: ||V_K^T · u_L_inc||_2 ∈ [0, 1].
  - Per-vector cosine similarity: |v_k^T · u_L_inc| for k=0..7.

Pre-registered thresholds:
  - Random baseline: sqrt(K/d_model) = sqrt(8/3584) ≈ 0.047. Threshold 0.30 =
    ~6× chance level.
  - H-A CONFIRMED: ≥ 4 distinct early layers L (excluding L=26 itself) with
    ||V_K^T u_L_inc||_2 > 0.30 → multi-layer accumulation confirmed.
  - H-A WEAK: ≥ 2 distinct early layers.
  - H-A FALSIFIED: ≤ 1 layer (V_K aligns only with L=26's own direction — pure
    cumulative, not multi-layer).

Output:
  docs/insights/_data/p0_2_HA_subspace_alignment.csv
    columns: layer, top_inc_sigma, sub_align_inc, sub_align_cum,
             v_0..v_7 (cosine sim per v_k with u_L_inc)
  docs/insights/_data/p0_2_HA_verdict.json
  docs/figures/P0-2_HA_subspace_alignment.png  — Figure 3
  docs/figures/P0-2_HA_per_vk_heatmap.png      — Figure 4

Usage:
  uv run python scripts/analyze_e6_multilayer_redundancy.py \\
      --model llava-onevision-qwen2-7b-ov \\
      --tags plotqa,infographicvqa \\
      --peak-layer 26 --K 8
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-onevision-qwen2-7b-ov")
    ap.add_argument("--tags", default="plotqa,infographicvqa")
    ap.add_argument("--peak-layer", type=int, default=26)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--strong-n", type=int, default=4,
                    help="≥ this many distinct early layers above threshold → H-A CONFIRMED.")
    ap.add_argument("--weak-n", type=int, default=2,
                    help="≥ this many distinct early layers above threshold → H-A WEAK.")
    ap.add_argument("--early-window", default="1,5,10,12,14,16,18,20,22,24,25",
                    help="Comma-separated 'early' layers to evaluate (excludes L=peak itself).")
    ap.add_argument("--data-out", default="docs/insights/_data")
    ap.add_argument("--fig-out", default="docs/figures")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def _load_pooled_D(model: str, tags: list[str]) -> tuple[torch.Tensor, dict]:
    base = PROJECT_ROOT / "outputs" / "e6_steering" / model
    Ds: list[torch.Tensor] = []
    counts: dict[str, int] = {}
    for tag in tags:
        p = base / f"calibration_{tag}" / "D_wrong.pt"
        if not p.exists():
            raise FileNotFoundError(f"D_wrong.pt missing for tag '{tag}' at {p}")
        D = torch.load(p, weights_only=True).float()
        counts[tag] = int(D.shape[0])
        Ds.append(D)
        print(f"[load] {tag}: D_wrong shape={tuple(D.shape)}")
    D = torch.cat(Ds, dim=0)
    counts["total"] = int(D.shape[0])
    return D, counts


def _layer_top_direction(M: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Return top-1 right singular vector + top singular value of M (n × d)."""
    _, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[0].clone(), float(S[0].item())


def _peak_subspace(M_peak: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V_K (K × d) + S[:K] for the peak-layer cumulative matrix M_peak (n × d)."""
    _, S, Vh = torch.linalg.svd(M_peak, full_matrices=False)
    return Vh[:K].clone(), S[:K].clone()


def main() -> None:
    args = _parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    early_layers = [int(x) for x in args.early_window.split(",") if x.strip()]
    print(f"[setup] model={args.model} tags={tags} peak={args.peak_layer} K={args.K} "
          f"threshold={args.threshold} strong-n≥{args.strong_n} weak-n≥{args.weak_n}")
    print(f"[setup] early_window layers={early_layers}")

    # ----- 1. Load pooled D_wrong -----
    D, counts = _load_pooled_D(args.model, tags)
    n, n_layers, d = D.shape
    print(f"[svd] D pooled shape={tuple(D.shape)}")
    if args.peak_layer >= n_layers:
        raise ValueError(f"peak layer {args.peak_layer} out of range (n_layers={n_layers}).")

    # ----- 2. V_K[L=peak] from cumulative D -----
    t0 = time.time()
    V_K, S_K = _peak_subspace(D[:, args.peak_layer, :], K=args.K)
    print(f"[peak] V_K shape={tuple(V_K.shape)}; S_K[:3]={[f'{s:.2f}' for s in S_K[:3]]} ... ; "
          f"||V_K|| should be 1.0 each: {[f'{V_K[k].norm().item():.4f}' for k in range(args.K)]}")

    # Random baseline: ||V_K^T u||_2 for u ~ uniform on sphere ≈ sqrt(K/d)
    random_baseline = (args.K / d) ** 0.5
    print(f"[baseline] random baseline sqrt(K/d) = sqrt({args.K}/{d}) = {random_baseline:.4f}")

    # ----- 3. Per-layer incremental & cumulative top directions -----
    print(f"\n[per-layer] computing top-1 SVD direction per layer (cumulative + incremental)...")
    rows: list[dict] = []
    for L in range(n_layers):
        # Cumulative: D[:, L, :] (anchor difference at layer L's output)
        u_cum, sigma_cum = _layer_top_direction(D[:, L, :])

        # Incremental: ΔD = D[:, L, :] - D[:, L-1, :] (anchor contribution from layer L's residual write)
        if L == 0:
            u_inc, sigma_inc = u_cum.clone(), sigma_cum  # at L=0, increment = cumulative
        else:
            delta = D[:, L, :] - D[:, L - 1, :]
            u_inc, sigma_inc = _layer_top_direction(delta)

        # Subspace alignment of top-1 directions (top-1 SVD-direction projection)
        proj_inc = (V_K @ u_inc).norm().item()  # ‖V_K^T u_L^inc‖_2 — NON-tautological at all L
        proj_cum = (V_K @ u_cum).norm().item()  # ‖V_K^T u_L^cum‖_2 — TAUTOLOGICAL at L=peak (= 1.0)

        # Per-v_k cosine sim with u_inc
        per_vk = [float((V_K[k] @ u_inc).abs().item()) for k in range(args.K)]

        # Explained variance fraction by V_K subspace (Frobenius-norm² ratio)
        # NOTE: NOT random-baseline-relative; at L=peak this is EV@K8(peak) (~0.21 for L=26),
        # NOT 1.0. At random L it equals K/d ≈ 0.0022 if D[:, L, :] is random-direction-isotropic.
        # The DIFFERENCE from random tells us if D[:, L, :]'s variance is selectively
        # concentrated within V_K[L=peak]'s subspace.
        D_L = D[:, L, :]
        proj_in_VK = D_L @ V_K.T  # (n, K)
        ev_frac_VK = float((proj_in_VK.pow(2).sum() / D_L.pow(2).sum()).item())
        # Random-isotropic baseline for K-dim subspace in d-dim space:
        ev_frac_baseline = args.K / d  # = 8/3584 ≈ 0.00223
        ev_frac_excess = ev_frac_VK / ev_frac_baseline  # how many × random isotropic

        row = {
            "layer": L,
            "sigma_cum": sigma_cum,
            "sigma_inc": sigma_inc,
            "sub_align_cum": proj_cum,
            "sub_align_inc": proj_inc,
            "ev_frac_VK": ev_frac_VK,
            "ev_frac_excess": ev_frac_excess,
            **{f"vk_cos_{k}": per_vk[k] for k in range(args.K)},
        }
        rows.append(row)
        if L in early_layers + [args.peak_layer, n_layers - 1] or L == 0:
            top_vk = sorted(enumerate(per_vk), key=lambda x: -x[1])[:3]
            print(f"  L={L:3d}: sigma_inc={sigma_inc:7.2f}  ‖V_K^T u_inc‖={proj_inc:.4f}  "
                  f"EV_frac_VK={ev_frac_VK:.4f} ({ev_frac_excess:.1f}× random)  top-3 vk: "
                  f"{[(k, f'{c:.3f}') for k, c in top_vk]}")
    print(f"[per-layer] elapsed: {time.time()-t0:.1f}s")

    # ----- 4. Verdict -----
    early_above = []
    for L in early_layers:
        proj = next(r for r in rows if r["layer"] == L)["sub_align_inc"]
        if proj > args.threshold:
            early_above.append((L, proj))

    if len(early_above) >= args.strong_n:
        verdict_label = "CONFIRMED"
    elif len(early_above) >= args.weak_n:
        verdict_label = "WEAK"
    else:
        verdict_label = "FALSIFIED"

    peak_align_inc = next(r for r in rows if r["layer"] == args.peak_layer)["sub_align_inc"]
    verdict = {
        "model": args.model,
        "n_total": counts["total"],
        "peak_layer": args.peak_layer,
        "K": args.K,
        "threshold": args.threshold,
        "random_baseline_sqrt_K_over_d": random_baseline,
        "early_window": early_layers,
        "early_layers_above_threshold": [{"layer": L, "sub_align_inc": p} for L, p in early_above],
        "n_early_layers_above_threshold": len(early_above),
        "strong_n": args.strong_n,
        "weak_n": args.weak_n,
        f"sub_align_inc_at_peak_L{args.peak_layer}": peak_align_inc,
        "verdict": verdict_label,
    }

    # ----- 5. Save canonical artifacts -----
    data_out = PROJECT_ROOT / args.data_out
    data_out.mkdir(parents=True, exist_ok=True)

    csv_path = data_out / "p0_2_HA_subspace_alignment.csv"
    fieldnames = (["layer", "sigma_cum", "sigma_inc", "sub_align_cum", "sub_align_inc",
                   "ev_frac_VK", "ev_frac_excess"]
                  + [f"vk_cos_{k}" for k in range(args.K)])
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[save] {csv_path}")

    verdict_path = data_out / "p0_2_HA_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))
    print(f"[save] {verdict_path}")

    print()
    print("==== H-A VERDICT ====")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print()

    if args.no_figures:
        return

    # ----- 6. Figures -----
    import matplotlib.pyplot as plt

    fig_out = PROJECT_ROOT / args.fig_out
    fig_out.mkdir(parents=True, exist_ok=True)

    # Figure 3: two stacked panels — top-1 alignment (top) + EV-fraction by V_K (bottom)
    layers = [r["layer"] for r in rows]
    align_inc = [r["sub_align_inc"] for r in rows]
    align_cum = [r["sub_align_cum"] for r in rows]
    ev_frac = [r["ev_frac_VK"] for r in rows]
    ev_frac_baseline = args.K / d
    fig, axs = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)

    # Top: ‖V_K^T u_L‖_2 — top-1 SVD direction projection (non-tautological inc; tautological cum at L=peak)
    ax = axs[0]
    ax.plot(layers, align_inc, marker="o", lw=1.4,
            label="‖V_K^T · u_L^inc‖₂  (NON-tautological — layer L's incremental top direction)")
    ax.plot(layers, align_cum, marker="s", lw=1.0, alpha=0.6,
            label=f"‖V_K^T · u_L^cum‖₂  (TAUTOLOGICAL at L={args.peak_layer} = 1.0 by construction)")
    ax.axhline(args.threshold, ls="--", color="crimson", lw=0.8,
               label=f"threshold = {args.threshold} (~{args.threshold/random_baseline:.0f}× random sqrt-baseline)")
    ax.axhline(random_baseline, ls=":", color="gray", lw=0.8,
               label=f"random subspace sqrt-baseline = {random_baseline:.3f}")
    ax.axvline(args.peak_layer, ls="--", color="black", lw=0.6, alpha=0.5)
    ax.set_ylabel("Subspace alignment with V_K[L=%d] (top-1 projection)" % args.peak_layer)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"H-A: alignment of per-layer anchor directions with V_K[L={args.peak_layer}]\n"
        f"Verdict {verdict_label}: {len(early_above)} of {len(early_layers)} early layers > {args.threshold} "
        f"(≥{args.strong_n} for CONFIRMED, ≥{args.weak_n} for WEAK)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # Bottom: EV-fraction by V_K subspace (non-tautological; at L=peak it's EV@K8 ≈ 0.21, NOT 1.0)
    ax = axs[1]
    ax.plot(layers, ev_frac, marker="o", lw=1.4, color="tab:purple",
            label="EV_frac_V_K(L) = ‖V_K V_K^T D[:, L, :]‖_F² / ‖D[:, L, :]‖_F²")
    ax.axhline(ev_frac_baseline, ls=":", color="gray", lw=0.8,
               label=f"random isotropic baseline K/d = {ev_frac_baseline:.4f}")
    ax.axvline(args.peak_layer, ls="--", color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("Layer L")
    ax.set_ylabel("Frac. of D[:, L, :] variance in V_K subspace")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"NON-tautological measure: at L={args.peak_layer} this equals EV@K{args.K} ≈ 0.21, NOT 1.0. "
        f"Sharp L=20 → L=22 transition: 9.5× → 30× random.",
        fontsize=9,
    )

    fig.tight_layout()
    p3 = fig_out / "P0-2_HA_subspace_alignment.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[save] {p3}")

    # Figure 4: per-v_k per-layer cosine sim heatmap (incremental)
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    heatmap = torch.zeros(args.K, n_layers)
    for r in rows:
        for k in range(args.K):
            heatmap[k, r["layer"]] = r[f"vk_cos_{k}"]
    im = ax.imshow(heatmap.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Layer L (incremental u_L^inc)")
    ax.set_ylabel(f"V_K[L={args.peak_layer}] direction index k")
    ax.set_yticks(range(args.K))
    ax.set_yticklabels([f"v_{k}" for k in range(args.K)])
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_title(
        f"|v_k^T · u_L^inc|  (per-v_k alignment with each layer's incremental anchor direction)\n"
        f"Diagonal/sparse pattern → multi-layer accumulation; single-column dominance → L={args.peak_layer}-only",
        fontsize=10,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("|cos(v_k, u_L^inc)|")
    fig.tight_layout()
    p4 = fig_out / "P0-2_HA_per_vk_heatmap.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"[save] {p4}")


if __name__ == "__main__":
    main()
