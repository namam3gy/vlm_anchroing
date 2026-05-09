"""analyze_e6_eigenvalue_spectrum.py — P0-2 per-layer SVD analysis.

Reads pooled (a − m) calibration difference matrix D_wrong from PlotQA + InfoVQA
calibration directories, runs full thin SVD per layer, computes scale-invariant
rank measures, and emits canonical CSV + figures.

Pre-registered measures:
  - Shannon effective rank (Roy & Vetterli 2007): exp(-Σ p_i log p_i),
    p_i = sv_i / Σ sv_j  — primary, scale-invariant
  - Participation ratio: (Σ sv_i²)² / Σ sv_i⁴  — scale-invariant
  - Stable rank: Σ sv_i² / sv_0²  — scale-invariant
  - Explained variance at K=8: Σ_{i<8} sv_i² / Σ sv_i²  — scale-invariant
  - sv_7/sv_8 ratio  — local elbow probe (pre-registered threshold ≥ 1.5)

Pre-registered acceptance criteria:
  (a) Clean elbow at rank-8 on L=26: sv_7/sv_8 ≥ 1.5 OR explained-variance
      at K=8 ≥ 0.70.
  (b) Effective rank decreases monotonically across L=10 → L=27 (or sharply
      drops at the L=20-26 transition).

Output:
  docs/insights/_data/p0_2_per_layer_spectrum.csv
    columns: layer, sv_0..sv_15, eff_rank_shannon, part_ratio, stable_rank,
             explained_var_K8, sv7_over_sv8
  docs/insights/_data/p0_2_per_layer_spectrum.md  — same data, table format
  docs/figures/P0-2_L26_spectrum.png  — log-spectrum at L=26 (Figure 1)
  docs/figures/P0-2_per_layer_rank_trajectory.png  — per-layer trajectory (Figure 2)

Usage:
  uv run python scripts/analyze_e6_eigenvalue_spectrum.py \\
      --model llava-onevision-qwen2-7b-ov
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
    ap.add_argument(
        "--model",
        default="llava-onevision-qwen2-7b-ov",
        help="HF-style model dir name under outputs/e6_steering/",
    )
    ap.add_argument(
        "--tags",
        default="plotqa,infographicvqa",
        help="Comma-separated calibration tags (each must have a "
             "calibration_<tag>/D_wrong.pt under the model dir).",
    )
    ap.add_argument(
        "--K-probe",
        type=int,
        default=8,
        help="Subspace rank to probe for the elbow test (default 8 = chosen E6 cell).",
    )
    ap.add_argument(
        "--top-svs",
        type=int,
        default=50,
        help="Number of top singular values to retain in the L=26 spectrum figure.",
    )
    ap.add_argument(
        "--peak-layer",
        type=int,
        default=26,
        help="Reference layer for the L=peak spectrum figure.",
    )
    ap.add_argument(
        "--data-out",
        default="docs/insights/_data",
        help="Output dir for canonical CSV + MD.",
    )
    ap.add_argument(
        "--fig-out",
        default="docs/figures",
        help="Output dir for figures.",
    )
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (CSV only).",
    )
    return ap.parse_args()


def _load_pooled_D(model: str, tags: list[str]) -> tuple[torch.Tensor, dict]:
    """Load D_wrong.pt from each calibration_<tag>/ and concatenate along sample axis."""
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


def _scale_invariant_rank_measures(S: torch.Tensor, K_probe: int) -> dict[str, float]:
    """Compute scale-invariant spectral rank measures from sorted-desc singular values S."""
    sv2 = S * S
    total_sv = S.sum()
    p = S / total_sv
    eff_rank_shannon = float(torch.exp(-(p * (p + 1e-30).log()).sum()).item())
    part_ratio = float((sv2.sum() ** 2 / (sv2 * sv2).sum()).item())
    stable_rank = float((sv2.sum() / sv2[0]).item())
    explained_var_K = float((sv2[:K_probe].sum() / sv2.sum()).item())
    sv_ratio_K = float((S[K_probe - 1] / S[K_probe]).item()) if K_probe < S.shape[0] else float("nan")
    return {
        "eff_rank_shannon": eff_rank_shannon,
        "part_ratio": part_ratio,
        "stable_rank": stable_rank,
        f"explained_var_K{K_probe}": explained_var_K,
        f"sv{K_probe-1}_over_sv{K_probe}": sv_ratio_K,
    }


def _per_layer_svd(D: torch.Tensor, K_top: int, K_probe: int) -> tuple[list[dict], dict[int, torch.Tensor]]:
    """Per-layer thin SVD; return per-layer rows + full singular values for each layer."""
    n, n_layers, d = D.shape
    rows: list[dict] = []
    full_svs: dict[int, torch.Tensor] = {}
    for L in range(n_layers):
        DL = D[:, L, :]
        S = torch.linalg.svdvals(DL)
        full_svs[L] = S.clone()
        row: dict = {"layer": L}
        for k in range(min(K_top, S.shape[0])):
            row[f"sv_{k}"] = float(S[k].item())
        row.update(_scale_invariant_rank_measures(S, K_probe))
        rows.append(row)
        if L % 4 == 0 or L == n_layers - 1:
            print(f"  L={L:3d}  sv_0={S[0]:.3f}  sv_{K_probe-1}={S[K_probe-1]:.3f}  "
                  f"sv_{K_probe}={S[K_probe]:.3f}  eff_rank={row['eff_rank_shannon']:.1f}  "
                  f"part_R={row['part_ratio']:.2f}")
    return rows, full_svs


def _save_canonical_csv(rows: list[dict], counts: dict, out_csv: Path, K_probe: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    K_top = max(int(k.split("_")[1]) for k in rows[0] if k.startswith("sv_") and k != "sv_ratio") + 1
    fieldnames = (
        ["layer"]
        + [f"sv_{i}" for i in range(K_top)]
        + ["eff_rank_shannon", "part_ratio", "stable_rank",
           f"explained_var_K{K_probe}", f"sv{K_probe-1}_over_sv{K_probe}"]
    )
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[save] {out_csv}  ({len(rows)} layers; n_total={counts['total']}; tags={list(counts.keys())[:-1]})")


def _save_md_table(rows: list[dict], counts: dict, out_md: Path, K_probe: int, peak_layer: int) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    tag_str = ", ".join(f"{k}={v}" for k, v in counts.items() if k != "total")
    lines.append(f"# P0-2 per-layer eigenvalue spectrum — pooled (a − m) calibration\n")
    lines.append(f"**Source.** D_wrong (PlotQA + InfoVQA pooled). N_total = {counts['total']} ({tag_str}).\n")
    lines.append(
        f"**Pre-registered acceptance criteria.**\n"
        f"- (a) Rank-K elbow at L={peak_layer}: `sv_{K_probe-1}/sv_{K_probe} ≥ 1.5` "
        f"OR `explained_var_K{K_probe} ≥ 0.70`.\n"
        f"- (b) Effective rank decreases monotonically across L=10 → final layer.\n"
    )
    lines.append("## Per-layer rank measures\n")
    lines.append(f"| L | sv_0 | sv_{K_probe-1} | sv_{K_probe} | sv{K_probe-1}/sv{K_probe} | EV@K{K_probe} | eff_rank | part_R | stable_R |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        L = r["layer"]
        lines.append(
            f"| {L} | {r['sv_0']:.2f} | {r[f'sv_{K_probe-1}']:.2f} | {r[f'sv_{K_probe}']:.2f} "
            f"| {r[f'sv{K_probe-1}_over_sv{K_probe}']:.3f} | {r[f'explained_var_K{K_probe}']:.4f} "
            f"| {r['eff_rank_shannon']:.1f} | {r['part_ratio']:.2f} | {r['stable_rank']:.2f} |\n"
        )
    out_md.write_text("".join(lines))
    print(f"[save] {out_md}")


def _save_full_svs_pt(full_svs: dict[int, torch.Tensor], top_n: int, out_pt: Path) -> None:
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    keep = {L: S[:top_n].clone() for L, S in full_svs.items()}
    torch.save(keep, out_pt)
    print(f"[save] {out_pt}  (top-{top_n} per layer for figure reuse)")


def _make_figures(
    rows: list[dict],
    full_svs: dict[int, torch.Tensor],
    counts: dict,
    K_probe: int,
    top_svs: int,
    peak_layer: int,
    fig_out: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig_out.mkdir(parents=True, exist_ok=True)

    # Figure 1: L=peak spectrum, log scale, K=K_probe annotated
    S_peak = full_svs[peak_layer]
    n_show = min(top_svs, S_peak.shape[0])
    xs = list(range(1, n_show + 1))
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.semilogy(xs, S_peak[:n_show].numpy(), marker="o", markersize=3.5, lw=1.0)
    ax.axvline(K_probe, ls="--", color="crimson", lw=1.0,
               label=f"K = {K_probe} (chosen E6 cell — em-deal-breaker selected, NOT spectrum elbow)")
    ax.set_xlabel("Singular value index k")
    ax.set_ylabel("Singular value σ_k(D[:, L=%d, :])" % peak_layer)
    sv_ratio = float(S_peak[K_probe-1] / S_peak[K_probe])
    ax.set_title(
        f"Spectrum of (a − m) calibration matrix at L={peak_layer}\n"
        f"PlotQA + InfoVQA pooled, n={counts['total']} wrong-base; "
        f"sv_{K_probe-1}/sv_{K_probe} = {sv_ratio:.3f} (no rank-{K_probe} elbow)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = fig_out / f"P0-2_L{peak_layer}_spectrum.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[save] {p1}")

    # Figure 2: per-layer trajectory of three rank measures + EV@K_probe
    layers = [r["layer"] for r in rows]
    eff = [r["eff_rank_shannon"] for r in rows]
    part = [r["part_ratio"] for r in rows]
    stab = [r["stable_rank"] for r in rows]
    ev = [r[f"explained_var_K{K_probe}"] for r in rows]

    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    for ax, (vals, ylab, title) in zip(
        axs.flat,
        [
            (eff, "Shannon effective rank exp(H(p))", "Shannon effective rank ↑ = more dispersed"),
            (part, "Participation ratio (Σσ²)²/Σσ⁴", "Participation ratio ↑ = more dispersed"),
            (stab, "Stable rank Σσ²/σ_0²", "Stable rank ↑ = more dispersed"),
            (ev, f"Explained var at K={K_probe} (top-{K_probe} σ²-fraction)", f"EV@K{K_probe} ↑ = more concentrated"),
        ],
    ):
        ax.plot(layers, vals, marker="o", markersize=4, lw=1.2)
        ax.axvline(peak_layer, ls="--", color="crimson", lw=0.8, alpha=0.6)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    for ax in axs[-1, :]:
        ax.set_xlabel("Layer L")
    fig.suptitle(
        f"Per-layer rank trajectory of (a − m) anchor calibration\n"
        f"PlotQA + InfoVQA pooled, n={counts['total']} wrong-base, model=llava-onevision-qwen2-7b-ov",
    )
    fig.tight_layout()
    p2 = fig_out / "P0-2_per_layer_rank_trajectory.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[save] {p2}")


def _verify_acceptance(rows: list[dict], counts: dict, K_probe: int, peak_layer: int) -> dict:
    """Apply pre-registered acceptance criteria and return verdict dict."""
    peak_row = next(r for r in rows if r["layer"] == peak_layer)
    sv_ratio = peak_row[f"sv{K_probe-1}_over_sv{K_probe}"]
    ev = peak_row[f"explained_var_K{K_probe}"]
    pass_a_ratio = sv_ratio >= 1.5
    pass_a_ev = ev >= 0.70
    pass_a = pass_a_ratio or pass_a_ev

    # (b) monotonic decrease over L=10 → final
    layers_after_10 = [r for r in rows if r["layer"] >= 10]
    eff_after_10 = [r["eff_rank_shannon"] for r in layers_after_10]
    final_layer = layers_after_10[-1]["layer"]
    pass_b = all(eff_after_10[i + 1] <= eff_after_10[i] for i in range(len(eff_after_10) - 1))

    direction_b = "INCREASES" if eff_after_10[-1] > eff_after_10[0] else "DECREASES"
    eff_at_peak_layer = next(r["eff_rank_shannon"] for r in rows if r["layer"] == peak_layer)
    eff_at_L10 = eff_after_10[0]
    return {
        "n_total": counts["total"],
        "model": "llava-onevision-qwen2-7b-ov",
        f"sv{K_probe-1}_over_sv{K_probe}_at_L{peak_layer}": sv_ratio,
        f"explained_var_K{K_probe}_at_L{peak_layer}": ev,
        "criterion_a_threshold_ratio": 1.5,
        "criterion_a_threshold_ev": 0.70,
        "criterion_a_pass_ratio": pass_a_ratio,
        "criterion_a_pass_ev": pass_a_ev,
        "criterion_a_pass": pass_a,
        "criterion_b_pass": pass_b,
        "eff_rank_at_L10": eff_at_L10,
        f"eff_rank_at_L{peak_layer}": eff_at_peak_layer,
        f"eff_rank_at_L{final_layer}": eff_after_10[-1],
        "criterion_b_direction": direction_b,
    }


def main() -> None:
    args = _parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(f"[setup] model={args.model} tags={tags} K_probe={args.K_probe} peak={args.peak_layer}")

    D, counts = _load_pooled_D(args.model, tags)
    n, n_layers, d = D.shape
    print(f"[svd] D pooled shape={tuple(D.shape)}; computing full thin SVD per layer")

    t0 = time.time()
    rows, full_svs = _per_layer_svd(D, K_top=16, K_probe=args.K_probe)
    print(f"[svd] elapsed: {time.time()-t0:.1f}s")

    out_csv = PROJECT_ROOT / args.data_out / "p0_2_per_layer_spectrum.csv"
    out_md = PROJECT_ROOT / args.data_out / "p0_2_per_layer_spectrum.md"
    out_pt = PROJECT_ROOT / args.data_out / "p0_2_full_svs_top.pt"
    _save_canonical_csv(rows, counts, out_csv, args.K_probe)
    _save_md_table(rows, counts, out_md, args.K_probe, args.peak_layer)
    _save_full_svs_pt(full_svs, top_n=args.top_svs, out_pt=out_pt)

    verdict = _verify_acceptance(rows, counts, args.K_probe, args.peak_layer)
    out_verdict = PROJECT_ROOT / args.data_out / "p0_2_acceptance_verdict.json"
    out_verdict.write_text(json.dumps(verdict, indent=2))
    print(f"[save] {out_verdict}")
    print()
    print("==== ACCEPTANCE VERDICT ====")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print()

    if not args.no_figures:
        _make_figures(rows, full_svs, counts, args.K_probe, args.top_svs,
                      args.peak_layer, PROJECT_ROOT / args.fig_out)


if __name__ == "__main__":
    main()
