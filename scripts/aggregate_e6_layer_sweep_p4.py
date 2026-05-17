"""aggregate_e6_layer_sweep_p4.py — per-cell paired-bootstrap CI for the P4 follow-up.

Inputs (per dataset, two sweep dirs each):
  outputs/e6_steering/<MODEL>/sweep_subspace_<ds>_<SCOPE>_p4_layer_sweep_K1_layers_K8/predictions.jsonl
  outputs/e6_steering/<MODEL>/sweep_subspace_<ds>_<SCOPE>_p4_layer_sweep_K1_L26_K1/predictions.jsonl

For every non-baseline cell in each sweep dir, runs the same paired-bootstrap
methodology as `build_e6_stage4_bootstrap_ci.py` (sid-paired resampling,
per-arm denominator/numerator recomputed each draw, B=10,000) for the four
metrics (adopt, df, em_a, em_b).

Outputs:
  docs/insights/_data/p4_layer_sweep_per_cell_ci.csv     # per (dataset × cell) row
  docs/insights/_data/p4_layer_sweep_summary.md          # human-readable table
  docs/insights/_data/p4_layer_sweep_bootstrap_draws.npz # raw draws (for reproducibility)
  docs/figures/p4_layer_sweep_delta_df.png               # Δdf curve: layer × K=8, K=1 marker

Usage:
  uv run python scripts/aggregate_e6_layer_sweep_p4.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_e6_stage4_bootstrap_ci import (  # noqa: E402
    _ci,
    _load_by_cell_sid,
    _paired_bootstrap_deltas,
    _paired_sids,
    _per_sid_indicators,
    _point_metrics,
)

MODEL = "llava-onevision-qwen2-7b-ov"
SCOPE = "plotqa_infovqa_pooled_n5k"
SWEEP_TAG = "p4_layer_sweep_K1"

# (label, ds_tag, sweep_subtag)
SWEEP_DIRS = [
    ("MathVista", "mathvista", "layers_K8"),
    ("MathVista", "mathvista", "L26_K1"),
    ("ChartQA", "chartqa", "layers_K8"),
    ("ChartQA", "chartqa", "L26_K1"),
    ("InfoVQA", "infographicvqa", "layers_K8"),
    ("InfoVQA", "infographicvqa", "L26_K1"),
    ("PlotQA", "plotqa", "layers_K8"),
    ("PlotQA", "plotqa", "L26_K1"),
    ("TallyQA", "tallyqa", "layers_K8"),
    ("TallyQA", "tallyqa", "L26_K1"),
]
METRICS = ("adopt", "df", "em_a", "em_b")


def _per_cell_rows(label: str, ds_tag: str, sweep_subtag: str,
                    bootstrap: int, seed: int,
                    e6_root: Path | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    base = e6_root if e6_root is not None else (PROJECT_ROOT / "outputs" / "e6_steering")
    sweep_dir = (
        base / MODEL
        / f"sweep_subspace_{ds_tag}_{SCOPE}_{SWEEP_TAG}_{sweep_subtag}"
    )
    pred = sweep_dir / "predictions.jsonl"
    if not pred.exists():
        print(f"[skip] {label}/{sweep_subtag}: missing {pred}")
        return [], {}
    by_cell = _load_by_cell_sid(pred)
    if "baseline" not in by_cell:
        print(f"[skip] {label}/{sweep_subtag}: no baseline records")
        return [], {}
    base = by_cell["baseline"]

    rows: list[dict] = []
    draws: dict[str, np.ndarray] = {}
    cells = sorted(c for c in by_cell if c != "baseline")
    for cell_label in cells:
        mit = by_cell[cell_label]
        sids = _paired_sids(base, mit)
        n = len(sids)
        if n == 0:
            print(f"[skip] {label}/{sweep_subtag}/{cell_label}: 0 paired sids")
            continue
        base_ind = _per_sid_indicators(base, sids)
        mit_ind = _per_sid_indicators(mit, sids)
        bm = _point_metrics(base_ind)
        mm = _point_metrics(mit_ind)
        d = _paired_bootstrap_deltas(base_ind, mit_ind, B=bootstrap, seed=seed)
        for m in METRICS:
            draws[f"{ds_tag}__{cell_label}__{m}"] = d[m].astype(np.float32)

        # Parse cell_label "L{LL}_K{KK}_a{alpha}"
        try:
            parts = cell_label.replace("L", "").replace("_K", " ").replace("_a", " ").split()
            L = int(parts[0])
            K = int(parts[1])
            alpha = float(parts[2])
        except Exception:  # pragma: no cover
            L, K, alpha = -1, -1, -1.0

        row = {
            "dataset": label,
            "ds_tag": ds_tag,
            "sweep_subtag": sweep_subtag,
            "cell_label": cell_label,
            "layer": L,
            "K": K,
            "alpha": alpha,
            "n_paired": n,
        }
        for m in METRICS:
            d_pt = mm[m] - bm[m]
            ci95 = _ci(d[m], alpha=0.05)
            row.update({
                f"{m}_baseline": bm[m],
                f"{m}_mitigation": mm[m],
                f"delta_{m}": d_pt,
                f"delta_{m}_ci95_lo": ci95[0],
                f"delta_{m}_ci95_hi": ci95[1],
            })
        rows.append(row)
        print(
            f"  {label:<11} {cell_label:<14} n={n:>5} "
            f"Δdf={row['delta_df']:+.4f} "
            f"95%[{row['delta_df_ci95_lo']:+.4f},{row['delta_df_ci95_hi']:+.4f}] "
            f"Δem(b)={row['delta_em_b']:+.4f} "
            f"95%[{row['delta_em_b_ci95_lo']:+.4f},{row['delta_em_b_ci95_hi']:+.4f}]"
        )
    return rows, draws


def _make_layer_sweep_figure(rows: list[dict], out_path: Path) -> None:
    """5-panel figure: one subplot per dataset, x=layer for K=8 + K=1 marker at L=26."""
    import matplotlib.pyplot as plt

    datasets = ["MathVista", "ChartQA", "InfoVQA", "PlotQA", "TallyQA"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), sharey=True)
    for ax, ds in zip(axes, datasets):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            ax.set_title(f"{ds}\n(no data)")
            continue
        # K=8 layer sweep
        k8 = sorted([r for r in ds_rows if r["K"] == 8], key=lambda r: r["layer"])
        if k8:
            xs = [r["layer"] for r in k8]
            ys = [r["delta_df"] * 100 for r in k8]
            lo = [r["delta_df_ci95_lo"] * 100 for r in k8]
            hi = [r["delta_df_ci95_hi"] * 100 for r in k8]
            ax.plot(xs, ys, "o-", color="#1f77b4", label="K=8")
            ax.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.2)
        # K=1 marker at L=26
        k1 = [r for r in ds_rows if r["K"] == 1 and r["layer"] == 26]
        if k1:
            r = k1[0]
            ax.errorbar([26], [r["delta_df"] * 100],
                         yerr=[[r["delta_df"] * 100 - r["delta_df_ci95_lo"] * 100],
                               [r["delta_df_ci95_hi"] * 100 - r["delta_df"] * 100]],
                         fmt="s", color="#d62728", label="K=1 @ L=26",
                         markersize=8)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(f"{ds}\nn_paired = {ds_rows[0]['n_paired']}")
        ax.set_xlabel("Layer L")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Δ direction-follow (pp)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "P4 §5.4 P3 verification: layer-sweep K=8 (blue) + K=1 fall-back at L=26 (red) "
        "on llava-onevision-qwen2-7b-ov, calibrated on PlotQA+InfoVQA pooled n5k.",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[write] {out_path}")


def _write_summary_md(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P4 §5.4 framework verification — layer sweep + K=1 falsification",
        "",
        "Auto-generated by `scripts/aggregate_e6_layer_sweep_p4.py`.",
        "",
        f"**Model**: `{MODEL}`. **Calibration scope**: PlotQA + InfoVQA pooled (N=5,000 wrong-base).",
        "",
        "**Cells**:",
        "- Layer sweep K=8 α=1.0: L ∈ {5, 10, 15, 20, 25, 27}",
        "- K=1 α=1.0 at L=26 (P2 single-direction falsification)",
        "",
        "Paired-bootstrap CI (B=10,000), sid-paired resampling, per-arm (num, den) re-computed each draw.",
        "Bold = 95 % CI excludes 0.",
        "",
        "## Per-cell Δdf(a) per dataset",
        "",
        "| Dataset | Cell (L_K_α) | n_paired | Δdf(a) pp | 95 % CI |",
        "|---|---|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["dataset"], x["K"], x["layer"])):
        d = r["delta_df"] * 100
        lo = r["delta_df_ci95_lo"] * 100
        hi = r["delta_df_ci95_hi"] * 100
        signf = "**" if (lo > 0 or hi < 0) else ""
        lines.append(
            f"| {r['dataset']} | {r['cell_label']} | {r['n_paired']} | "
            f"{signf}{d:+.2f}{signf} | [{lo:+.2f}, {hi:+.2f}] |"
        )
    lines.append("")
    lines.append("## Per-cell Δem(b) per dataset (multiplicity-robust signal)")
    lines.append("")
    lines.append("| Dataset | Cell (L_K_α) | n_paired | Δem(b) pp | 95 % CI |")
    lines.append("|---|---|---:|---:|---|")
    for r in sorted(rows, key=lambda x: (x["dataset"], x["K"], x["layer"])):
        d = r["delta_em_b"] * 100
        lo = r["delta_em_b_ci95_lo"] * 100
        hi = r["delta_em_b_ci95_hi"] * 100
        signf = "**" if (lo > 0 or hi < 0) else ""
        lines.append(
            f"| {r['dataset']} | {r['cell_label']} | {r['n_paired']} | "
            f"{signf}{d:+.2f}{signf} | [{lo:+.2f}, {hi:+.2f}] |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--out-data", default=str(PROJECT_ROOT / "docs" / "insights" / "_data"))
    ap.add_argument("--out-fig", default=str(PROJECT_ROOT / "docs" / "figures"))
    ap.add_argument(
        "--e6-root", type=str, default=None,
        help="Override `outputs/e6_steering/` (where the per-dataset "
             "sweep_subspace_<ds>_<scope>_<subtag>/predictions.jsonl "
             "files live). Reproducer notebooks point this at an "
             "isolated tree.",
    )
    args = ap.parse_args()

    out_data = Path(args.out_data)
    out_fig = Path(args.out_fig)
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    e6_root = Path(args.e6_root).resolve() if args.e6_root else None

    all_rows: list[dict] = []
    all_draws: dict[str, np.ndarray] = {}

    for label, ds_tag, sweep_subtag in SWEEP_DIRS:
        rows, draws = _per_cell_rows(label, ds_tag, sweep_subtag,
                                       bootstrap=args.bootstrap, seed=args.seed,
                                       e6_root=e6_root)
        all_rows.extend(rows)
        all_draws.update(draws)

    if not all_rows:
        print("[err] no data found in any P4 sweep dir — runs not finished?")
        return

    # Per-cell × per-dataset CSV
    csv_path = out_data / "p4_layer_sweep_per_cell_ci.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"[write] {csv_path}")

    # Raw draws
    np.savez(out_data / "p4_layer_sweep_bootstrap_draws.npz", **all_draws)
    print(f"[write] {out_data / 'p4_layer_sweep_bootstrap_draws.npz'}")

    # Markdown
    _write_summary_md(all_rows, out_data / "p4_layer_sweep_summary.md")

    # Figure
    _make_layer_sweep_figure(all_rows, out_fig / "p4_layer_sweep_delta_df.png")


if __name__ == "__main__":
    main()
