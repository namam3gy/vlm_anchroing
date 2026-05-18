"""Render the §5.2 paper figures (paper_5_2a, paper_5_2b) from the
aggregated CSVs produced by `aggregate_e6_pilot_grid.py` and
`aggregate_e6_layer_sweep_p4.py`. Mirrors the notebook's figure cells so
the notebook + this script stay in sync; useful when running headless.

Outputs (both PDF + PNG):
  outputs/paper/section_5_figures/paper_5_2a_E6_pilot_grid_plotqa.{pdf,png}
  outputs/paper/section_5_figures/paper_5_2b_layer_sweep_delta_df.{pdf,png}
  docs/figures/paper_5_2a_E6_pilot_grid_plotqa.png
  docs/figures/paper_5_2b_layer_sweep_delta_df.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "outputs" / "paper" / "section_5_e6_steering" / "_data"
PDF_OUT = REPO / "outputs" / "paper" / "section_5_figures"
PNG_OUT = REPO / "docs" / "figures"
PDF_OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

PILOT_LAYERS = [14, 20, 22, 26]
PILOT_ALPHAS = [0.5, 1.0, 2.0]
PILOT_KS = [1, 2, 4, 8]


def save_figure(fig, stem: str):
    pdf = PDF_OUT / f"{stem}.pdf"
    png = PNG_OUT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def fig_pilot_grid() -> plt.Figure | None:
    src = DATA / "E6_pilot_grid.csv"
    if not src.exists():
        print(f"  (skipped — {src} missing)")
        return None
    grid = pd.read_csv(src)
    grid = grid[grid["calib"] == "plotqa"]
    chosen_L, chosen_K, chosen_alpha = 26, 8, 1.0

    metrics = [
        ("delta_adopt", "Δ adopt(a) pp"),
        ("delta_df",    "Δ df(a) pp"),
        ("delta_em_a",  "Δ em(a) pp"),
        ("delta_em_b",  "Δ em(b) pp"),
    ]
    fig, axes = plt.subplots(len(metrics), len(PILOT_KS),
                             figsize=(13, 9.0), dpi=150,
                             sharex=True, sharey=True)
    for col, K in enumerate(PILOT_KS):
        for row, (col_metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            sub = grid[grid["K"] == K]
            piv = sub.pivot_table(index="layer", columns="alpha", values=col_metric)
            piv = piv.reindex(index=PILOT_LAYERS, columns=PILOT_ALPHAS)
            piv = piv * 100.0  # to pp
            vmax = float(piv.abs().values.max()) if piv.size else 1.0
            ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i, L in enumerate(PILOT_LAYERS):
                for j, alpha in enumerate(PILOT_ALPHAS):
                    v = piv.loc[L, alpha]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                                fontsize=8, color="black")
                    if (L, K, alpha) == (chosen_L, chosen_K, chosen_alpha):
                        ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45),
                                                   0.9, 0.9, fill=False,
                                                   edgecolor="black", linewidth=2.0))
            if row == 0:
                ax.set_title(f"K={K}", fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(range(len(PILOT_ALPHAS))); ax.set_xticklabels(PILOT_ALPHAS, fontsize=8)
            ax.set_yticks(range(len(PILOT_LAYERS))); ax.set_yticklabels(PILOT_LAYERS, fontsize=8)
            if row == len(metrics) - 1:
                ax.set_xlabel("α", fontsize=9)
    fig.suptitle("§5.2a — E6 pilot grid (PlotQA × OneVision, n=250)\n"
                 "Δ vs baseline; ▢ chosen cell = L=26, K=8, α=1.0 (§6.2.2 selection)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def fig_layer_sweep() -> plt.Figure | None:
    src = DATA / "p4_layer_sweep_per_cell_ci.csv"
    if not src.exists():
        print(f"  (skipped — {src} missing)")
        return None
    sweep = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    color = {"plotqa": "#1F4FA8", "infographicvqa": "#C8102E",
             "tallyqa": "#1A7F3F", "chartqa": "#F2A900", "mathvista": "#6C7280"}
    label = {"plotqa": "PlotQA", "infographicvqa": "InfoVQA",
             "tallyqa": "TallyQA", "chartqa": "ChartQA", "mathvista": "MathVista"}
    for ds_tag, c in color.items():
        head = sweep[(sweep["ds_tag"] == ds_tag) & (sweep["K"] == 8)]
        if len(head):
            head = head.sort_values("layer")
            ax.plot(head["layer"], head["delta_df"] * 100,
                    color=c, marker="o", label=f"{label[ds_tag]} K=8")
        k1 = sweep[(sweep["ds_tag"] == ds_tag) & (sweep["K"] == 1)]
        if len(k1):
            k1 = k1.sort_values("layer")
            ax.plot(k1["layer"], k1["delta_df"] * 100,
                    color=c, marker="s", linestyle="--", alpha=0.6,
                    label=f"{label[ds_tag]} K=1")
    ax.axhline(0, color="#888", linewidth=0.7, linestyle=":")
    ax.set_xlabel("layer (L)")
    ax.set_ylabel("Δ df(a)  pp  (negative ⇒ anchoring reduced)")
    ax.set_title("§5.2b — 5-dataset Δdf(a) at α=1.0 — K=8 (solid) vs K=1 (dashed)\n"
                 "K=1 sign-reversal at mid-stack supports §5.2 K-subspace argument")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = fig_pilot_grid()
    if fig is not None:
        save_figure(fig, "paper_5_2a_E6_pilot_grid_plotqa")
    fig = fig_layer_sweep()
    if fig is not None:
        save_figure(fig, "paper_5_2b_layer_sweep_delta_df")
