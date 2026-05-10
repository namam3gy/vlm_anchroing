"""Generate the 4 new figures for the paper-style PPTX summary deck.

Outputs (under docs/figures/):
  - paper_M2_variant_comparison.png  — wrong>correct rate-gap per adopt variant
  - paper_L1_confidence_quartile.png — Q1→Q4 adopt/df trend on the canonical entropy proxy
  - paper_E5e_mathvista_bars.png     — wrong-base S1 adopt(a) vs adopt(m) per model
  - paper_cross_dataset_summary.png  — wrong-base S1 adopt across 4 datasets × models

All four are static PNGs at 1600x1000 (~16:10) for easy slide embedding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs" / "insights" / "_data"
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Midnight Executive palette
NAVY = "#1E2761"
ICE = "#CADCFC"
ACCENT_GOLD = "#F2A900"
ACCENT_RED = "#C8102E"
GREY = "#6C7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_M2_variant_comparison() -> Path:
    """Bar chart: per-adopt-variant mean(wrong − correct) gap on S0/S1 cells."""
    df = pd.read_csv(DATA_DIR / "M2_metric_variants_long.csv")
    cells = df[
        (df["metric"] == "adopt")
        & (df["cond_class"] == "a")
        & (df["stratum"].isin(["S0", "S1"]))
        & (df["base_correct"].isin(["wrong", "correct"]))
    ]
    piv = cells.pivot_table(
        index=["experiment", "dataset", "model", "stratum", "variant_id"],
        columns="base_correct",
        values="rate",
        aggfunc="first",
    ).dropna(subset=["wrong", "correct"]).reset_index()
    piv["gap"] = piv["wrong"] - piv["correct"]

    summary = piv.groupby("variant_id").agg(
        n=("gap", "size"),
        mean_gap=("gap", "mean"),
        wins=("gap", lambda v: int((v > 0).sum())),
    ).reset_index()
    summary["winrate"] = summary["wins"] / summary["n"]

    label_order = [
        "A_raw__D_all", "A_clean__D_paired", "A_clean__D_all",
        "A_paired__D_clean", "A_paired__D_all", "A_paired__D_paired",
    ]
    summary = summary.set_index("variant_id").reindex(label_order).reset_index()

    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=150)
    colors = [ACCENT_RED if v < 0 else (NAVY if i == len(summary) - 1 else GREY)
              for i, v in enumerate(summary["mean_gap"])]
    xpos = np.arange(len(summary))
    bars = ax.bar(xpos, summary["mean_gap"], color=colors)
    ymax = float(summary["mean_gap"].max())
    for bar, win, n in zip(bars, summary["wins"], summary["n"]):
        h = bar.get_height()
        if h >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.05,
                    f"{win}/{n}", ha="center", va="bottom", fontsize=11, color="#222", fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h - ymax * 0.05,
                    f"{win}/{n}", ha="center", va="top", fontsize=11, color="#222", fontweight="bold")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("mean(wrong − correct) gap on adopt rate")
    ax.set_title("M2 — Adopt variant signal preservation\n(higher gap + 22/22 wins → recommended canonical)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(summary["variant_id"], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(summary["mean_gap"].min(), 0) - ymax * 0.15, ymax * 1.25)
    fig.tight_layout()
    out = FIG_DIR / "paper_M2_variant_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_L1_confidence_quartile() -> Path:
    """Bar chart: per-bin adopt/df trend on the worked-example cell.

    Plots the canonical worked example for paper §4.4 Figure 5: PlotQA S1
    anchor arm × LLaVA-OneVision-7b (Main) × cross_entropy. 6-bin headline
    (switched from 4-bin 2026-05-10 — see roadmap §10). Same cell as the
    §4.4 worked-example table. Source CSV: L1_confidence_quartile_long_6bin.csv.
    """
    df = pd.read_csv(DATA_DIR / "L1_confidence_quartile_long_6bin.csv")
    cell = df[
        (df["experiment"] == "experiment_e7_plotqa_full")
        & (df["dataset"] == "PlotQA")
        & (df["model"] == "llava-onevision-qwen2-7b-ov")
        & (df["cond_class"] == "a")
        & (df["stratum"] == "S1")
        & (df["proxy"] == "cross_entropy")
    ].set_index("quartile")
    if cell.empty:
        raise RuntimeError("worked-example row missing from L1_confidence_quartile_long_6bin.csv")

    bin_order = [f"B{i+1}" for i in range(6)]
    by_q_adopt = cell["adopt_rate"].reindex(bin_order)
    by_q_df = cell["direction_follow_rate"].reindex(bin_order)
    by_q_n = cell["n"].reindex(bin_order)

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    x = np.arange(6)
    width = 0.38
    bars_a = ax.bar(x - width / 2, by_q_adopt.values, width,
                    color=NAVY, edgecolor="black", linewidth=0.4,
                    label="adopt_rate (M2)")
    bars_d = ax.bar(x + width / 2, by_q_df.values, width,
                    color=ACCENT_GOLD, edgecolor="black", linewidth=0.4,
                    label="direction_follow_rate (M2)")
    for i, v in enumerate(by_q_adopt.values):
        ax.text(i - width / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, color=NAVY)
    for i, v in enumerate(by_q_df.values):
        ax.text(i + width / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, color="#7a5a00")
    for i, n in enumerate(by_q_n.values):
        ax.text(i, -0.018, f"n={int(n)}", ha="center", fontsize=8, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([
        "B1\n(most conf)",
        "B2", "B3", "B4", "B5",
        "B6\n(least conf)",
    ])
    gap_df = by_q_df.values[-1] - by_q_df.values[0]
    gap_adopt = by_q_adopt.values[-1] - by_q_adopt.values[0]
    ax.set_ylabel("rate (M2)\nPlotQA × LLaVA-OneVision-7b (Main), S1 worked example", fontsize=10)
    ax.set_title("Figure 5 — L1 6-bin confidence gradient (cross_entropy proxy)\n"
                 f"Less confident base → more anchor pull (B6−B1 gap: df {gap_df:+.3f}, adopt {gap_adopt:+.3f})")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(-0.04, max(by_q_adopt.max(), by_q_df.max()) * 1.30)
    fig.tight_layout()
    out = FIG_DIR / "paper_L1_confidence_quartile.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_E5e_mathvista_bars() -> Path:
    """Side-by-side bars: gemma3-27b/llava/qwen wrong-base S1 adopt(a) vs adopt(m)."""
    df = pd.read_csv(DATA_DIR / "experiment_e5e_mathvista_full_per_cell.csv")
    df = df[(df["base_correct"] == False) & (df["stratum"] == "S1") & (df["n"] >= 100)]
    piv = df.pivot_table(index="model", columns="cond_class", values="adopt_M2", aggfunc="first")
    piv = piv.reindex(["gemma3-27b-it", "llava-next-interleaved-7b", "qwen2.5-vl-7b-instruct"])
    piv = piv[["a", "m"]].rename(columns={"a": "anchor", "m": "masked"})

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(len(piv))
    width = 0.35
    bars_a = ax.bar(x - width / 2, piv["anchor"], width, label="anchor", color=NAVY)
    bars_m = ax.bar(x + width / 2, piv["masked"], width, label="masked", color=ICE,
                    edgecolor=NAVY, linewidth=1.2)
    for b, v in zip(bars_a, piv["anchor"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", color=NAVY, fontsize=11, fontweight="bold")
    for b, v in zip(bars_m, piv["masked"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", color=NAVY, fontsize=11)
    for i, (a, m) in enumerate(zip(piv["anchor"], piv["masked"])):
        gap = a - m
        ax.annotate(f"+{gap*100:.1f} pp", xy=(i, a + 0.015), ha="center",
                    color=ACCENT_GOLD, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=10)
    ax.set_ylabel("adopt_rate (M2, wrong-base S1)")
    ax.set_title("E5e MathVista (γ-α) — digit-pixel causality cross-model\n"
                 "gemma3-27b is the strongest single cell in the program")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, piv.values.max() * 1.3)
    fig.tight_layout()
    out = FIG_DIR / "paper_E5e_mathvista_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cross_dataset_summary() -> Path | None:
    """Heatmap of wrong-base S1 adopt + df across the 5-dataset main matrix.

    Reads `docs/insights/_data/main_panel_5dataset_per_cell.csv`
    (built by scripts/build_e5e_e7_5dataset_summary.py). Restricts to the
    6-model §4.3 main panel (drops llava-next-interleaved-7b — see roadmap
    §3.0a / commit 0e7998e: low native resolution, not informative for
    chart/figure datasets). Orders rows by mean df(a) ascending so most-
    robust models sit at the top.
    """
    src = DATA_DIR / "main_panel_5dataset_per_cell.csv"
    if not src.exists():
        print(f"[skip] {src.name} missing — run scripts/build_e5e_e7_5dataset_summary.py first")
        return None
    df = pd.read_csv(src)
    df = df[df["cond_class"] == "a"].copy()
    df = df[df["model"] != "llava-next-interleaved-7b"]

    pretty = {
        "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7b",
        "qwen2.5-vl-32b-instruct": "Qwen2.5-VL-32b",
        "llava-onevision-qwen2-7b-ov": "LLaVA-OneVision-7b (Main)",
        "internvl3-8b": "InternVL3-8b",
        "gemma3-27b-it": "Gemma3-27b",
        "gemma3-4b-it": "Gemma3-4b",
    }
    df["model_short"] = df["model"].map(pretty).fillna(df["model"])

    metrics = [
        ("adopt_M2",            "adopt_rate(a)",  0.30),
        ("direction_follow_M2", "df(a) C-form",   0.45),
    ]
    dataset_order = ["TallyQA", "ChartQA", "MathVista", "PlotQA", "InfographicVQA"]
    dataset_label = {"InfographicVQA": "InfoVQA"}
    row_order = (df.groupby("model_short")["direction_follow_M2"].mean()
                   .sort_values(ascending=True).index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=150)
    for ax, (col, title, vmax) in zip(axes, metrics):
        piv = df.pivot_table(index="model_short", columns="dataset", values=col, aggfunc="first")
        cols_order = [c for c in dataset_order if c in piv.columns]
        piv = piv.loc[row_order, cols_order]
        im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if pd.isna(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#666666", fontsize=11)
                    continue
                color = "white" if v > vmax * 0.55 else "black"
                weight = "bold" if v >= vmax * 0.5 else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight=weight)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([dataset_label.get(c, c) for c in piv.columns],
                           fontsize=10, rotation=20, ha="right")
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=10)
        ax.set_title(f"S1 wrong-base {title}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.05)
    n_models = len(row_order)
    n_datasets = len([c for c in dataset_order
                      if c in df["dataset"].unique()])
    fig.suptitle(
        f"5-dataset main matrix ({n_models} models × {n_datasets} datasets, "
        "S1 wrong-base, C-form). Rows sorted by mean df(a) ascending — "
        "Qwen2.5-VL family panel-min, gemma3-4b panel-max.",
        y=1.02, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "paper_cross_dataset_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    out1 = fig_M2_variant_comparison()
    out2 = fig_L1_confidence_quartile()
    out3 = fig_E5e_mathvista_bars()
    out4 = fig_cross_dataset_summary()
    print("Generated:")
    for p in (out1, out2, out3, out4):
        if p is None:
            continue
        print(f"  {p}")


if __name__ == "__main__":
    main()
