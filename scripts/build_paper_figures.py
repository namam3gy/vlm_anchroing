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
    """Line chart: per-quartile adopt/df trend on the canonical entropy proxy."""
    df = pd.read_csv(DATA_DIR / "L1_confidence_quartile_long.csv")
    sub = df[
        (df["proxy"] == "entropy_top_k")
        & (df["cond_class"] == "a")
        & (df["stratum"].isin(["S0", "S1"]))
    ]

    by_q_adopt = sub.groupby("quartile")["adopt_rate"].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
    by_q_df = sub.groupby("quartile")["direction_follow_rate"].mean().reindex(["Q1", "Q2", "Q3", "Q4"])

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    x = np.arange(4)
    ax.plot(x, by_q_adopt.values, "-o", color=NAVY, lw=3, ms=12,
            label="adopt_rate (M2)")
    ax.plot(x, by_q_df.values, "--s", color=ACCENT_GOLD, lw=3, ms=10,
            label="direction_follow_rate (M2)")
    for i, (a, d) in enumerate(zip(by_q_adopt.values, by_q_df.values)):
        ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=10, color=NAVY)
        ax.text(i, d + 0.005, f"{d:.3f}", ha="center", fontsize=10, color=ACCENT_GOLD)

    ax.set_xticks(x)
    ax.set_xticklabels([
        "Q1\n(most confident)",
        "Q2",
        "Q3",
        "Q4\n(least confident)",
    ])
    ax.set_ylabel("rate (mean over signal-bearing cells)")
    ax.set_title("L1 — Confidence-modulated anchoring (entropy_top_k proxy, S0/S1)\n"
                 "Less confident base → more anchor pull (graded)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(by_q_adopt.max(), by_q_df.max()) * 1.4)
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


def fig_cross_dataset_summary() -> Path:
    """Heatmap-ish bar chart: wrong-base S1 adopt across (dataset × model)."""
    rows = []
    e5b = pd.read_csv(DATA_DIR / "E5b_per_stratum.csv")
    e5b_w = e5b[(e5b["stratum"] == "S1") & (e5b["base"] == "wrong")]
    if not e5b_w.empty:
        for _, r in e5b_w.iterrows():
            rows.append({"dataset": r["dataset"], "model": "llava-interleave-7b",
                         "adopt": r["adopt_cond"]})

    e5e_chart = pd.read_csv(DATA_DIR / "experiment_e5e_chartqa_full_per_cell.csv")
    e5e_chart = e5e_chart[(e5e_chart["stratum"] == "S1") & (e5e_chart["cond_class"] == "a")
                          & (e5e_chart["base_correct"] == False)]
    for _, r in e5e_chart.iterrows():
        rows.append({"dataset": "ChartQA", "model": r["model"], "adopt": r["adopt_M2"]})

    e5e_math = pd.read_csv(DATA_DIR / "experiment_e5e_mathvista_full_per_cell.csv")
    e5e_math = e5e_math[(e5e_math["stratum"] == "S1") & (e5e_math["cond_class"] == "a")
                        & (e5e_math["base_correct"] == False) & (e5e_math["n"] >= 100)]
    for _, r in e5e_math.iterrows():
        rows.append({"dataset": "MathVista", "model": r["model"], "adopt": r["adopt_M2"]})

    df = pd.DataFrame(rows)
    df["model_short"] = df["model"].str.replace("-instruct", "").str.replace("-it", "")\
                                    .str.replace("llava-next-interleaved-7b", "llava-interl-7b")
    piv = df.pivot_table(index="model_short", columns="dataset", values="adopt", aggfunc="first")
    cols_order = [c for c in ["VQAv2", "TallyQA", "ChartQA", "MathVista"] if c in piv.columns]
    piv = piv[cols_order]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=0.20)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.isna(v):
                continue
            color = "white" if v > 0.10 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=12,
                    fontweight="bold" if v >= 0.10 else "normal")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=12)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=11)
    ax.set_title("Cross-dataset wrong-base S1 adopt_rate (M2)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("adopt_rate")
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
        print(f"  {p}")


if __name__ == "__main__":
    main()
