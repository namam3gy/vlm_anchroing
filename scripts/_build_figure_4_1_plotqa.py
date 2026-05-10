"""Build paper Figure 2 (§4.1 PlotQA wrong vs correct df bars).

Reads per-model wrong/correct df from section41_swap_analysis.csv, draws a
grouped bar chart (red = wrong-base, blue = correct-base) for the 6-model
PlotQA panel, sorted by df_all descending.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "insights" / "_data" / "section41_swap_analysis.csv"
OUT = ROOT / "docs" / "figures" / "paper_4_1_PlotQA_correct_vs_wrong_df.png"

PRETTY = {
    "gemma3-4b-it": "Gemma3-4b",
    "gemma3-27b-it": "Gemma3-27b",
    "llava-next-interleaved-7b": "LLaVA-Interleave-7b †",
    "llava-onevision-qwen2-7b-ov": "LLaVA-OneVision-7b\n(Main)",
    "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7b",
    "qwen2.5-vl-32b-instruct": "Qwen2.5-VL-32b",
}


def main() -> None:
    df = pd.read_csv(CSV)
    sub = df[df["dataset"] == "PlotQA"].copy().sort_values("df_all", ascending=False)

    labels = [PRETTY.get(m, m) for m in sub["model"]]
    df_w = sub["df_wrong"].to_numpy()
    df_c = sub["df_correct"].to_numpy()
    gap = (df_w - df_c) * 100  # pp

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.6))
    bw = ax.bar([i - width / 2 for i in x], df_w, width,
                color="#d62728", edgecolor="black", linewidth=0.5,
                label="wrong-base")
    bc = ax.bar([i + width / 2 for i in x], df_c, width,
                color="#1f77b4", edgecolor="black", linewidth=0.5,
                label="correct-base")

    for i, g in enumerate(gap):
        sign = "+" if g >= 0 else ""
        color = "black" if g >= 0 else "#b00020"
        ax.text(i, max(df_w[i], df_c[i]) + 0.012,
                f"Δ {sign}{g:.1f} pp", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylabel("direction-follow rate df(a)", fontsize=10)
    ax.set_title("§4.1 Figure 2 — PlotQA wrong vs correct base "
                 "(6-model, S1 anchor)",
                 fontsize=11)
    ax.set_ylim(0, max(df_w.max(), df_c.max()) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
