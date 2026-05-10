"""Build paper Figure 3 (§4.2 digit-pixel causality, two orthogonal slices).

Two-panel figure showing wrong-base × S1 paired adopt rates for the (a, m)
contrast:

  Top    PlotQA × 6-model panel (E7, full {a, d, m})
  Bottom LLaVA-OneVision × 5-dataset (E5b 4 datasets + E5e TallyQA backfill)

Each panel shows paired bars: adopt(a) red, adopt(m) blue; (a-m) gap in pp
labelled above the larger bar.

Inputs:
  docs/insights/_data/experiment_e7_plotqa_full_per_cell.csv
  docs/insights/_data/experiment_e5b_5strat_plotqa_onevision_per_cell.csv
  docs/insights/_data/experiment_e5b_5strat_mathvista_onevision_per_cell.csv
  docs/insights/_data/experiment_e5b_5strat_chartqa_onevision_per_cell.csv
  docs/insights/_data/experiment_e5b_5strat_infographicvqa_onevision_per_cell.csv
  docs/insights/_data/experiment_e5e_tallyqa_full_per_cell.csv

Output: docs/figures/paper_4_2_digit_pixel_causality.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "insights" / "_data"
OUT = ROOT / "docs" / "figures" / "paper_4_2_digit_pixel_causality.png"

PRETTY_MODEL = {
    "gemma3-4b-it": "Gemma3-4b",
    "gemma3-27b-it": "Gemma3-27b",
    "llava-next-interleaved-7b": "LLaVA-Interleave-7b",
    "llava-onevision-qwen2-7b-ov": "LLaVA-OneVision-7b\n(Main)",
    "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7b",
    "qwen2.5-vl-32b-instruct": "Qwen2.5-VL-32b",
}

ONEVISION = "llava-onevision-qwen2-7b-ov"


def wb_s1(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sub = df[(df["stratum"] == "S1")
             & (df["base_correct"] == False)
             & (df["cond_class"].isin(["a", "m"]))]
    return sub[["model", "cond_class", "n_pb_ne_anchor", "adopt_M2"]]


def panel_plotqa() -> pd.DataFrame:
    """PlotQA × 6 models from E7, sorted by adopt(a) desc."""
    raw = wb_s1(DATA / "experiment_e7_plotqa_full_per_cell.csv")
    pivot = raw.pivot_table(index="model", columns="cond_class",
                            values="adopt_M2").reset_index()
    pivot["n_wb"] = (raw[raw["cond_class"] == "a"]
                     .set_index("model")["n_pb_ne_anchor"]
                     .reindex(pivot["model"]).to_numpy())
    pivot["gap_pp"] = (pivot["a"] - pivot["m"]) * 100
    pivot = pivot.sort_values("a", ascending=False).reset_index(drop=True)
    pivot["label"] = pivot["model"].map(PRETTY_MODEL)
    return pivot


def panel_onevision() -> pd.DataFrame:
    """OneVision × 5 datasets, sorted by adopt(a) desc."""
    rows = []
    sources = [
        ("PlotQA",    "experiment_e5b_5strat_plotqa_onevision_per_cell.csv"),
        ("MathVista", "experiment_e5b_5strat_mathvista_onevision_per_cell.csv"),
        ("InfoVQA",   "experiment_e5b_5strat_infographicvqa_onevision_per_cell.csv"),
        ("ChartQA",   "experiment_e5b_5strat_chartqa_onevision_per_cell.csv"),
        ("TallyQA",   "experiment_e5e_tallyqa_full_per_cell.csv"),
    ]
    for ds, fname in sources:
        df = wb_s1(DATA / fname)
        df = df[df["model"] == ONEVISION]
        a = df[df["cond_class"] == "a"]["adopt_M2"].iloc[0]
        m = df[df["cond_class"] == "m"]["adopt_M2"].iloc[0]
        n = df[df["cond_class"] == "a"]["n_pb_ne_anchor"].iloc[0]
        rows.append({"label": ds, "a": a, "m": m,
                     "gap_pp": (a - m) * 100, "n_wb": n})
    out = pd.DataFrame(rows).sort_values("a", ascending=False).reset_index(drop=True)
    return out


def draw_panel(ax, df: pd.DataFrame, title: str) -> None:
    x = range(len(df))
    width = 0.38
    a = df["a"].to_numpy()
    m = df["m"].to_numpy()
    gap = df["gap_pp"].to_numpy()

    ax.bar([i - width / 2 for i in x], a, width,
           color="#d62728", edgecolor="black", linewidth=0.5,
           label="adopt(a) — anchor digit visible")
    ax.bar([i + width / 2 for i in x], m, width,
           color="#1f77b4", edgecolor="black", linewidth=0.5,
           label="adopt(m) — anchor digit inpainted")

    for i, g in enumerate(gap):
        sign = "+" if g >= 0 else ""
        color = "black" if g >= 0 else "#b00020"
        ax.text(i, max(a[i], m[i]) + 0.005,
                f"Δ {sign}{g:.1f} pp", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"].tolist(), rotation=15, ha="right",
                       fontsize=9.5)
    ax.set_ylabel("wrong-base × S1 paired adopt", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(0.20, max(a.max(), m.max()) * 1.25))
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main() -> None:
    plotqa = panel_plotqa()
    onevision = panel_onevision()

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.4))
    draw_panel(
        axes[0], plotqa,
        "§4.2 (top) PlotQA × 6-model — digit-pixel (a−m) gap "
        "(wrong-base × S1, E7)",
    )
    draw_panel(
        axes[1], onevision,
        "§4.2 (bottom) LLaVA-OneVision (Main) × 5 datasets — digit-pixel (a−m) gap "
        "(wrong-base × S1, E5b/E5e)",
    )
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("\nPlotQA panel:")
    print(plotqa[["label", "a", "m", "gap_pp", "n_wb"]].to_string(index=False))
    print("\nOneVision panel:")
    print(onevision[["label", "a", "m", "gap_pp", "n_wb"]].to_string(index=False))


if __name__ == "__main__":
    main()
