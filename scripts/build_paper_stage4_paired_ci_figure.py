"""Render the §6.2.3 canonical figure: 5-dataset × {Δdf(a), Δem(b)} × 95 % CI.

Reads `docs/insights/_data/stage4_final_per_dataset_ci.csv`, writes
`docs/figures/paper_6_2_3_stage4_5dataset_paired_ci.png`. One canonical output;
no CLI flags. Run with `uv run python scripts/build_paper_stage4_paired_ci_figure.py`
from the project root.

Layout: two side-by-side horizontal forest plots. Rows sorted by `n_paired`
descending (TallyQA → PlotQA → InfoVQA → ChartQA → MathVista). Left panel =
Δdf(a) with headline direction negative; right panel = Δem(b) with headline
direction positive. 95 % CI whiskers only. Bonferroni-20 CIs are present in the
source CSV but intentionally not drawn (see §A.5 prose).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "insights" / "_data" / "stage4_final_per_dataset_ci.csv"
OUT_PATH = ROOT / "docs" / "figures" / "paper_6_2_3_stage4_5dataset_paired_ci.png"

DATASET_DISPLAY = {
    "TallyQA": "TallyQA",
    "PlotQA": "PlotQA",
    "InfoVQA": "InfoVQA",
    "ChartQA": "ChartQA",
    "MathVista": "MathVista",
}

# Headline colors: anchoring-effect reduction (negative is good) vs capability
# preservation (positive is good). Neutral gray when 95 % CI crosses zero.
COLOR_HEADLINE_DF = "#2c7fb8"  # steel blue (Δdf, negative direction)
COLOR_HEADLINE_EM = "#2ca25f"  # sea green (Δem(b), positive direction)
COLOR_NEUTRAL = "#9e9e9e"


def _excludes_zero_negative(lo: float, hi: float) -> bool:
    return hi < 0.0


def _excludes_zero_positive(lo: float, hi: float) -> bool:
    return lo > 0.0


def _draw_forest_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    point_col: str,
    lo_col: str,
    hi_col: str,
    direction: str,  # "negative" or "positive"
    headline_color: str,
    title: str,
    xlabel: str,
) -> None:
    """Render one forest panel.

    `df` rows are taken in caller-provided order (top→bottom in figure). Values
    in source CSV are fractions; we render in percentage points.
    """
    n_rows = len(df)
    y_positions = list(range(n_rows, 0, -1))  # top row gets largest y

    for y, (_, row) in zip(y_positions, df.iterrows()):
        point = row[point_col] * 100.0
        lo = row[lo_col] * 100.0
        hi = row[hi_col] * 100.0
        if direction == "negative":
            excludes = _excludes_zero_negative(lo, hi)
        else:
            excludes = _excludes_zero_positive(lo, hi)
        color = headline_color if excludes else COLOR_NEUTRAL
        ax.hlines(y=y, xmin=lo, xmax=hi, color=color, linewidth=2.4, zorder=2)
        ax.plot(
            [lo, hi], [y, y], marker="|", markersize=9, color=color,
            linestyle="None", zorder=3,
        )
        ax.scatter(
            [point], [y], color=color, s=58, zorder=4, edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            1.02, y, f"n = {int(row['n_paired']):,}",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="#444444",
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6, zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DATASET_DISPLAY[d] for d in df["dataset"]], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0.4, n_rows + 0.6)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Canonical CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    expected = {"TallyQA", "PlotQA", "InfoVQA", "ChartQA", "MathVista"}
    missing = expected - set(df["dataset"])
    if missing:
        raise ValueError(f"CSV missing expected datasets: {missing}")

    df = df.sort_values("n_paired", ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_df, ax_em) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    _draw_forest_panel(
        ax_df,
        df,
        point_col="delta_df",
        lo_col="delta_df_ci95_lo",
        hi_col="delta_df_ci95_hi",
        direction="negative",
        headline_color=COLOR_HEADLINE_DF,
        title=r"$\Delta$ df(a)  —  anchoring effect (headline: $\downarrow$)",
        xlabel=r"$\Delta$ direction-follow rate (pp)",
    )
    _draw_forest_panel(
        ax_em,
        df,
        point_col="delta_em_b",
        lo_col="delta_em_b_ci95_lo",
        hi_col="delta_em_b_ci95_hi",
        direction="positive",
        headline_color=COLOR_HEADLINE_EM,
        title=r"$\Delta$ em(b)  —  non-anchored arm (headline: $\uparrow$)",
        xlabel=r"$\Delta$ exact-match on non-anchored arm (pp)",
    )

    # Allow right-side n-annotations to render outside axis.
    for ax in (ax_df, ax_em):
        ax.margins(x=0.18)

    fig.suptitle(
        r"E6 Stage-4 (L = 26, K = 8, $\alpha$ = 1.0): 5-dataset paired-bootstrap deltas",
        fontsize=12.5, y=0.995,
    )
    fig.text(
        0.5, 0.93,
        "B = 10,000 paired sids; 95 % CI shown. Bonferroni-20 corrected CIs in §A.5.",
        ha="center", va="top", fontsize=9.5, color="#555555",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
