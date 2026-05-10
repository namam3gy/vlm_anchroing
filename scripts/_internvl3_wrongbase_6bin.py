"""§6.6 / §D B2 falsification — re-run in 6-bin from canonical L1 pair records.

Uses `L1_confidence_pair_records.csv` (the same source as the original 4-bin
B2 falsification — paired records with cross_entropy proxy and per-row
anchor_direction_followed_moved flag) and re-bins into 6 equal-frequency
bins on the wrong-base subset (`exact_match_b == 0`) per InternVL3 cell.

This matches the original 2026-05-04 sample selection exactly (same pair
records → same denominator), only changing the bin-count from 4 → 6.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV = ROOT / "docs/insights/_data/L1_confidence_pair_records.csv"


def per_bin(sub: pd.DataFrame, n_bins: int = 6) -> pd.DataFrame:
    """Sort by cross_entropy ascending (lower = more confident = B1)."""
    sub = sub.dropna(subset=["cross_entropy"]).copy()
    n = len(sub)
    if n < n_bins:
        return pd.DataFrame()
    sub = sub.sort_values("cross_entropy", ascending=True).reset_index(drop=True)
    sub["bin"] = ((np.arange(n) * n_bins) // n).clip(max=n_bins - 1)
    rows = []
    for k in range(n_bins):
        cell = sub[sub["bin"] == k]
        if len(cell) == 0:
            rows.append((f"B{k+1}", 0, np.nan, np.nan))
            continue
        n_k = len(cell)
        n_pb_ne_anchor = int((cell["pred_b_equal_anchor"] == 0).sum())
        n_numeric = int(cell["numeric_distance_to_anchor"].notna().sum())
        adopt = float(cell.loc[cell["pred_b_equal_anchor"] == 0, "anchor_adopted"].sum() / max(1, n_pb_ne_anchor))
        df_rate = float(cell.loc[cell["numeric_distance_to_anchor"].notna(), "anchor_direction_followed_moved"].sum() / max(1, n_numeric))
        rows.append((f"B{k+1}", n_k, adopt, df_rate))
    return pd.DataFrame(rows, columns=["bin", "n", "adopt", "df"])


def main() -> None:
    df = pd.read_csv(PAIR_CSV)
    print(f"Loaded {len(df):,} pair records")

    # InternVL3 chart-stack a-arm S1, wrong-base
    sub = df[(df["model"] == "internvl3-8b") & (df["cond_class"] == "a") & (df["stratum"] == "S1")]
    print(f"InternVL3 a-arm S1: {len(sub):,} records")
    print()

    datasets = ["ChartQA", "InfographicVQA", "PlotQA", "MathVista", "TallyQA"]
    print("=== B2 6-bin wrong-base × df (paired-records canonical) ===")
    print(f"{'dataset':18s}  n_wrong  B1 df   B6 df   Δ(B6-B1)")
    print("-" * 60)
    for ds in datasets:
        cell = sub[sub["dataset"] == ds]
        if len(cell) == 0:
            print(f"{ds:18s}  (no pair records)")
            continue
        wrong = cell[cell["exact_match_b"] == 0]
        result = per_bin(wrong, n_bins=6)
        if result.empty:
            print(f"{ds:18s}  (n_wrong={len(wrong)} < 6)")
            continue
        n_total = int(result["n"].sum())
        b1, b6 = result.iloc[0]["df"], result.iloc[-1]["df"]
        print(f"{ds:18s}  {n_total:7d}  {b1:.4f}  {b6:.4f}  {b6-b1:+.4f}")

    # Also all-base for sanity (should match §6.6 all-base 6-bin numbers)
    print("\n=== Sanity: 6-bin all-base × df ===")
    print(f"{'dataset':18s}  n      B1 df   B6 df   Δ(B6-B1)")
    print("-" * 60)
    for ds in datasets:
        cell = sub[sub["dataset"] == ds]
        if len(cell) == 0:
            continue
        result = per_bin(cell, n_bins=6)
        if result.empty:
            continue
        n_total = int(result["n"].sum())
        b1, b6 = result.iloc[0]["df"], result.iloc[-1]["df"]
        print(f"{ds:18s}  {n_total:7d}  {b1:.4f}  {b6:.4f}  {b6-b1:+.4f}")


if __name__ == "__main__":
    main()
