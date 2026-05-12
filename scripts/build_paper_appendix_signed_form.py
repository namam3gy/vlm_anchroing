"""Build Appendix G signed-form robustness table for §6 direction-follow metric.

For OneVision (Main) on 5 paper datasets (PlotQA, InfoVQA, TallyQA, ChartQA,
MathVista), partitions a-arm and m-arm predictions into {toward, away, tie}
relative to the anchor direction, then derives the signed-form net pull and
contrasts it against the paper's rate-form direction_follow metric.

Reads from canonical full-run prediction CSVs (single-stratum S1), writes
`docs/insights/_data/appendix_G_signed_form_onevision.csv`. Run with
`uv run python scripts/build_paper_appendix_signed_form.py` from the project root.

Sign convention (matches `vlm_anchor.metrics` C-form):
    toward  ↔ (pa - pb) * (anchor - pb) >  0  AND  pa != pb
    away    ↔ (pa - pb) * (anchor - pb) <  0  AND  pa != pb
    tie     ↔ pa == pb  (or anchor == pb, in which case the sample is dropped)

Only samples with numeric pa, pb, anchor AND anchor != pb contribute to the
denominator (same eligibility filter as the headline direction-follow rate).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "docs" / "insights" / "_data" / "appendix_G_signed_form_onevision.csv"

RUNS = {
    "PlotQA":    "outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/20260502-132624/predictions.csv",
    "InfoVQA":   "outputs/experiment_e7_infographicvqa_full/llava-onevision-qwen2-7b-ov/20260502-152105/predictions.csv",
    "TallyQA":   "outputs/experiment_e5e_tallyqa_full/llava-onevision-qwen2-7b-ov/20260502-083926/predictions.csv",
    "ChartQA":   "outputs/experiment_e5e_chartqa_full/llava-onevision-qwen2-7b-ov/20260502-211028/predictions.csv",
    "MathVista": "outputs/experiment_e5e_mathvista_full/llava-onevision-qwen2-7b-ov/20260502-212440/predictions.csv",
}

ARM_CONDITION = {
    "a": "target_plus_irrelevant_number_S1",
    "m": "target_plus_irrelevant_number_masked_S1",
}


def _split_one_arm(df: pd.DataFrame, arm_condition: str) -> dict:
    """Return counts and fractions of {toward, away, tie} for a single arm."""
    b = df.loc[df["condition"] == "target_only", ["sample_instance_id", "prediction"]]
    b = b.rename(columns={"prediction": "pb"})
    a = df.loc[df["condition"] == arm_condition,
               ["sample_instance_id", "prediction", "anchor_value"]]
    a = a.rename(columns={"prediction": "pa"})
    merged = b.merge(a, on="sample_instance_id", how="inner")
    merged["pb_n"] = pd.to_numeric(merged["pb"], errors="coerce")
    merged["pa_n"] = pd.to_numeric(merged["pa"], errors="coerce")
    merged["anchor_n"] = pd.to_numeric(merged["anchor_value"], errors="coerce")
    eligible = merged.dropna(subset=["pb_n", "pa_n", "anchor_n"]).copy()
    eligible = eligible[eligible["anchor_n"] != eligible["pb_n"]]
    n = len(eligible)
    if n == 0:
        return dict(n=0, P_toward=float("nan"), P_away=float("nan"),
                    P_tie=float("nan"), net_pull=float("nan"),
                    toward_over_away=float("nan"))
    dot = (eligible["pa_n"] - eligible["pb_n"]) * (eligible["anchor_n"] - eligible["pb_n"])
    moved = eligible["pa_n"] != eligible["pb_n"]
    toward = int(((dot > 0) & moved).sum())
    away = int(((dot < 0) & moved).sum())
    tie = int((~moved).sum())
    P_toward = toward / n
    P_away = away / n
    P_tie = tie / n
    return dict(
        n=n,
        n_toward=toward,
        n_away=away,
        n_tie=tie,
        P_toward=P_toward,
        P_away=P_away,
        P_tie=P_tie,
        net_pull=P_toward - P_away,
        toward_over_away=(toward / away) if away > 0 else float("inf"),
    )


def build_table() -> pd.DataFrame:
    rows = []
    for dataset, rel_path in RUNS.items():
        df = pd.read_csv(ROOT / rel_path, low_memory=False)
        stats_a = _split_one_arm(df, ARM_CONDITION["a"])
        stats_m = _split_one_arm(df, ARM_CONDITION["m"])
        rows.append({
            "dataset": dataset,
            "n_eligible": stats_a["n"],
            # a-arm
            "P_toward_a": stats_a["P_toward"],
            "P_away_a":   stats_a["P_away"],
            "P_tie_a":    stats_a["P_tie"],
            "toward_over_away_a": stats_a["toward_over_away"],
            "net_pull_a": stats_a["net_pull"],
            # m-arm
            "P_toward_m": stats_m["P_toward"],
            "P_away_m":   stats_m["P_away"],
            "P_tie_m":    stats_m["P_tie"],
            "net_pull_m": stats_m["net_pull"],
            # (a − m) paired diffs in both metric forms
            "df_rate_a_minus_m":  stats_a["P_toward"] - stats_m["P_toward"],
            "net_pull_a_minus_m": stats_a["net_pull"] - stats_m["net_pull"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    table = build_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(table.to_string(index=False))
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
