"""Build per-question universal-susceptibility scores from the 7-model VQAv2 main runs.

Output: a CSV ranking every question by mean cross-model `moved_closer_rate`,
with top/bottom decile flags. This is the sample-frame for E1 (attention mass
on hard cases) and any future paired analysis that wants
"items where every model anchors" vs "items where no model anchors".

Usage:
    uv run python research/scripts/extract_susceptibility_strata.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vlm_anchor.analysis import build_paired_dataframe, filter_anchor_distance_outliers


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "outputs" / "experiment"
OUT = PROJECT / "research" / "insights" / "_data" / "susceptibility_strata.csv"


def _load() -> pd.DataFrame:
    frames = []
    for model_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if model_dir.name == "analysis":
            continue
        cand = sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "predictions.csv").exists())
        if not cand:
            continue
        df = pd.read_csv(cand[-1] / "predictions.csv", low_memory=False)
        df["model"] = model_dir.name
        df["model_root"] = str(cand[-1].resolve())
        df["experiment_root"] = str(ROOT.resolve())
        df["experiment_name"] = ROOT.resolve().name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    raw = _load()
    from vlm_anchor.analysis import CONDITION_LABELS, CONDITION_ORDER, parse_int_like
    raw["condition"] = pd.Categorical(raw["condition"], CONDITION_ORDER, ordered=True)
    raw["condition_label"] = raw["condition"].map(CONDITION_LABELS)
    raw["prediction_int"] = raw["prediction"].map(parse_int_like)
    raw["ground_truth_int"] = raw["ground_truth"].map(parse_int_like)
    raw["anchor_int"] = raw["anchor_value"].map(parse_int_like)
    raw["is_numeric_prediction"] = raw["prediction_int"].notna()
    raw["is_numeric_ground_truth"] = raw["ground_truth_int"].notna()
    raw["is_numeric_anchor"] = raw["anchor_int"].notna()
    raw["sample_instance_index"] = pd.to_numeric(raw["sample_instance_index"], errors="coerce")
    raw["question_id"] = pd.to_numeric(raw["question_id"], errors="coerce")
    raw["image_id"] = pd.to_numeric(raw["image_id"], errors="coerce")
    raw = raw.sort_values(["model", "sample_instance_id", "condition"]).reset_index(drop=True)

    paired = build_paired_dataframe(raw)
    _, paired, _, _ = filter_anchor_distance_outliers(raw, paired)
    valid = paired.loc[paired["number_numeric_mask"]].copy()

    per_model_q = (
        valid.groupby(["model", "question_id"])
        .agg(
            moved_closer_rate=("moved_closer_to_anchor", "mean"),
            adoption_rate=("number_anchor_adopted", "mean"),
            mean_pull=("anchor_pull", "mean"),
            n=("anchor_pull", "size"),
        )
        .reset_index()
    )

    per_q = (
        per_model_q.groupby("question_id")
        .agg(
            n_models=("model", "nunique"),
            mean_moved_closer=("moved_closer_rate", "mean"),
            std_moved_closer=("moved_closer_rate", "std"),
            mean_adoption=("adoption_rate", "mean"),
            mean_pull=("mean_pull", "mean"),
        )
        .reset_index()
    )

    # join with question text + image_id from a single representative row
    rep = (
        valid[["question_id", "question", "question_type", "image_id", "base_ground_truth"]]
        .drop_duplicates(subset=["question_id"], keep="first")
    )
    per_q = per_q.merge(rep, on="question_id", how="left")

    per_q = per_q.sort_values("mean_moved_closer", ascending=False).reset_index(drop=True)
    decile_cuts = np.percentile(per_q["mean_moved_closer"].dropna(), [10, 90])
    per_q["susceptibility_stratum"] = "middle"
    per_q.loc[per_q["mean_moved_closer"] <= decile_cuts[0], "susceptibility_stratum"] = "bottom_decile_resistant"
    per_q.loc[per_q["mean_moved_closer"] >= decile_cuts[1], "susceptibility_stratum"] = "top_decile_susceptible"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_q.to_csv(OUT, index=False)

    n_top = int((per_q["susceptibility_stratum"] == "top_decile_susceptible").sum())
    n_bot = int((per_q["susceptibility_stratum"] == "bottom_decile_resistant").sum())
    print(f"questions: {len(per_q)} | top decile: {n_top} | bottom decile: {n_bot}")
    print(f"top-decile threshold: mean_moved_closer >= {decile_cuts[1]:.3f}")
    print(f"bottom-decile threshold: mean_moved_closer <= {decile_cuts[0]:.3f}")
    print(f"\nTop-5 universally susceptible questions:")
    print(per_q.head(5)[["question_id", "image_id", "question", "base_ground_truth", "mean_moved_closer", "std_moved_closer"]].to_string(index=False))
    print(f"\nBottom-5 universally resistant questions:")
    print(per_q.tail(5)[["question_id", "image_id", "question", "base_ground_truth", "mean_moved_closer", "std_moved_closer"]].to_string(index=False))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
