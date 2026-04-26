"""Phase-A re-analysis of the existing 7-model VQAv2 runs.

Produces per-insight CSVs under docs/insights/_data/. The accompanying
insight markdowns are then written by hand from those numbers (so the prose
stays grounded in measured values, not paraphrase).

Usage:
    uv run python scripts/phase_a_data_mining.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from vlm_anchor.analysis import (
    build_paired_dataframe,
    filter_anchor_distance_outliers,
    load_experiment_records,
    summarize_anchor_distance_response,
    summarize_failure_stratification,
    summarize_question_type_behavior,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_ROOT = PROJECT_ROOT / "outputs" / "experiment"
STRENGTHEN_ROOT = PROJECT_ROOT / "outputs" / "experiment_anchor_strengthen_prompt"
OUT_DIR = PROJECT_ROOT / "research" / "insights" / "_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_model_runs(root: Path) -> dict[str, Path]:
    """For each model dir under root, pick the latest run that has predictions.csv."""
    runs: dict[str, Path] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name == "analysis":
            continue
        candidates = sorted(p for p in model_dir.iterdir() if p.is_dir() and (p / "predictions.csv").exists())
        if candidates:
            runs[model_dir.name] = candidates[-1]
    return runs


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    runs = _resolve_model_runs(root)
    if not runs:
        raise FileNotFoundError(f"No model runs with predictions.csv under {root}")
    frames = []
    for model, run_dir in runs.items():
        df = pd.read_csv(run_dir / "predictions.csv")
        df["model"] = model
        df["model_root"] = str(run_dir.resolve())
        df["experiment_root"] = str(root.resolve())
        df["experiment_name"] = root.resolve().name
        frames.append(df)
    records = pd.concat(frames, ignore_index=True)

    # apply the same casting that load_experiment_records does
    from vlm_anchor.analysis import CONDITION_LABELS, CONDITION_ORDER, parse_int_like
    records["condition"] = pd.Categorical(records["condition"], CONDITION_ORDER, ordered=True)
    records["condition_label"] = records["condition"].map(CONDITION_LABELS)
    records["prediction_int"] = records["prediction"].map(parse_int_like)
    records["ground_truth_int"] = records["ground_truth"].map(parse_int_like)
    records["anchor_int"] = records["anchor_value"].map(parse_int_like)
    records["is_numeric_prediction"] = records["prediction_int"].notna()
    records["is_numeric_ground_truth"] = records["ground_truth_int"].notna()
    records["is_numeric_anchor"] = records["anchor_int"].notna()
    records["sample_instance_index"] = pd.to_numeric(records["sample_instance_index"], errors="coerce")
    records["question_id"] = pd.to_numeric(records["question_id"], errors="coerce")
    records["image_id"] = pd.to_numeric(records["image_id"], errors="coerce")
    records = records.sort_values(["model", "sample_instance_id", "condition"]).reset_index(drop=True)

    paired = build_paired_dataframe(records)
    records_f, paired_f, _, summary = filter_anchor_distance_outliers(records, paired)
    return records_f, paired_f, summary


def a1_asymmetric_on_wrong(paired: pd.DataFrame) -> pd.DataFrame:
    """Adoption / direction-follow / pull, stratified by base correctness."""
    valid = paired.loc[paired["number_numeric_mask"]].copy()
    rows = []
    for model, sub in valid.groupby("model", sort=True):
        for outcome_label, mask in [("base_correct", sub["base_correct"]), ("base_wrong", ~sub["base_correct"])]:
            seg = sub.loc[mask]
            if seg.empty:
                continue
            rows.append({
                "model": model,
                "stratum": outcome_label,
                "n_pairs": int(seg.shape[0]),
                "adoption_rate": float(seg["number_anchor_adopted"].mean()),
                "moved_closer_rate": float(seg["moved_closer_to_anchor"].mean()),
                "mean_anchor_pull": float(seg["anchor_pull"].mean()),
                "median_anchor_pull": float(seg["anchor_pull"].median()),
                "mean_signed_anchor_movement": float(seg["signed_anchor_movement"].mean()),
                "changed_rate": float(seg["number_changed"].mean()),
            })
    df = pd.DataFrame(rows)

    # paired (wide) view per model for easy gap reading
    wide_rows = []
    for model, sub in df.groupby("model"):
        cols = sub.set_index("stratum")
        if "base_wrong" not in cols.index or "base_correct" not in cols.index:
            continue
        wide_rows.append({
            "model": model,
            "n_correct": int(cols.loc["base_correct", "n_pairs"]),
            "n_wrong": int(cols.loc["base_wrong", "n_pairs"]),
            "adoption_correct": cols.loc["base_correct", "adoption_rate"],
            "adoption_wrong": cols.loc["base_wrong", "adoption_rate"],
            "adoption_gap": cols.loc["base_wrong", "adoption_rate"] - cols.loc["base_correct", "adoption_rate"],
            "moved_closer_correct": cols.loc["base_correct", "moved_closer_rate"],
            "moved_closer_wrong": cols.loc["base_wrong", "moved_closer_rate"],
            "moved_closer_gap": cols.loc["base_wrong", "moved_closer_rate"] - cols.loc["base_correct", "moved_closer_rate"],
            "pull_correct": cols.loc["base_correct", "mean_anchor_pull"],
            "pull_wrong": cols.loc["base_wrong", "mean_anchor_pull"],
            "pull_gap": cols.loc["base_wrong", "mean_anchor_pull"] - cols.loc["base_correct", "mean_anchor_pull"],
        })
    wide = pd.DataFrame(wide_rows)
    return df, wide


def a2_per_anchor_value(paired: pd.DataFrame) -> pd.DataFrame:
    """For each anchor digit, mean signed shift and adoption rate per model."""
    valid = paired.loc[paired["number_numeric_mask"]].copy()
    rows = []
    for (model, anchor_int), seg in valid.groupby(["model", "number_anchor_int"], sort=True):
        rows.append({
            "model": model,
            "anchor_value": int(anchor_int),
            "n_pairs": int(seg.shape[0]),
            "adoption_rate": float(seg["number_anchor_adopted"].mean()),
            "moved_closer_rate": float(seg["moved_closer_to_anchor"].mean()),
            "mean_anchor_pull": float(seg["anchor_pull"].mean()),
            "median_anchor_pull": float(seg["anchor_pull"].median()),
            "mean_signed_shift": float((seg["number_prediction_int"] - seg["base_prediction_int"]).mean()),
            "median_signed_shift": float((seg["number_prediction_int"] - seg["base_prediction_int"]).median()),
        })
    return pd.DataFrame(rows)


def a4_shift_distribution(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Histogram of per-pair shift sizes; bucket counts."""
    valid = paired.loc[paired["number_numeric_mask"]].copy()
    rows = []
    for model, sub in valid.groupby("model", sort=True):
        shifts = (sub["number_prediction_int"] - sub["base_prediction_int"]).astype(float)
        rows.append({
            "model": model,
            "n_pairs": int(sub.shape[0]),
            "share_no_change": float((shifts == 0).mean()),
            "share_change_small": float(((shifts != 0) & (shifts.abs() <= 2)).mean()),
            "share_change_large": float((shifts.abs() >= 5).mean()),
            "share_full_anchor_match": float(sub["number_anchor_adopted"].mean()),
            "shift_p10": float(np.percentile(shifts.dropna(), 10)),
            "shift_p25": float(np.percentile(shifts.dropna(), 25)),
            "shift_median": float(shifts.median()),
            "shift_p75": float(np.percentile(shifts.dropna(), 75)),
            "shift_p90": float(np.percentile(shifts.dropna(), 90)),
        })
    summary = pd.DataFrame(rows)

    # raw histograms for plotting later
    bins = np.arange(-10, 11)
    hist_rows = []
    for model, sub in valid.groupby("model", sort=True):
        shifts = (sub["number_prediction_int"] - sub["base_prediction_int"]).astype(int)
        counts, edges = np.histogram(shifts.clip(-10, 10), bins=bins)
        for edge, count in zip(edges[:-1], counts):
            hist_rows.append({"model": model, "shift_bin": int(edge), "count": int(count)})
    hist = pd.DataFrame(hist_rows)
    return summary, hist


def a6_failure_modes(paired: pd.DataFrame) -> pd.DataFrame:
    """Bucket each number-condition prediction into 5 failure modes."""
    sub = paired.copy()
    rows = []
    for model, seg in sub.groupby("model", sort=True):
        n = int(seg.shape[0])
        non_numeric = int((~seg["number_numeric_mask"]).sum())
        valid = seg.loc[seg["number_numeric_mask"]]
        exact_anchor = int((valid["number_prediction_int"] == valid["number_anchor_int"]).sum())
        unchanged = int((valid["number_prediction_int"] == valid["base_prediction_int"]).sum())
        # exclude exact_anchor and unchanged from "graded toward anchor"
        moved = valid.loc[(valid["number_prediction_int"] != valid["base_prediction_int"]) &
                         (valid["number_prediction_int"] != valid["number_anchor_int"])]
        toward_anchor = int(moved["moved_closer_to_anchor"].fillna(False).sum())
        away_from_anchor = int(moved.shape[0] - toward_anchor)
        rows.append({
            "model": model,
            "n": n,
            "share_non_numeric": non_numeric / n,
            "share_exact_anchor": exact_anchor / n,
            "share_unchanged": unchanged / n,
            "share_graded_toward_anchor": toward_anchor / n,
            "share_orthogonal_or_away": away_from_anchor / n,
        })
    return pd.DataFrame(rows)


def a7_cross_model_agreement(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-question susceptibility scores across models + correlation matrix."""
    valid = paired.loc[paired["number_numeric_mask"]].copy()
    # collapse multiple sample_instance_index per (model, question_id) to a per-question score
    per_q = (
        valid.groupby(["model", "question_id"])
        .agg(
            adoption_rate=("number_anchor_adopted", "mean"),
            moved_closer_rate=("moved_closer_to_anchor", "mean"),
            mean_anchor_pull=("anchor_pull", "mean"),
            n=("anchor_pull", "size"),
        )
        .reset_index()
    )

    # pivot to model x question_id matrix on moved_closer_rate
    pivot = per_q.pivot(index="question_id", columns="model", values="moved_closer_rate")
    corr = pivot.corr(method="spearman")
    return per_q, corr


def main() -> None:
    artifacts: dict[str, str] = {}

    print("[load] standard run")
    rec_std, pair_std, _ = _load(STANDARD_ROOT)

    print(f"  records={len(rec_std)} pairs={len(pair_std)} models={pair_std['model'].nunique()}")

    print("[A1] asymmetric anchoring on wrong cases")
    a1_long, a1_wide = a1_asymmetric_on_wrong(pair_std)
    a1_long.to_csv(OUT_DIR / "A1_asymmetric_long.csv", index=False)
    a1_wide.to_csv(OUT_DIR / "A1_asymmetric_wide.csv", index=False)
    artifacts["A1"] = str(OUT_DIR / "A1_asymmetric_wide.csv")

    print("[A2] per-anchor-value pull")
    a2 = a2_per_anchor_value(pair_std)
    a2.to_csv(OUT_DIR / "A2_per_anchor_value.csv", index=False)
    artifacts["A2"] = str(OUT_DIR / "A2_per_anchor_value.csv")

    print("[A3] question-type stratification")
    a3 = summarize_question_type_behavior(pair_std)
    a3.to_csv(OUT_DIR / "A3_question_type.csv", index=False)
    artifacts["A3"] = str(OUT_DIR / "A3_question_type.csv")

    print("[A4] per-pair shift distribution")
    a4_summary, a4_hist = a4_shift_distribution(pair_std)
    a4_summary.to_csv(OUT_DIR / "A4_shift_summary.csv", index=False)
    a4_hist.to_csv(OUT_DIR / "A4_shift_histogram.csv", index=False)
    artifacts["A4_summary"] = str(OUT_DIR / "A4_shift_summary.csv")
    artifacts["A4_hist"] = str(OUT_DIR / "A4_shift_histogram.csv")

    print("[A5] standard vs strengthen prompt comparison")
    if STRENGTHEN_ROOT.exists():
        rec_str, pair_str, _ = _load(STRENGTHEN_ROOT)
        a5_rows = []
        for label, paired in [("standard", pair_std), ("strengthen", pair_str)]:
            valid = paired.loc[paired["number_numeric_mask"]]
            for model, seg in valid.groupby("model", sort=True):
                a5_rows.append({
                    "model": model,
                    "prompt": label,
                    "n_pairs": int(seg.shape[0]),
                    "adoption_rate": float(seg["number_anchor_adopted"].mean()),
                    "moved_closer_rate": float(seg["moved_closer_to_anchor"].mean()),
                    "mean_anchor_pull": float(seg["anchor_pull"].mean()),
                    "median_anchor_pull": float(seg["anchor_pull"].median()),
                    "mean_signed_movement": float(seg["signed_anchor_movement"].mean()),
                })
        a5 = pd.DataFrame(a5_rows)
        a5.to_csv(OUT_DIR / "A5_prompt_comparison.csv", index=False)

        # paired view
        wide_rows = []
        for model in sorted(set(a5["model"])):
            sub = a5.loc[a5["model"] == model].set_index("prompt")
            if "standard" in sub.index and "strengthen" in sub.index:
                wide_rows.append({
                    "model": model,
                    "adoption_std": sub.loc["standard", "adoption_rate"],
                    "adoption_str": sub.loc["strengthen", "adoption_rate"],
                    "adoption_delta": sub.loc["strengthen", "adoption_rate"] - sub.loc["standard", "adoption_rate"],
                    "moved_closer_std": sub.loc["standard", "moved_closer_rate"],
                    "moved_closer_str": sub.loc["strengthen", "moved_closer_rate"],
                    "moved_closer_delta": sub.loc["strengthen", "moved_closer_rate"] - sub.loc["standard", "moved_closer_rate"],
                })
        pd.DataFrame(wide_rows).to_csv(OUT_DIR / "A5_prompt_comparison_wide.csv", index=False)
        artifacts["A5"] = str(OUT_DIR / "A5_prompt_comparison_wide.csv")
    else:
        print("  (skipped: strengthen root missing)")

    print("[A6] failure-mode taxonomy")
    a6 = a6_failure_modes(pair_std)
    a6.to_csv(OUT_DIR / "A6_failure_modes.csv", index=False)
    artifacts["A6"] = str(OUT_DIR / "A6_failure_modes.csv")

    print("[A7] cross-model item agreement")
    a7_per_q, a7_corr = a7_cross_model_agreement(pair_std)
    a7_per_q.to_csv(OUT_DIR / "A7_per_question.csv", index=False)
    a7_corr.to_csv(OUT_DIR / "A7_model_correlation.csv")
    artifacts["A7_per_question"] = str(OUT_DIR / "A7_per_question.csv")
    artifacts["A7_corr"] = str(OUT_DIR / "A7_model_correlation.csv")

    print("[bonus] anchor-distance response (already in analysis.py)")
    a_dist = summarize_anchor_distance_response(pair_std)
    a_dist.to_csv(OUT_DIR / "A_extra_anchor_distance_response.csv", index=False)
    artifacts["anchor_distance"] = str(OUT_DIR / "A_extra_anchor_distance_response.csv")

    with open(OUT_DIR / "_artifact_index.json", "w") as fh:
        json.dump(artifacts, fh, indent=2)
    print(f"\n[done] artifacts written under {OUT_DIR}")


if __name__ == "__main__":
    main()
