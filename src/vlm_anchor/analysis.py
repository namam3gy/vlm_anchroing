from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CONDITION_ORDER = [
    "target_only",
    "target_plus_irrelevant_number",
    "target_plus_irrelevant_neutral",
]
CONDITION_LABELS = {
    "target_only": "target only",
    "target_plus_irrelevant_number": "+ irrelevant number",
    "target_plus_irrelevant_neutral": "+ irrelevant neutral",
}
CONDITION_PREFIXES = {
    "target_only": "base",
    "target_plus_irrelevant_number": "number",
    "target_plus_irrelevant_neutral": "neutral",
}
DISTRACTOR_LABELS = {
    "number": "irrelevant number",
    "neutral": "irrelevant neutral",
}
DISTANCE_BIN_LABELS = ["[0,5)", "[5,10)", "[10,20)", "[20,50)", "[50,+)"]
DISTANCE_BIN_EDGES = [0, 5, 10, 20, 50, np.inf]
DEFAULT_OUTLIER_IQR_MULTIPLIER = 1.5


def set_notebook_style() -> None:
    import plotly.io as pio
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    pio.templates.default = "plotly_white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def parse_int_like(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() else None

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.0+", text):
        return int(float(text))
    return None


def trimmed_mean(values: Sequence[float], proportion: float = 0.1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    trim = int(arr.size * proportion)
    if trim > 0 and arr.size > (2 * trim):
        arr = arr[trim:-trim]
    return float(arr.mean())


def bootstrap_mean_ci(
    values: Iterable[float],
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
    ci: float = 95.0,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        value = float(arr[0])
        return (value, value)

    rng = np.random.default_rng(rng_seed)
    draws = np.empty(bootstrap_samples, dtype=float)
    for idx in range(bootstrap_samples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        draws[idx] = sample.mean()

    alpha = (100.0 - ci) / 2.0
    return (float(np.percentile(draws, alpha)), float(np.percentile(draws, 100.0 - alpha)))


def _safe_rate(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return float(numeric.mean())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_prediction_files(experiment_root: str | Path, model_filter: Sequence[str] | None = None) -> list[Path]:
    root = Path(experiment_root)
    if root.is_file():
        return [root]

    prediction_files: list[Path] = []
    direct = root / "predictions.csv"
    if direct.exists():
        prediction_files = [direct]
    else:
        prediction_files = sorted(
            child / "predictions.csv"
            for child in root.iterdir()
            if child.is_dir() and (child / "predictions.csv").exists()
        )

    if model_filter:
        wanted = set(model_filter)
        prediction_files = [path for path in prediction_files if path.parent.name in wanted]

    return prediction_files


def load_experiment_records(
    experiment_root: str | Path,
    model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    prediction_files = resolve_prediction_files(experiment_root, model_filter=model_filter)
    if not prediction_files:
        raise FileNotFoundError(f"No predictions.csv files found under {experiment_root}")

    records = []
    for prediction_path in prediction_files:
        df = pd.read_csv(prediction_path)
        df["model"] = df.get("model") if "model" in df.columns else prediction_path.parent.name
        df["model_root"] = str(prediction_path.parent.resolve())
        df["experiment_root"] = str(Path(experiment_root).resolve())
        df["experiment_name"] = Path(experiment_root).resolve().name
        records.append(df)

    records_df = pd.concat(records)
    if records_df.empty:
        raise ValueError(f"No records found under {experiment_root}")

    records_df["condition"] = pd.Categorical(records_df["condition"], CONDITION_ORDER, ordered=True)
    records_df["condition_label"] = records_df["condition"].map(CONDITION_LABELS)
    records_df["prediction_int"] = records_df["prediction"].map(parse_int_like)
    records_df["ground_truth_int"] = records_df["ground_truth"].map(parse_int_like)
    records_df["anchor_int"] = records_df["anchor_value"].map(parse_int_like)
    records_df["is_numeric_prediction"] = records_df["prediction_int"].notna()
    records_df["is_numeric_ground_truth"] = records_df["ground_truth_int"].notna()
    records_df["is_numeric_anchor"] = records_df["anchor_int"].notna()
    records_df["sample_instance_index"] = pd.to_numeric(records_df["sample_instance_index"], errors="coerce")
    records_df["question_id"] = pd.to_numeric(records_df["question_id"], errors="coerce")
    records_df["image_id"] = pd.to_numeric(records_df["image_id"], errors="coerce")
    return records_df.sort_values(["model", "sample_instance_id", "condition"]).reset_index(drop=True)


def build_paired_dataframe(records_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "experiment_root",
        "experiment_name",
        "model",
        "model_root",
        "sample_instance_id",
        "sample_instance_index",
        "question_id",
        "image_id",
        "question",
        "question_type",
    ]
    skip_cols = set(key_cols + ["condition", "condition_label"])
    value_cols = [column for column in records_df.columns if column not in skip_cols]

    frames: dict[str, pd.DataFrame] = {}
    for condition in CONDITION_ORDER:
        prefix = CONDITION_PREFIXES[condition]
        subset = records_df.loc[records_df["condition"] == condition, key_cols + value_cols].copy()
        rename_map = {column: f"{prefix}_{column}" for column in value_cols}
        frames[prefix] = subset.rename(columns=rename_map)

    paired_df = frames["base"].merge(frames["number"], on=key_cols, how="inner").merge(frames["neutral"], on=key_cols, how="inner")

    paired_df["base_correct"] = paired_df["base_exact_match"] == 1
    paired_df["number_correct"] = paired_df["number_exact_match"] == 1
    paired_df["neutral_correct"] = paired_df["neutral_exact_match"] == 1
    paired_df["number_accuracy_delta"] = paired_df["number_standard_vqa_accuracy"] - paired_df["base_standard_vqa_accuracy"]
    paired_df["neutral_accuracy_delta"] = paired_df["neutral_standard_vqa_accuracy"] - paired_df["base_standard_vqa_accuracy"]
    paired_df["number_exact_delta"] = paired_df["number_exact_match"] - paired_df["base_exact_match"]
    paired_df["neutral_exact_delta"] = paired_df["neutral_exact_match"] - paired_df["base_exact_match"]

    number_numeric_mask = (
        paired_df["base_prediction_int"].notna()
        & paired_df["number_prediction_int"].notna()
        & paired_df["base_ground_truth_int"].notna()
        & paired_df["number_anchor_int"].notna()
    )
    neutral_numeric_mask = (
        paired_df["base_prediction_int"].notna()
        & paired_df["neutral_prediction_int"].notna()
        & paired_df["base_ground_truth_int"].notna()
    )

    paired_df["number_numeric_mask"] = number_numeric_mask
    paired_df["neutral_numeric_mask"] = neutral_numeric_mask
    paired_df["number_shift"] = np.where(
        number_numeric_mask,
        paired_df["number_prediction_int"] - paired_df["base_prediction_int"],
        np.nan,
    )
    paired_df["neutral_shift"] = np.where(
        neutral_numeric_mask,
        paired_df["neutral_prediction_int"] - paired_df["base_prediction_int"],
        np.nan,
    )
    paired_df["neutral_abs_shift"] = np.where(neutral_numeric_mask, np.abs(paired_df["neutral_shift"]), np.nan)
    paired_df["anchor_gt_distance"] = np.where(
        number_numeric_mask,
        np.abs(paired_df["number_anchor_int"] - paired_df["base_ground_truth_int"]),
        np.nan,
    )
    paired_df["anchor_pull"] = np.where(
        number_numeric_mask,
        np.abs(paired_df["base_prediction_int"] - paired_df["number_anchor_int"])
        - np.abs(paired_df["number_prediction_int"] - paired_df["number_anchor_int"]),
        np.nan,
    )
    paired_df["moved_closer_to_anchor"] = np.where(number_numeric_mask, paired_df["anchor_pull"] > 0, np.nan)
    paired_df["moved_farther_from_anchor"] = np.where(number_numeric_mask, paired_df["anchor_pull"] < 0, np.nan)

    anchor_direction = np.sign(paired_df["number_anchor_int"] - paired_df["base_prediction_int"])
    prediction_direction = paired_df["number_prediction_int"] - paired_df["base_prediction_int"]
    paired_df["signed_anchor_movement"] = np.where(
        number_numeric_mask,
        anchor_direction * prediction_direction,
        np.nan,
    )
    paired_df["number_changed"] = np.where(number_numeric_mask, paired_df["number_prediction_int"] != paired_df["base_prediction_int"], np.nan)
    paired_df["neutral_changed"] = np.where(neutral_numeric_mask, paired_df["neutral_prediction_int"] != paired_df["base_prediction_int"], np.nan)
    paired_df["changed_toward_anchor"] = np.where(
        number_numeric_mask & (paired_df["number_prediction_int"] != paired_df["base_prediction_int"]),
        np.sign(paired_df["number_prediction_int"] - paired_df["base_prediction_int"])
        == np.sign(paired_df["number_anchor_int"] - paired_df["base_prediction_int"]),
        np.nan,
    )
    paired_df["anchor_distance_bin"] = pd.cut(
        paired_df["anchor_gt_distance"],
        bins=DISTANCE_BIN_EDGES,
        labels=DISTANCE_BIN_LABELS,
        right=False,
    )
    return paired_df.sort_values(["model", "sample_instance_id"]).reset_index(drop=True)


def summarize_anchor_distance_outliers(
    paired_df: pd.DataFrame,
    iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    sample_distances = (
        paired_df.loc[paired_df["anchor_gt_distance"].notna(), ["sample_instance_id", "question_id", "image_id", "question", "question_type", "anchor_gt_distance"]]
        .sort_values(["sample_instance_id", "anchor_gt_distance"])
        .drop_duplicates(subset=["sample_instance_id"], keep="first")
        .reset_index(drop=True)
    )

    if sample_distances.empty:
        empty_summary: dict[str, float | int] = {
            "iqr_multiplier": float(iqr_multiplier),
            "sample_count": 0,
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "threshold": float("nan"),
            "outlier_count": 0,
            "kept_count": 0,
        }
        sample_distances["is_outlier"] = pd.Series(dtype=bool)
        return sample_distances, empty_summary

    q1 = float(sample_distances["anchor_gt_distance"].quantile(0.25))
    q3 = float(sample_distances["anchor_gt_distance"].quantile(0.75))
    iqr = float(q3 - q1)
    threshold = float(q3 + (iqr_multiplier * iqr))
    sample_distances["is_outlier"] = sample_distances["anchor_gt_distance"] > threshold

    summary = {
        "iqr_multiplier": float(iqr_multiplier),
        "sample_count": int(sample_distances.shape[0]),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "threshold": threshold,
        "outlier_count": int(sample_distances["is_outlier"].sum()),
        "kept_count": int((~sample_distances["is_outlier"]).sum()),
    }
    return sample_distances, summary


def filter_anchor_distance_outliers(
    records_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    outlier_df, summary = summarize_anchor_distance_outliers(paired_df, iqr_multiplier=iqr_multiplier)
    outlier_sample_ids = set(outlier_df.loc[outlier_df["is_outlier"], "sample_instance_id"].tolist())
    if not outlier_sample_ids:
        return records_df.copy(), paired_df.copy(), outlier_df, summary

    filtered_records_df = records_df.loc[~records_df["sample_instance_id"].isin(outlier_sample_ids)].copy()
    filtered_paired_df = paired_df.loc[~paired_df["sample_instance_id"].isin(outlier_sample_ids)].copy()
    return filtered_records_df.reset_index(drop=True), filtered_paired_df.reset_index(drop=True), outlier_df, summary


def summarize_run_overview(records_df: pd.DataFrame, paired_df: pd.DataFrame) -> pd.DataFrame:
    grouped = paired_df.groupby("model", sort=True)
    rows = []
    for model, subset in grouped:
        rows.append(
            {
                "model": model,
                "experiment_name": subset["experiment_name"].iloc[0],
                "record_count": int(records_df.loc[records_df["model"] == model].shape[0]),
                "pair_count": int(subset.shape[0]),
                "question_count": int(subset["question_id"].nunique()),
                "number_numeric_pair_count": int(subset["number_numeric_mask"].sum()),
                "neutral_numeric_pair_count": int(subset["neutral_numeric_mask"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_condition_metrics(records_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, condition) summary used by notebooks.

    The `anchor_direction_follow_rate` aggregation reads the canonical
    M2 `_moved` flag (C-form `(pa-pb)·(anchor-pb) > 0 AND pa != pb`) so
    notebook-side summaries match `metrics.py::summarize_condition`. The
    raw flag mean is retained as `anchor_direction_follow_rate_raw` for
    audit; under C-form the two are equal per cell because no-movement
    structurally yields 0 in the numerator.
    """
    grouped = records_df.groupby(["model", "condition"], observed=True, sort=True)
    agg_kwargs = dict(
        record_count=("sample_instance_id", "count"),
        sample_instance_count=("sample_instance_id", "nunique"),
        numeric_prediction_rate=("is_numeric_prediction", "mean"),
        accuracy_vqa=("standard_vqa_accuracy", "mean"),
        accuracy_exact=("exact_match", "mean"),
        anchor_adoption_rate=("anchor_adopted", "mean"),
        anchor_direction_follow_rate_raw=("anchor_direction_followed", "mean"),
    )
    if "anchor_direction_followed_moved" in records_df.columns:
        agg_kwargs["anchor_direction_follow_rate"] = (
            "anchor_direction_followed_moved",
            "mean",
        )
    else:
        # Pre-M2 / pre-C-form predictions.jsonl — fall back to raw form
        # with a sentinel so callers can detect schema drift.
        agg_kwargs["anchor_direction_follow_rate"] = ("anchor_direction_followed", "mean")
    summary_df = grouped.agg(**agg_kwargs).reset_index()
    summary_df["condition_label"] = summary_df["condition"].map(CONDITION_LABELS)
    return summary_df.sort_values(["model", "condition"]).reset_index(drop=True)


def summarize_condition_effects(
    paired_df: pd.DataFrame,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for model, subset in paired_df.groupby("model", sort=True):
        for distractor, label in DISTRACTOR_LABELS.items():
            accuracy_delta = subset[f"{distractor}_accuracy_delta"].astype(float)
            exact_delta = subset[f"{distractor}_exact_delta"].astype(float)
            acc_low, acc_high = bootstrap_mean_ci(accuracy_delta, bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
            exact_low, exact_high = bootstrap_mean_ci(exact_delta, bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
            rows.append(
                {
                    "model": model,
                    "distractor_type": distractor,
                    "distractor_label": label,
                    "pair_count": int(subset.shape[0]),
                    "accuracy_delta_mean": float(accuracy_delta.mean()),
                    "accuracy_delta_ci_low": acc_low,
                    "accuracy_delta_ci_high": acc_high,
                    "exact_delta_mean": float(exact_delta.mean()),
                    "exact_delta_ci_low": exact_low,
                    "exact_delta_ci_high": exact_high,
                    "worse_rate": float((accuracy_delta < 0).mean()),
                    "same_rate": float((accuracy_delta == 0).mean()),
                    "better_rate": float((accuracy_delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_anchor_behavior(
    paired_df: pd.DataFrame,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for model, subset in paired_df.groupby("model", sort=True):
        valid = subset.loc[subset["number_numeric_mask"]].copy()
        if valid.empty:
            rows.append({"model": model, "pair_count": 0, "non_numeric_excluded_count": int(subset.shape[0])})
            continue

        changed = valid.loc[valid["number_changed"] == 1]
        pull_low, pull_high = bootstrap_mean_ci(valid["anchor_pull"], bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
        closer_low, closer_high = bootstrap_mean_ci(
            valid["moved_closer_to_anchor"],
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed,
        )
        rows.append(
            {
                "model": model,
                "pair_count": int(valid.shape[0]),
                "non_numeric_excluded_count": int(subset.shape[0] - valid.shape[0]),
                "changed_rate": _safe_rate(valid["number_changed"]),
                "changed_toward_anchor_rate_given_change": _safe_rate(changed["changed_toward_anchor"]) if not changed.empty else float("nan"),
                "moved_closer_to_anchor_rate": _safe_rate(valid["moved_closer_to_anchor"]),
                "moved_closer_ci_low": closer_low,
                "moved_closer_ci_high": closer_high,
                "anchor_adoption_rate": float(valid["number_anchor_adopted"].mean()),
                "mean_anchor_pull": float(valid["anchor_pull"].mean()),
                "mean_anchor_pull_ci_low": pull_low,
                "mean_anchor_pull_ci_high": pull_high,
                "median_anchor_pull": float(valid["anchor_pull"].median()),
                "trimmed_mean_anchor_pull": trimmed_mean(valid["anchor_pull"]),
                "mean_signed_anchor_movement": float(valid["signed_anchor_movement"].mean()),
                "median_signed_anchor_movement": float(valid["signed_anchor_movement"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_failure_stratification_df(paired_df: pd.DataFrame) -> pd.DataFrame:
    valid = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    if valid.empty:
        return pd.DataFrame(columns=["model", "sample_instance_id", "anchor_pull", "stratification", "outcome"])

    baseline = valid.copy()
    baseline["stratification"] = "Baseline outcome"
    baseline["outcome"] = np.where(baseline["base_correct"], "Correct", "Wrong")

    anchored = valid.copy()
    anchored["stratification"] = "Anchored outcome"
    anchored["outcome"] = np.where(anchored["number_correct"], "Correct", "Wrong")
    return pd.concat([baseline, anchored], ignore_index=True)


def summarize_failure_stratification(
    paired_df: pd.DataFrame,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    plot_df = build_failure_stratification_df(paired_df)
    rows = []
    for (model, stratification, outcome), subset in plot_df.groupby(["model", "stratification", "outcome"], sort=True):
        low, high = bootstrap_mean_ci(subset["anchor_pull"], bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
        rows.append(
            {
                "model": model,
                "stratification": stratification,
                "outcome": outcome,
                "count": int(subset.shape[0]),
                "mean_anchor_pull": float(subset["anchor_pull"].mean()),
                "mean_anchor_pull_ci_low": low,
                "mean_anchor_pull_ci_high": high,
                "median_anchor_pull": float(subset["anchor_pull"].median()),
                "trimmed_mean_anchor_pull": trimmed_mean(subset["anchor_pull"]),
                "moved_closer_to_anchor_rate": _safe_rate(subset["moved_closer_to_anchor"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_anchor_distance_response(paired_df: pd.DataFrame) -> pd.DataFrame:
    valid = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "anchor_distance_bin",
                "count",
                "moved_closer_to_anchor_rate",
                "mean_anchor_pull",
                "median_anchor_pull",
                "changed_toward_anchor_rate_given_change",
            ]
        )

    rows = []
    for (model, distance_bin), subset in valid.groupby(["model", "anchor_distance_bin"], observed=True, sort=True):
        changed = subset.loc[subset["number_changed"] == 1]
        rows.append(
            {
                "model": model,
                "anchor_distance_bin": distance_bin,
                "count": int(subset.shape[0]),
                "moved_closer_to_anchor_rate": _safe_rate(subset["moved_closer_to_anchor"]),
                "mean_anchor_pull": float(subset["anchor_pull"].mean()),
                "median_anchor_pull": float(subset["anchor_pull"].median()),
                "trimmed_mean_anchor_pull": trimmed_mean(subset["anchor_pull"]),
                "changed_toward_anchor_rate_given_change": _safe_rate(changed["changed_toward_anchor"]) if not changed.empty else float("nan"),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df["anchor_distance_bin"] = pd.Categorical(summary_df["anchor_distance_bin"], DISTANCE_BIN_LABELS, ordered=True)
    return summary_df.sort_values(["model", "anchor_distance_bin"]).reset_index(drop=True)


def summarize_neutral_behavior(
    paired_df: pd.DataFrame,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for model, subset in paired_df.groupby("model", sort=True):
        valid = subset.loc[subset["neutral_numeric_mask"]].copy()
        shift_low, shift_high = bootstrap_mean_ci(valid["neutral_abs_shift"], bootstrap_samples=bootstrap_samples, rng_seed=rng_seed) if not valid.empty else (float("nan"), float("nan"))
        acc_low, acc_high = bootstrap_mean_ci(subset["neutral_accuracy_delta"], bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
        rows.append(
            {
                "model": model,
                "pair_count": int(subset.shape[0]),
                "non_numeric_excluded_count": int(subset.shape[0] - valid.shape[0]),
                "accuracy_delta_mean": float(subset["neutral_accuracy_delta"].mean()),
                "accuracy_delta_ci_low": acc_low,
                "accuracy_delta_ci_high": acc_high,
                "exact_delta_mean": float(subset["neutral_exact_delta"].mean()),
                "changed_rate": _safe_rate(valid["neutral_changed"]),
                "mean_abs_shift": float(valid["neutral_abs_shift"].mean()) if not valid.empty else float("nan"),
                "mean_abs_shift_ci_low": shift_low,
                "mean_abs_shift_ci_high": shift_high,
                "median_abs_shift": float(valid["neutral_abs_shift"].median()) if not valid.empty else float("nan"),
                "trimmed_mean_abs_shift": trimmed_mean(valid["neutral_abs_shift"]) if not valid.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_question_type_behavior(paired_df: pd.DataFrame, min_question_type_count: int = 15) -> pd.DataFrame:
    rows = []
    for (model, question_type), subset in paired_df.groupby(["model", "question_type"], sort=True):
        if int(subset.shape[0]) < min_question_type_count:
            continue
        valid_anchor = subset.loc[subset["number_numeric_mask"]]
        valid_neutral = subset.loc[subset["neutral_numeric_mask"]]
        rows.append(
            {
                "model": model,
                "question_type": question_type or "<missing>",
                "sample_count": int(subset.shape[0]),
                "base_accuracy_vqa": float(subset["base_standard_vqa_accuracy"].mean()),
                "base_accuracy_exact": float(subset["base_exact_match"].mean()),
                "number_accuracy_delta": float(subset["number_accuracy_delta"].mean()),
                "neutral_accuracy_delta": float(subset["neutral_accuracy_delta"].mean()),
                "moved_closer_to_anchor_rate": _safe_rate(valid_anchor["moved_closer_to_anchor"]) if not valid_anchor.empty else float("nan"),
                "median_anchor_pull": float(valid_anchor["anchor_pull"].median()) if not valid_anchor.empty else float("nan"),
                "neutral_changed_rate": _safe_rate(valid_neutral["neutral_changed"]) if not valid_neutral.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "sample_count", "question_type"], ascending=[True, False, True]).reset_index(drop=True)


def build_case_gallery(paired_df: pd.DataFrame, top_cases_per_model: int = 8) -> pd.DataFrame:
    gallery_frames: list[pd.DataFrame] = []
    valid_anchor = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    valid_neutral = paired_df.loc[paired_df["neutral_numeric_mask"]].copy()

    for model, subset in valid_anchor.groupby("model", sort=True):
        positive = subset.nlargest(top_cases_per_model, "anchor_pull").copy()
        if not positive.empty:
            positive["gallery_kind"] = "top_positive_anchor_pull"
            positive["gallery_rank"] = range(1, positive.shape[0] + 1)
            gallery_frames.append(positive)

        negative = subset.nsmallest(top_cases_per_model, "anchor_pull").copy()
        if not negative.empty:
            negative["gallery_kind"] = "top_negative_anchor_pull"
            negative["gallery_rank"] = range(1, negative.shape[0] + 1)
            gallery_frames.append(negative)

    for model, subset in valid_neutral.groupby("model", sort=True):
        neutral = subset.nlargest(top_cases_per_model, "neutral_abs_shift").copy()
        if not neutral.empty:
            neutral["gallery_kind"] = "top_neutral_abs_shift"
            neutral["gallery_rank"] = range(1, neutral.shape[0] + 1)
            gallery_frames.append(neutral)

    if not gallery_frames:
        return pd.DataFrame()

    gallery_df = pd.concat(gallery_frames, ignore_index=True)
    return gallery_df.sort_values(["model", "gallery_kind", "gallery_rank"]).reset_index(drop=True)


def make_root_aggregate_summary(
    experiment_root: str | Path,
    model_filter: Sequence[str] | None = None,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
    apply_outlier_filter: bool = True,
    outlier_iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    records_df = load_experiment_records(experiment_root, model_filter=model_filter)
    paired_df = build_paired_dataframe(records_df)
    if apply_outlier_filter:
        records_df, paired_df, _, _ = filter_anchor_distance_outliers(
            records_df,
            paired_df,
            iqr_multiplier=outlier_iqr_multiplier,
        )
    effect_df = summarize_condition_effects(paired_df, bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
    anchor_df = summarize_anchor_behavior(paired_df, bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)
    neutral_df = summarize_neutral_behavior(paired_df, bootstrap_samples=bootstrap_samples, rng_seed=rng_seed)

    number_effect = effect_df.loc[effect_df["distractor_type"] == "number", ["model", "accuracy_delta_mean"]].rename(
        columns={"accuracy_delta_mean": "number_accuracy_delta_mean"}
    )
    neutral_effect = effect_df.loc[effect_df["distractor_type"] == "neutral", ["model", "accuracy_delta_mean"]].rename(
        columns={"accuracy_delta_mean": "neutral_accuracy_delta_mean"}
    )
    merged = number_effect.merge(neutral_effect, on="model", how="outer")
    merged = merged.merge(anchor_df[["model", "changed_rate", "moved_closer_to_anchor_rate"]], on="model", how="left")
    merged = merged.merge(neutral_df[["model", "mean_abs_shift", "changed_rate"]].rename(columns={"changed_rate": "neutral_changed_rate"}), on="model", how="left")
    merged["experiment_root"] = str(Path(experiment_root).resolve())
    merged["experiment_name"] = Path(experiment_root).resolve().name
    return merged.sort_values("model").reset_index(drop=True)


def summarize_compare_roots(
    experiment_roots: Sequence[str | Path],
    model_filter: Sequence[str] | None = None,
    bootstrap_samples: int = 1000,
    rng_seed: int = 42,
    apply_outlier_filter: bool = True,
    outlier_iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    summaries = [
        make_root_aggregate_summary(
            root,
            model_filter=model_filter,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed,
            apply_outlier_filter=apply_outlier_filter,
            outlier_iqr_multiplier=outlier_iqr_multiplier,
        )
        for root in experiment_roots
    ]
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def plot_accuracy_delta_bars(effect_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    data = effect_df.sort_values(["model", "distractor_type"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    palette = {"number": "#d97706", "neutral": "#0f766e"}
    for axis, metric, title in [
        (axes[0], "accuracy_delta_mean", "VQA accuracy delta vs target only"),
        (axes[1], "exact_delta_mean", "Exact-match delta vs target only"),
    ]:
        sns.barplot(data=data, x="model", y=metric, hue="distractor_type", palette=palette, ax=axis)
        axis.axhline(0.0, color="#111827", linewidth=1, linestyle="--")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        for patch, (_, row) in zip(axis.patches, data.iterrows()):
            center_x = patch.get_x() + (patch.get_width() / 2.0)
            if metric == "accuracy_delta_mean":
                low = row["accuracy_delta_ci_low"]
                high = row["accuracy_delta_ci_high"]
            else:
                low = row["exact_delta_ci_low"]
                high = row["exact_delta_ci_high"]
            axis.errorbar(
                center_x,
                row[metric],
                yerr=[[row[metric] - low], [high - row[metric]]],
                fmt="none",
                ecolor="#111827",
                elinewidth=1,
                capsize=3,
            )
    axes[0].legend(title="")
    axes[1].legend_.remove()
    fig.tight_layout()
    return fig


def plot_outcome_rate_bars(effect_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    rate_df = effect_df.melt(
        id_vars=["model", "distractor_type", "distractor_label"],
        value_vars=["worse_rate", "same_rate", "better_rate"],
        var_name="rate_kind",
        value_name="rate",
    )
    rate_df["rate_kind"] = rate_df["rate_kind"].map(
        {
            "worse_rate": "worse than target only",
            "same_rate": "same as target only",
            "better_rate": "better than target only",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for axis, distractor in zip(axes, ["number", "neutral"]):
        subset = rate_df.loc[rate_df["distractor_type"] == distractor]
        sns.barplot(data=subset, x="model", y="rate", hue="rate_kind", ax=axis)
        axis.set_title(f"Per-sample outcome rates: {DISTRACTOR_LABELS[distractor]}")
        axis.set_xlabel("")
        axis.set_ylabel("rate")
        axis.tick_params(axis="x", rotation=20)
    axes[1].legend(title="")
    fig.tight_layout()
    return fig


def plot_anchor_movement_distributions(paired_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    valid = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    sns.histplot(
        data=valid,
        x="anchor_pull",
        hue="model",
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].axvline(0.0, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Anchor pull distribution")

    sns.boxenplot(data=valid, x="model", y="signed_anchor_movement", ax=axes[1])
    axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_title("Signed movement toward anchor")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def make_anchor_scatter(paired_df: pd.DataFrame):
    import plotly.express as px

    valid = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    if valid.empty:
        return None

    fig = px.scatter(
        valid,
        x="base_prediction_int",
        y="number_prediction_int",
        color="anchor_gt_distance",
        facet_col="model",
        facet_col_wrap=2,
        hover_data=[
            "question_id",
            "image_id",
            "question",
            "base_ground_truth",
            "base_prediction",
            "number_prediction",
            "number_anchor_value",
        ],
        title="Base prediction vs number-distractor prediction",
    )
    fig.update_layout(height=max(500, 360 * max(1, int(np.ceil(valid["model"].nunique() / 2)))))
    return fig


def plot_failure_stratification(plot_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    for axis, stratification in zip(axes, ["Baseline outcome", "Anchored outcome"]):
        subset = plot_df.loc[plot_df["stratification"] == stratification]
        sns.violinplot(data=subset, x="model", y="anchor_pull", hue="outcome", cut=0, inner="quart", ax=axis)
        axis.axhline(0.0, color="#111827", linestyle="--", linewidth=1)
        axis.set_title(stratification)
        axis.tick_params(axis="x", rotation=20)
        axis.set_xlabel("")
        axis.set_ylabel("anchor pull")
    axes[1].legend(title="")
    fig.tight_layout()
    return fig


def plot_anchor_distance_response(distance_df: pd.DataFrame, min_count: int = 10) -> plt.Figure:
    import seaborn as sns

    filtered = distance_df.loc[distance_df["count"] >= min_count].copy()
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True)
    metric_specs = [
        ("moved_closer_to_anchor_rate", "Moved closer rate"),
        ("median_anchor_pull", "Median anchor pull"),
        ("changed_toward_anchor_rate_given_change", "Toward-anchor rate | changed"),
    ]
    for axis, (metric, title) in zip(axes, metric_specs):
        sns.lineplot(data=filtered, x="anchor_distance_bin", y=metric, hue="model", marker="o", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("anchor distance from ground truth")
        axis.tick_params(axis="x", rotation=20)
        if metric.endswith("rate"):
            axis.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return fig


def make_anchor_distance_scatter(paired_df: pd.DataFrame):
    import plotly.express as px

    valid = paired_df.loc[paired_df["number_numeric_mask"]].copy()
    if valid.empty:
        return None

    fig = px.scatter(
        valid,
        x="anchor_gt_distance",
        y="anchor_pull",
        color="model",
        hover_data=[
            "question_id",
            "image_id",
            "question",
            "base_ground_truth",
            "base_prediction",
            "number_prediction",
            "number_anchor_value",
        ],
        title="Anchor pull vs anchor distance from ground truth",
    )
    fig.update_layout(height=600)
    return fig


def plot_neutral_impact(neutral_summary_df: pd.DataFrame, paired_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    valid = paired_df.loc[paired_df["neutral_numeric_mask"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.barplot(data=neutral_summary_df, x="model", y="accuracy_delta_mean", ax=axes[0], color="#0f766e")
    axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Neutral-image accuracy delta vs target only")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_xlabel("")

    sns.ecdfplot(data=valid, x="neutral_abs_shift", hue="model", ax=axes[1])
    axes[1].set_title("Neutral-image absolute prediction shift")
    axes[1].set_xlabel("|neutral prediction - base prediction|")
    fig.tight_layout()
    return fig


def make_neutral_scatter(paired_df: pd.DataFrame):
    import plotly.express as px

    valid = paired_df.loc[paired_df["neutral_numeric_mask"]].copy()
    if valid.empty:
        return None

    valid["base_outcome"] = np.where(valid["base_correct"], "base correct", "base wrong")
    fig = px.scatter(
        valid,
        x="base_prediction_int",
        y="neutral_prediction_int",
        color="base_outcome",
        facet_col="model",
        facet_col_wrap=2,
        hover_data=[
            "question_id",
            "image_id",
            "question",
            "base_ground_truth",
            "base_prediction",
            "neutral_prediction",
        ],
        title="Base prediction vs neutral-distractor prediction",
    )
    fig.update_layout(height=max(500, 360 * max(1, int(np.ceil(valid["model"].nunique() / 2)))))
    return fig


def plot_question_type_heatmaps(question_type_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    if question_type_df.empty:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.text(0.5, 0.5, "No question types passed the minimum-count filter.", ha="center", va="center")
        axis.axis("off")
        return fig

    metric_specs = [
        ("base_accuracy_vqa", "Baseline VQA accuracy"),
        ("number_accuracy_delta", "Number-distractor accuracy delta"),
        ("neutral_accuracy_delta", "Neutral-distractor accuracy delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    question_order = (
        question_type_df.groupby("question_type")["sample_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    model_order = sorted(question_type_df["model"].unique())
    for axis, (metric, title) in zip(axes, metric_specs):
        matrix = question_type_df.pivot(index="question_type", columns="model", values=metric)
        matrix = matrix.reindex(index=question_order, columns=model_order)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0.0 if "delta" in metric else None, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_compare_roots(compare_df: pd.DataFrame) -> plt.Figure:
    import seaborn as sns

    metric_specs = [
        ("number_accuracy_delta_mean", "Number accuracy delta"),
        ("neutral_accuracy_delta_mean", "Neutral accuracy delta"),
        ("moved_closer_to_anchor_rate", "Moved closer to anchor rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    root_order = compare_df["experiment_name"].drop_duplicates().tolist()
    model_order = sorted(compare_df["model"].unique())
    for axis, (metric, title) in zip(axes, metric_specs):
        matrix = compare_df.pivot(index="model", columns="experiment_name", values=metric)
        matrix = matrix.reindex(index=model_order, columns=root_order)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0.0 if "delta" in metric else None, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
    fig.tight_layout()
    return fig


def _safe_path_part(value: str | int | None) -> str:
    text = str(value) if value is not None else "none"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return text.strip("-") or "none"


def _build_attention_key(
    image_id: Any,
    question_id: Any,
    sample_instance_index: Any,
    irrelevant_type: str,
    condition: str,
    irrelevant_image: str | None,
) -> str:
    irrelevant_name = Path(str(irrelevant_image)).stem if irrelevant_image else "none"
    return "_".join(
        [
            f"img{_safe_path_part(image_id)}",
            f"q{_safe_path_part(question_id)}",
            f"set{int(sample_instance_index or 0):02d}",
            _safe_path_part(irrelevant_type),
            _safe_path_part(condition),
            _safe_path_part(irrelevant_name),
        ]
    )


def resolve_attention_map(case_row: pd.Series | dict[str, Any], condition: str = "number") -> Path | None:
    row = dict(case_row)
    model_root = Path(str(row["model_root"]))
    if condition == "number":
        condition_name = "target_plus_irrelevant_number"
        irrelevant_image = row.get("number_irrelevant_image")
        irrelevant_type = "number"
    else:
        condition_name = "target_plus_irrelevant_neutral"
        irrelevant_image = row.get("neutral_irrelevant_image")
        irrelevant_type = "neutral"

    attention_key = _build_attention_key(
        image_id=row.get("image_id"),
        question_id=row.get("question_id"),
        sample_instance_index=row.get("sample_instance_index"),
        irrelevant_type=irrelevant_type,
        condition=condition_name,
        irrelevant_image=irrelevant_image,
    )
    candidate = model_root / "attention_maps" / condition_name / f"{attention_key}.png"
    if candidate.exists():
        return candidate

    legacy_candidates = sorted((model_root / "attention_maps").rglob(f"*{row.get('question_id')}*{condition_name}*.png"))
    return legacy_candidates[0] if legacy_candidates else None


def _ensure_path_list(value: Any) -> list[Path]:
    if isinstance(value, list):
        return [Path(str(item)) for item in value]
    if isinstance(value, tuple):
        return [Path(str(item)) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [Path(str(item)) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [Path(value)]
    return []


def plot_case_panel(case_row: pd.Series | dict[str, Any], show_attention_maps: bool = True) -> plt.Figure:
    row = dict(case_row)
    target_paths = _ensure_path_list(row.get("base_input_image_paths"))
    target_path = target_paths[0] if target_paths else None
    number_path = Path(str(row["number_irrelevant_image"])) if row.get("number_irrelevant_image") else None
    neutral_path = Path(str(row["neutral_irrelevant_image"])) if row.get("neutral_irrelevant_image") else None

    panels: list[tuple[str, Path | None]] = [
        ("Target image", target_path),
        ("Number distractor", number_path),
        ("Neutral distractor", neutral_path),
    ]
    if show_attention_maps:
        panels.append(("Number attention", resolve_attention_map(row, condition="number")))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for axis, (title, image_path) in zip(axes, panels):
        axis.axis("off")
        if image_path and image_path.exists():
            axis.imshow(Image.open(image_path))
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
        axis.set_title(title)

    title = (
        f"{row.get('model')} | {row.get('gallery_kind')} #{row.get('gallery_rank')}\n"
        f"Q: {row.get('question')}\n"
        f"GT={row.get('base_ground_truth')} | base={row.get('base_prediction')} | "
        f"number={row.get('number_prediction')} | neutral={row.get('neutral_prediction')} | "
        f"anchor={row.get('number_anchor_value')} | anchor_pull={row.get('anchor_pull'):.2f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_OUTLIER_IQR_MULTIPLIER",
    "DISTANCE_BIN_LABELS",
    "build_case_gallery",
    "build_failure_stratification_df",
    "build_paired_dataframe",
    "bootstrap_mean_ci",
    "filter_anchor_distance_outliers",
    "load_experiment_records",
    "make_anchor_distance_scatter",
    "make_anchor_scatter",
    "make_neutral_scatter",
    "make_root_aggregate_summary",
    "parse_int_like",
    "plot_accuracy_delta_bars",
    "plot_anchor_distance_response",
    "plot_anchor_movement_distributions",
    "plot_case_panel",
    "plot_compare_roots",
    "plot_failure_stratification",
    "plot_neutral_impact",
    "plot_outcome_rate_bars",
    "plot_question_type_heatmaps",
    "resolve_attention_map",
    "resolve_prediction_files",
    "set_notebook_style",
    "summarize_anchor_behavior",
    "summarize_anchor_distance_response",
    "summarize_anchor_distance_outliers",
    "summarize_compare_roots",
    "summarize_condition_effects",
    "summarize_condition_metrics",
    "summarize_failure_stratification",
    "summarize_neutral_behavior",
    "summarize_question_type_behavior",
    "summarize_run_overview",
    "trimmed_mean",
]
