from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from vlm_anchor.utils import dump_json, ensure_dir, extract_first_number, normalize_numeric_text


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
CONDITION_COLORS = {
    "target_only": "#1f3c88",
    "target_plus_irrelevant_number": "#d97706",
    "target_plus_irrelevant_neutral": "#0f766e",
}


def save_experiment_analysis_figures(records: list[dict], output_dir: str | Path) -> None:
    if not records:
        return

    output_dir = ensure_dir(output_dir)
    model_names = sorted(set(r["model"] for r in records))
    analysis = {model_name: _compute_model_analysis([r for r in records if r["model"] == model_name]) for model_name in model_names}

    dump_json({"models": analysis}, output_dir / "summary.json")
    _plot_condition_metrics(analysis, output_dir / "condition_metrics.png")
    _plot_anchoring_effects(analysis, output_dir / "anchoring_effects.png")


def _compute_model_analysis(records: list[dict]) -> dict[str, Any]:
    condition_metrics: dict[str, dict[str, float | int | None]] = {}
    available_conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in records)]

    for condition in available_conditions:
        subset = [r for r in records if r["condition"] == condition]
        abs_errors = [_numeric_abs_error(r["prediction"], r["ground_truth"]) for r in subset]
        abs_errors = [err for err in abs_errors if err is not None]
        distances = [r["numeric_distance_to_anchor"] for r in subset if r["numeric_distance_to_anchor"] is not None]
        condition_metrics[condition] = {
            "count": len(subset),
            "accuracy_vqa": mean(r["standard_vqa_accuracy"] for r in subset),
            "accuracy_exact": mean(r["exact_match"] for r in subset),
            "mean_abs_error": mean(abs_errors) if abs_errors else None,
            "anchor_adoption_rate": mean(r["anchor_adopted"] for r in subset),
            "anchor_direction_follow_rate": mean(r["anchor_direction_followed"] for r in subset),
            "median_distance_to_anchor": median(distances) if distances else None,
        }

    paired = _compute_paired_effects(records)
    return {
        "num_records": len(records),
        "conditions": condition_metrics,
        "paired_effects": paired,
    }


def _compute_paired_effects(records: list[dict]) -> dict[str, float | int | None]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        group_key = str(row.get("sample_instance_id") or row["question_id"])
        grouped.setdefault(group_key, {})[row["condition"]] = row

    accuracy_delta_number: list[float] = []
    accuracy_delta_neutral: list[float] = []
    exact_delta_number: list[float] = []
    exact_delta_neutral: list[float] = []
    abs_error_delta_number: list[float] = []
    abs_error_delta_neutral: list[float] = []
    anchor_pull_values: list[float] = []
    moved_closer_flags: list[int] = []

    for condition_rows in grouped.values():
        base = condition_rows.get("target_only")
        number = condition_rows.get("target_plus_irrelevant_number")
        neutral = condition_rows.get("target_plus_irrelevant_neutral")
        if not base or not number or not neutral:
            continue

        accuracy_delta_number.append(number["standard_vqa_accuracy"] - base["standard_vqa_accuracy"])
        accuracy_delta_neutral.append(neutral["standard_vqa_accuracy"] - base["standard_vqa_accuracy"])
        exact_delta_number.append(number["exact_match"] - base["exact_match"])
        exact_delta_neutral.append(neutral["exact_match"] - base["exact_match"])

        base_error = _numeric_abs_error(base["prediction"], base["ground_truth"])
        number_error = _numeric_abs_error(number["prediction"], number["ground_truth"])
        neutral_error = _numeric_abs_error(neutral["prediction"], neutral["ground_truth"])
        if base_error is not None and number_error is not None:
            abs_error_delta_number.append(number_error - base_error)
        if base_error is not None and neutral_error is not None:
            abs_error_delta_neutral.append(neutral_error - base_error)

        base_pred = _to_numeric(base["prediction"])
        number_pred = _to_numeric(number["prediction"])
        anchor_value = _to_numeric(number["anchor_value"])
        if base_pred is None or number_pred is None or anchor_value is None:
            continue

        anchor_pull = abs(base_pred - anchor_value) - abs(number_pred - anchor_value)
        anchor_pull_values.append(anchor_pull)
        moved_closer_flags.append(int(anchor_pull > 0))

    return {
        "pair_count": len(accuracy_delta_number),
        "accuracy_delta_number_vs_target_only": _mean_or_none(accuracy_delta_number),
        "accuracy_delta_neutral_vs_target_only": _mean_or_none(accuracy_delta_neutral),
        "exact_delta_number_vs_target_only": _mean_or_none(exact_delta_number),
        "exact_delta_neutral_vs_target_only": _mean_or_none(exact_delta_neutral),
        "abs_error_delta_number_vs_target_only": _mean_or_none(abs_error_delta_number),
        "abs_error_delta_neutral_vs_target_only": _mean_or_none(abs_error_delta_neutral),
        "moved_closer_to_anchor_rate": _mean_or_none(moved_closer_flags),
        "mean_anchor_pull": _mean_or_none(anchor_pull_values),
    }


def _plot_condition_metrics(analysis: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    model_names = list(analysis)
    fig, axes = plt.subplots(len(model_names), 3, figsize=(18, 5 * len(model_names)), squeeze=False)
    metric_specs = [
        ("accuracy_vqa", "VQA accuracy", (0.0, 1.05)),
        ("accuracy_exact", "Exact-match accuracy", (0.0, 1.05)),
        ("mean_abs_error", "Mean absolute error", None),
    ]

    for row_idx, model_name in enumerate(model_names):
        condition_metrics = analysis[model_name]["conditions"]
        conditions = [c for c in CONDITION_ORDER if c in condition_metrics]
        labels = [CONDITION_LABELS.get(c, c) for c in conditions]
        colors = [CONDITION_COLORS.get(c, "#4b5563") for c in conditions]

        for col_idx, (metric_key, title, ylim) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = [condition_metrics[c].get(metric_key) for c in conditions]
            positions = np.arange(len(labels))
            bars = ax.bar(positions, _plot_values(values), color=colors)
            ax.set_title(title)
            ax.set_xticks(positions, labels, rotation=15, ha="right")
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0.0)
            if col_idx == 0:
                ax.set_ylabel(model_name)
            _annotate_bars(ax, bars, values)

    fig.suptitle("Condition-level behavior by model", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_anchoring_effects(analysis: dict[str, dict[str, Any]], output_path: str | Path) -> None:
    model_names = list(analysis)
    fig, axes = plt.subplots(len(model_names), 4, figsize=(22, 5 * len(model_names)), squeeze=False)

    for row_idx, model_name in enumerate(model_names):
        paired = analysis[model_name]["paired_effects"]
        condition_metrics = analysis[model_name]["conditions"]

        delta_specs = [
            (
                "Accuracy delta vs target only",
                [
                    paired.get("accuracy_delta_number_vs_target_only"),
                    paired.get("accuracy_delta_neutral_vs_target_only"),
                ],
            ),
            (
                "Exact-match delta vs target only",
                [
                    paired.get("exact_delta_number_vs_target_only"),
                    paired.get("exact_delta_neutral_vs_target_only"),
                ],
            ),
            (
                "MAE delta vs target only",
                [
                    paired.get("abs_error_delta_number_vs_target_only"),
                    paired.get("abs_error_delta_neutral_vs_target_only"),
                ],
            ),
        ]
        delta_labels = ["+ irrelevant number", "+ irrelevant neutral"]
        delta_colors = [CONDITION_COLORS["target_plus_irrelevant_number"], CONDITION_COLORS["target_plus_irrelevant_neutral"]]

        for col_idx, (title, values) in enumerate(delta_specs):
            ax = axes[row_idx, col_idx]
            positions = np.arange(len(delta_labels))
            bars = ax.bar(positions, _plot_values(values), color=delta_colors)
            ax.axhline(0.0, color="#374151", linewidth=1, linestyle="--")
            ax.set_title(title)
            ax.set_xticks(positions, delta_labels, rotation=15, ha="right")
            if col_idx == 0:
                ax.set_ylabel(model_name)
            _annotate_bars(ax, bars, values)

        anchor_ax = axes[row_idx, 3]
        anchor_values = [
            condition_metrics.get("target_plus_irrelevant_number", {}).get("anchor_adoption_rate"),
            condition_metrics.get("target_plus_irrelevant_number", {}).get("anchor_direction_follow_rate"),
            paired.get("moved_closer_to_anchor_rate"),
        ]
        anchor_labels = ["anchor adopted", "direction followed", "moved closer"]
        positions = np.arange(len(anchor_labels))
        anchor_bars = anchor_ax.bar(
            positions,
            _plot_values(anchor_values),
            color=["#b91c1c", "#d97706", "#0f766e"],
        )
        anchor_ax.set_ylim(0.0, 1.05)
        anchor_ax.set_title("Anchoring impact rates")
        anchor_ax.set_xticks(positions, anchor_labels, rotation=15, ha="right")
        mean_anchor_pull = paired.get("mean_anchor_pull")
        if mean_anchor_pull is not None:
            anchor_ax.text(
                0.02,
                0.98,
                f"mean anchor pull: {mean_anchor_pull:.2f}",
                transform=anchor_ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#d1d5db"},
            )
        _annotate_bars(anchor_ax, anchor_bars, anchor_values)

    fig.suptitle("Anchoring effect relative to target-only baseline", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars: Any, values: list[float | int | None]) -> None:
    for bar, value in zip(bars, values):
        if value is None or not np.isfinite(value):
            continue
        offset = 0.02 if value >= 0 else -0.02
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=9,
        )


def _numeric_abs_error(prediction: str | None, ground_truth: str | None) -> float | None:
    pred = _to_numeric(prediction)
    gt = _to_numeric(ground_truth)
    if pred is None or gt is None:
        return None
    return abs(pred - gt)


def _plot_values(values: list[float | int | None]) -> list[float]:
    return [np.nan if value is None else float(value) for value in values]


def _to_numeric(value: Any) -> float | None:
    normalized = normalize_numeric_text(extract_first_number(value))
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _mean_or_none(values: list[float | int]) -> float | None:
    return mean(values) if values else None
