"""Per-sample evaluation and condition-level summarisation under M2 metrics.

Canonical definitions (`docs/insights/M2-metric-definition-evidence.md`):

    adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
    direction_follow_rate = #( (pa-pb)·(anchor-pb) > 0  AND  pa != pb )
                            / #(numeric pair AND anchor present)
    exact_match           = #(pa == gt) / #(numeric pair)
    anchor_effect_M       = M(a-arm) - M(d-arm)

The ``direction_follow_rate`` numerator measures whether ``pa`` (the
anchor-condition prediction) shifted **from the baseline ``pb`` toward the
anchor stimulus**. Using ``pb`` (not ``gt``) as the reference makes the
metric depend only on model outputs and the anchor draw — a direct measure
of anchor pull, robust to per-question stimulus variability.

Per-row flags persisted to ``predictions.jsonl``:

    anchor_adopted                       — paired numerator (M1)
    anchor_direction_followed            — DF_raw, sign-based
    anchor_direction_followed_moved      — DF_moved (= DF_raw AND pa_ne_pb)
    pred_b_equal_anchor                  — pb_eq_a
    pred_diff_from_base                  — pa_ne_pb
    numeric_distance_to_anchor

Aggregated rates use the new flags; legacy raw rates remain available
under ``*_marginal`` / ``*_raw`` names for back-compat audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from vlm_anchor.utils import extract_first_number, normalize_numeric_text


@dataclass
class VQASampleEval:
    prediction: str
    normalized_prediction: str
    ground_truth: str
    normalized_ground_truth: str
    standard_vqa_accuracy: float
    exact_match: int
    anchor_value: str | None
    anchor_adopted: int
    anchor_direction_followed: int
    anchor_direction_followed_moved: int
    pred_b_equal_anchor: int
    pred_diff_from_base: int
    numeric_distance_to_anchor: float | None



def standard_vqa_accuracy(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_numeric_text(extract_first_number(prediction))
    normalized_answers = [normalize_numeric_text(extract_first_number(a)) for a in answers]
    matches = sum(1 for a in normalized_answers if a == pred)
    return min(1.0, matches / 3.0)


def _is_int_str(s: str | None) -> bool:
    return bool(s) and s.lstrip("-").isdigit()


def evaluate_sample(
    prediction: str,
    gt_answer: str,
    all_answers: list[str],
    anchor_value: str | None,
    *,
    base_prediction: str | None,
) -> VQASampleEval:
    """Per-sample evaluation under M2 canonical metrics.

    Per-row flags use canonical naming: ``pa = pred_a`` (this row's parsed
    prediction), ``pb = pred_b`` (the matching base / target_only prediction
    threaded in via ``base_prediction``), ``anchor`` (anchor value rendered
    in the irrelevant image), ``gt`` (ground truth).

    Adopt numerator (paired, M1):
        anchor_adopted = (pa == anchor) AND (pb != anchor)

    Direction-follow numerators (C-form, gt-free):
        anchor_direction_followed       = (pa - pb) * (anchor - pb) > 0
        anchor_direction_followed_moved = anchor_direction_followed AND (pa != pb)

    Denominator filters live in ``summarize_condition``; this function only
    populates the per-row indicators.
    """
    pa = normalize_numeric_text(extract_first_number(prediction))
    gt = normalize_numeric_text(extract_first_number(gt_answer))
    acc = standard_vqa_accuracy(pa, all_answers)
    exact = int(pa == gt)

    anchor = normalize_numeric_text(str(anchor_value)) if anchor_value is not None else None

    pb_raw = base_prediction if base_prediction is not None else ""
    pb = normalize_numeric_text(extract_first_number(pb_raw))
    pb_is_int = _is_int_str(pb)

    pa_eq_anchor = bool(anchor) and pa == anchor
    pb_eq_anchor = bool(anchor) and pb_is_int and pb == anchor

    if anchor is None or not pa_eq_anchor or not pb_is_int:
        # Cannot establish baseline → conservative zero.
        anchor_adopted = 0
    else:
        anchor_adopted = int(not pb_eq_anchor)

    pred_diff_from_base = int(pb_is_int and pa != pb)

    direction_followed = 0
    direction_followed_moved = 0
    distance: float | None = None
    if (
        anchor
        and _is_int_str(pa)
        and _is_int_str(anchor)
    ):
        pa_i, anchor_i = int(pa), int(anchor)
        # C-form: pa shifted from pb toward the anchor side of pb (gt-free).
        if pb_is_int and (pa_i - int(pb)) * (anchor_i - int(pb)) > 0:
            direction_followed = 1
        if pb_is_int and direction_followed and int(pb) != pa_i:
            direction_followed_moved = 1
        distance = float(abs(pa_i - anchor_i))

    return VQASampleEval(
        prediction=prediction,
        normalized_prediction=pa,
        ground_truth=gt_answer,
        normalized_ground_truth=gt,
        standard_vqa_accuracy=acc,
        exact_match=exact,
        anchor_value=anchor,
        anchor_adopted=anchor_adopted,
        anchor_direction_followed=direction_followed,
        anchor_direction_followed_moved=direction_followed_moved,
        pred_b_equal_anchor=int(pb_eq_anchor),
        pred_diff_from_base=pred_diff_from_base,
        numeric_distance_to_anchor=distance,
    )



def summarize_condition(records: list[dict], condition_name: str) -> dict:
    """Per-condition aggregate statistics under M2 metrics.

    Headline rates use M2 denominators:

        anchor_adoption_rate       =  Σ anchor_adopted                          /  Σ (pred_b != anchor)
        anchor_direction_follow_rate = Σ anchor_direction_followed_moved        /  Σ (numeric pair AND anchor present)

    Legacy / pre-M2 forms also reported for audit:

        anchor_adoption_rate_marginal      =  Σ anchor_adopted                / count_subset (D_all)
        anchor_direction_follow_rate_raw   =  Σ anchor_direction_followed     / Σ (numeric pair AND anchor present)
    """
    subset = [r for r in records if r["condition"] == condition_name]
    if not subset:
        return {"condition": condition_name, "count": 0}

    n = len(subset)

    def _flag(r: dict, key: str) -> int:
        v = r.get(key)
        return int(v) if v is not None else 0

    n_pb_ne_a = sum(1 for r in subset if not _flag(r, "pred_b_equal_anchor"))

    # numeric_distance_to_anchor is set IFF pa and anchor are both int-parseable
    # AND anchor is present (see evaluate_sample line 117); gt does NOT gate
    # the predicate. Reuse it as the "numeric pair AND anchor present" denominator
    # for direction_follow_rate.
    n_numeric_with_anchor = sum(1 for r in subset if r.get("numeric_distance_to_anchor") is not None)

    sum_adopt = sum(_flag(r, "anchor_adopted") for r in subset)
    sum_df_raw = sum(_flag(r, "anchor_direction_followed") for r in subset)
    sum_df_moved = sum(_flag(r, "anchor_direction_followed_moved") for r in subset)

    summary = {
        "condition": condition_name,
        "count": n,
        "accuracy_vqa": mean(r["standard_vqa_accuracy"] for r in subset),
        "accuracy_exact": mean(r["exact_match"] for r in subset),
        "n_pb_ne_anchor_denominator": n_pb_ne_a,
        "n_numeric_anchor_denominator": n_numeric_with_anchor,
        "anchor_adoption_rate": (sum_adopt / n_pb_ne_a) if n_pb_ne_a else None,
        "anchor_adoption_rate_marginal": sum_adopt / n,
        "anchor_direction_follow_rate": (sum_df_moved / n_numeric_with_anchor) if n_numeric_with_anchor else None,
        "anchor_direction_follow_rate_raw": (sum_df_raw / n_numeric_with_anchor) if n_numeric_with_anchor else None,
        "mean_distance_to_anchor": mean(
            r["numeric_distance_to_anchor"] for r in subset if r["numeric_distance_to_anchor"] is not None
        ) if any(r["numeric_distance_to_anchor"] is not None for r in subset) else None,
    }
    return summary



def summarize_experiment(records: list[dict], base_condition: str = "target_only") -> dict:
    conditions = sorted(set(r["condition"] for r in records))
    summary = {c: summarize_condition(records, c) for c in conditions}
    base_acc = summary.get(base_condition, {}).get("accuracy_vqa")
    base_adopt = summary.get(base_condition, {}).get("anchor_adoption_rate") or 0.0
    base_df = summary.get(base_condition, {}).get("anchor_direction_follow_rate") or 0.0

    if base_acc is not None:
        for c in conditions:
            cond = summary[c]
            if "accuracy_vqa" in cond:
                cond["accuracy_drop_vs_target_only"] = base_acc - cond["accuracy_vqa"]
                cond_adopt = cond.get("anchor_adoption_rate") or 0.0
                cond_df = cond.get("anchor_direction_follow_rate") or 0.0
                cond["anchor_susceptibility_gap_vs_target_only"] = cond_adopt - base_adopt
                cond["direction_follow_gap_vs_target_only"] = cond_df - base_df
    return summary
