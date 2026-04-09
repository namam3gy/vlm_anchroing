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
    numeric_distance_to_anchor: float | None



def standard_vqa_accuracy(prediction: str, answers: Iterable[str]) -> float:
    pred = normalize_numeric_text(extract_first_number(prediction))
    normalized_answers = [normalize_numeric_text(extract_first_number(a)) for a in answers]
    matches = sum(1 for a in normalized_answers if a == pred)
    return min(1.0, matches / 3.0)



def evaluate_sample(prediction: str, gt_answer: str, all_answers: list[str], anchor_value: str | None) -> VQASampleEval:
    pred = normalize_numeric_text(extract_first_number(prediction))
    gt = normalize_numeric_text(extract_first_number(gt_answer))
    acc = standard_vqa_accuracy(pred, all_answers)
    exact = int(pred == gt)

    anchor_val = normalize_numeric_text(str(anchor_value)) if anchor_value is not None else None
    anchor_adopted = int(bool(anchor_val) and pred == anchor_val)

    direction_followed = 0
    distance = None
    if anchor_val and pred and gt and pred.lstrip("-").isdigit() and gt.lstrip("-").isdigit() and anchor_val.lstrip("-").isdigit():
        pred_int, gt_int, anchor_int = int(pred), int(gt), int(anchor_val)
        direction_followed = int((pred_int - gt_int) * (anchor_int - gt_int) > 0)
        distance = abs(pred_int - anchor_int)

    return VQASampleEval(
        prediction=prediction,
        normalized_prediction=pred,
        ground_truth=gt_answer,
        normalized_ground_truth=gt,
        standard_vqa_accuracy=acc,
        exact_match=exact,
        anchor_value=anchor_val,
        anchor_adopted=anchor_adopted,
        anchor_direction_followed=direction_followed,
        numeric_distance_to_anchor=distance,
    )



def summarize_condition(records: list[dict], condition_name: str) -> dict:
    subset = [r for r in records if r["condition"] == condition_name]
    if not subset:
        return {
            "condition": condition_name,
            "count": 0,
        }
    return {
        "condition": condition_name,
        "count": len(subset),
        "accuracy_vqa": mean(r["standard_vqa_accuracy"] for r in subset),
        "accuracy_exact": mean(r["exact_match"] for r in subset),
        "anchor_adoption_rate": mean(r["anchor_adopted"] for r in subset),
        "anchor_direction_follow_rate": mean(r["anchor_direction_followed"] for r in subset),
        "mean_distance_to_anchor": mean(
            r["numeric_distance_to_anchor"] for r in subset if r["numeric_distance_to_anchor"] is not None
        ) if any(r["numeric_distance_to_anchor"] is not None for r in subset) else None,
    }



def summarize_experiment(records: list[dict], base_condition: str = "target_only") -> dict:
    conditions = sorted(set(r["condition"] for r in records))
    summary = {c: summarize_condition(records, c) for c in conditions}
    base_acc = summary.get(base_condition, {}).get("accuracy_vqa")

    if base_acc is not None:
        for c in conditions:
            cond = summary[c]
            if "accuracy_vqa" in cond:
                cond["accuracy_drop_vs_target_only"] = base_acc - cond["accuracy_vqa"]
                cond["anchor_susceptibility_gap_vs_target_only"] = cond.get("anchor_adoption_rate", 0.0) - summary[base_condition].get("anchor_adoption_rate", 0.0)
                cond["direction_follow_gap_vs_target_only"] = cond.get("anchor_direction_follow_rate", 0.0) - summary[base_condition].get("anchor_direction_follow_rate", 0.0)
    return summary
