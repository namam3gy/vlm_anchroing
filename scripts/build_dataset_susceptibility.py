"""Build a per-dataset susceptibility CSV from a single model's predictions.jsonl.

The cross-model VQAv2 susceptibility CSV (`docs/insights/_data/susceptibility_strata.csv`)
was computed across 7 VQAv2 main-panel models. The 5-dataset main matrix
(PlotQA / InfographicVQA / TallyQA / ChartQA / MathVista) lacks an equivalent
cross-model panel at the §7.1-7.3 analysis tier, so this builder produces a
**single-model proxy** susceptibility CSV from one model's predictions.

Per-question score:
    moved_closer = 1 if the (a-arm prediction) moved toward the anchor digit
                     (relative to the b-arm baseline) else 0
    score        = mean(moved_closer) across the question's repetitions / strata
                     (typically 1 if there's only one a-arm record per question).

Top decile by score → "top_decile_susceptible". Bottom decile → "bottom_decile_resistant".

Output columns match the 7-model CSV so `extract_attention_mass.py
--susceptibility-csv` can consume it without modification:

    question_id, n_models, mean_moved_closer, std_moved_closer,
    mean_adoption, mean_pull, question, question_type, image_id,
    base_ground_truth, susceptibility_stratum

Usage::

    uv run python scripts/build_dataset_susceptibility.py \
        --predictions outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/<ts>/predictions.jsonl \
        --output docs/insights/_data/susceptibility_plotqa.csv \
        --top-n 100 --bottom-n 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="Path to a single-model predictions.jsonl.")
    p.add_argument("--output", required=True,
                   help="Output CSV path.")
    p.add_argument("--top-n", type=int, default=100,
                   help="Top N (most susceptible) questions.")
    p.add_argument("--bottom-n", type=int, default=100,
                   help="Bottom N (most resistant) questions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"predictions not found: {pred_path}")

    rows = []
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print(f"[load] {len(df)} records from {pred_path.name}")

    # Per question_id: compute moved_closer + adoption + numeric pull.
    # moved_closer is sign-based: 1 if (pa-pb)*(anchor-pb) > 0 (C-form raw),
    # else 0; only counted when both pa and pb are numeric and anchor present.
    a_conds = ["target_plus_irrelevant_number", "target_plus_irrelevant_number_S1"]
    b_cond = "target_only"

    qrows = []
    for qid, sub in df.groupby("question_id"):
        a_rec = sub[sub["condition"].isin(a_conds)]
        b_rec = sub[sub["condition"] == b_cond]
        if a_rec.empty or b_rec.empty:
            continue
        moved_list, adopt_list, pull_list = [], [], []
        for _, ar in a_rec.iterrows():
            for _, br in b_rec.iterrows():
                anchor = ar.get("anchor_value")
                pa = ar.get("prediction")
                pb = br.get("prediction")
                try:
                    pa_i = int(pa) if pa not in (None, "") else None
                    pb_i = int(pb) if pb not in (None, "") else None
                    a_i = int(anchor) if anchor not in (None, "") else None
                except (TypeError, ValueError):
                    continue
                if a_i is None or pa_i is None or pb_i is None:
                    continue
                # C-form sign: (pa-pb) and (anchor-pb) same sign and pa != pb
                if pa_i != pb_i:
                    moved = 1 if (pa_i - pb_i) * (a_i - pb_i) > 0 else 0
                else:
                    moved = 0
                moved_list.append(moved)
                adopt_list.append(int(pa_i == a_i and pb_i != a_i))
                pull_list.append(abs(pa_i - pb_i) if pa_i != pb_i else 0.0)
        if not moved_list:
            continue
        qrows.append({
            "question_id": int(qid),
            "n_models": 1,
            "mean_moved_closer": sum(moved_list) / len(moved_list),
            "std_moved_closer": 0.0,
            "mean_adoption": sum(adopt_list) / len(adopt_list),
            "mean_pull": sum(pull_list) / len(pull_list),
            "question": (b_rec["question"].iloc[0] if "question" in b_rec else ""),
            "question_type": (b_rec["question_type"].iloc[0] if "question_type" in b_rec else ""),
            "image_id": (b_rec["image_id"].iloc[0] if "image_id" in b_rec else ""),
            "base_ground_truth": (b_rec["ground_truth"].iloc[0] if "ground_truth" in b_rec else ""),
        })

    out = pd.DataFrame(qrows)
    if out.empty:
        raise SystemExit("no eligible questions after filtering")

    # Stratify by mean_moved_closer.
    sorted_by = out.sort_values("mean_moved_closer", ascending=False).reset_index(drop=True)
    top_n = min(args.top_n, len(sorted_by) // 2)
    bottom_n = min(args.bottom_n, len(sorted_by) // 2)
    sorted_by["susceptibility_stratum"] = "middle"
    sorted_by.loc[: top_n - 1, "susceptibility_stratum"] = "top_decile_susceptible"
    sorted_by.loc[len(sorted_by) - bottom_n :, "susceptibility_stratum"] = "bottom_decile_resistant"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_by.to_csv(out_path, index=False)
    print(f"[done] wrote {len(sorted_by)} rows to {out_path}")
    print(f"  top_decile_susceptible: {top_n}, bottom_decile_resistant: {bottom_n}")
    print(f"  mean_moved_closer range: [{sorted_by['mean_moved_closer'].min():.3f}, "
          f"{sorted_by['mean_moved_closer'].max():.3f}]")


if __name__ == "__main__":
    main()
