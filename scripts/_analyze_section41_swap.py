"""§4.1 dataset-swap analysis.

Pulls the §4.1 Table 2 analog (acc(b), adopt(a), df(a), em(a)) and the §4.1
Insight-1 wrong-correct gap from existing predictions.jsonl on the three
candidate replacement datasets (PlotQA, TallyQA, InfographicVQA), 7-model
panel where available.

All-base S1 anchor arm. C-form direction-follow.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

JOBS = [
    ("experiment_e7_plotqa_full", "PlotQA"),
    ("experiment_e7_infographicvqa_full", "InfoVQA"),
    ("experiment_e5e_tallyqa_full", "TallyQA"),
]

MODELS = [
    "gemma3-4b-it",
    "gemma3-27b-it",
    "internvl3-8b",
    "llava-next-interleaved-7b",
    "llava-onevision-qwen2-7b-ov",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    m = NUM_RE.search(s)
    return float(m.group(0)) if m else float("nan")


def latest_run(model_dir: Path) -> Path | None:
    """Pick the run with the most prediction rows (avoid smoke-run aliases)."""
    best, best_n = None, -1
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        pred = run_dir / "predictions.jsonl"
        if not pred.exists():
            continue
        n = sum(1 for _ in pred.open())
        if n > best_n:
            best, best_n = run_dir, n
    return best


def load_predictions(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").open()]
    return pd.DataFrame(rows)


def m2_metrics(pa: pd.Series, pb: pd.Series, anchor: pd.Series, gt: pd.Series) -> dict:
    valid = pa.notna() & pb.notna() & anchor.notna() & gt.notna()
    pa, pb, anchor, gt = pa[valid], pb[valid], anchor[valid], gt[valid]
    n = int(valid.sum())
    if n == 0:
        return dict(n=0, n_denom_adopt=0, adopt=float("nan"), df=float("nan"), em=float("nan"))
    pb_ne_anchor = pb != anchor
    n_denom_adopt = int(pb_ne_anchor.sum())
    pa_eq_anchor = pa == anchor
    adopt = float((pa_eq_anchor & pb_ne_anchor).sum() / max(1, n_denom_adopt))
    pa_ne_pb = pa != pb
    sign_ok = ((pa - pb) * (anchor - pb)) > 0
    df = float((pa_ne_pb & sign_ok).sum() / n)
    em = float((pa == gt).sum() / n)
    return dict(n=n, n_denom_adopt=n_denom_adopt, adopt=adopt, df=df, em=em)


def per_run(df: pd.DataFrame) -> dict:
    """Compute §4.1-analog headline + wrong/correct split for one model run."""
    df = df.copy()
    df["pred_num"] = df["prediction"].apply(parse_num)
    df["gt_num"] = df["ground_truth"].apply(parse_num)
    df["anchor_num"] = df["anchor_value"].apply(parse_num)

    # b-arm: target_only condition (no stratum)
    b = df[df["condition"] == "target_only"].set_index("sample_instance_id")
    if len(b) == 0:
        return {}

    # a-arm S1 — match condition that contains target_plus_irrelevant_number
    # but NOT masked/neutral. Stratum filter via anchor_stratum_id == 1 if multi-strata,
    # else the bare condition.
    a_mask = df["condition"].str.contains("target_plus_irrelevant_number", na=False) & ~df["condition"].str.contains("masked|neutral", na=False)
    a = df[a_mask].copy()
    # If multi-strata data is present, restrict to S1 only
    if a["condition"].str.contains("_S[2-9]", na=False).any():
        a = a[a["condition"].str.endswith("_S1") | (a["condition"] == "target_plus_irrelevant_number")]
    a = a.set_index("sample_instance_id")
    if len(a) == 0:
        return {}

    common = a.index.intersection(b.index)
    pa = a.loc[common, "pred_num"]
    pb = b.loc[common, "pred_num"]
    anchor = a.loc[common, "anchor_num"]
    gt = a.loc[common, "gt_num"]

    all_base = m2_metrics(pa, pb, anchor, gt)

    # b-arm acc(b) over the b-arm itself (not paired)
    pb_full = b["pred_num"]
    gt_b = b["gt_num"]
    valid_b = pb_full.notna() & gt_b.notna()
    acc_b = float((pb_full[valid_b] == gt_b[valid_b]).sum() / max(1, valid_b.sum()))
    n_b = int(valid_b.sum())

    # wrong/correct split based on pb_full
    base_correct_b = (pb_full == gt_b) & valid_b
    sids_correct = set(b.index[base_correct_b])
    sids_wrong = set(b.index[~base_correct_b & valid_b])

    mask_w = pa.index.isin(sids_wrong)
    mask_c = pa.index.isin(sids_correct)
    wrong = m2_metrics(pa[mask_w], pb[mask_w], anchor[mask_w], gt[mask_w])
    correct = m2_metrics(pa[mask_c], pb[mask_c], anchor[mask_c], gt[mask_c])

    return dict(
        n_b=n_b, acc_b=acc_b,
        n_pair=all_base["n"],
        adopt_all=all_base["adopt"], df_all=all_base["df"], em_all=all_base["em"],
        n_wrong=wrong["n"], adopt_wrong=wrong["adopt"], df_wrong=wrong["df"], em_wrong=wrong["em"],
        n_correct=correct["n"], adopt_correct=correct["adopt"], df_correct=correct["df"], em_correct=correct["em"],
        wrong_minus_correct_adopt=wrong["adopt"] - correct["adopt"]
            if not (np.isnan(wrong["adopt"]) or np.isnan(correct["adopt"])) else float("nan"),
        wrong_minus_correct_df=wrong["df"] - correct["df"]
            if not (np.isnan(wrong["df"]) or np.isnan(correct["df"])) else float("nan"),
    )


def main() -> None:
    rows = []
    for exp_dir, dataset in JOBS:
        for model in MODELS:
            model_dir = ROOT / "outputs" / exp_dir / model
            if not model_dir.exists():
                continue
            run_dir = latest_run(model_dir)
            if run_dir is None:
                continue
            df = load_predictions(run_dir)
            res = per_run(df)
            if not res:
                continue
            rows.append(dict(dataset=dataset, model=model, run=run_dir.name, **res))

    out = pd.DataFrame(rows)
    out_csv = ROOT / "docs" / "insights" / "_data" / "section41_swap_analysis.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")

    for ds in ["PlotQA", "TallyQA", "InfoVQA"]:
        sub = out[out["dataset"] == ds].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("df_all", ascending=False)
        print(f"\n=== {ds} — §4.1 Table 2 analog (all-base S1, paired n) ===")
        cols = ["model", "n_b", "n_pair", "acc_b", "adopt_all", "df_all", "em_all"]
        formatters = {"acc_b": "{:.3f}".format, "adopt_all": "{:.3f}".format,
                      "df_all": "{:.3f}".format, "em_all": "{:.3f}".format}
        print(sub[cols].to_string(index=False, formatters=formatters))

        print(f"\n--- {ds} — wrong > correct gap (Insight 1) ---")
        cols2 = ["model", "n_wrong", "n_correct",
                 "adopt_wrong", "adopt_correct", "wrong_minus_correct_adopt",
                 "df_wrong", "df_correct", "wrong_minus_correct_df"]
        f2 = {c: "{:.3f}".format for c in cols2 if c not in ("model", "n_wrong", "n_correct")}
        print(sub[cols2].to_string(index=False, formatters=f2))


if __name__ == "__main__":
    main()
