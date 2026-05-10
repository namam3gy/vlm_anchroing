"""B3 max=16 InfoVQA smoke — re-run in 6-bin from raw predictions.

Mimics original 4-bin B3 sample selection: paired sids only (a-arm S1 +
b-arm), wrong-base subset, equal-frequency 6-bin partition on
answer_span_cross_entropy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/_diag/internvl3_infovqa_max16/20260504-232337"
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(x):
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    m = NUM_RE.search(str(x))
    return float(m.group(0)) if m else float("nan")


def main() -> None:
    rows = [json.loads(line) for line in (RUN_DIR / "predictions.jsonl").open()]
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows from {RUN_DIR.name}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    df["pred_num"] = df["prediction"].apply(parse_num)
    df["gt_num"] = df["ground_truth"].apply(parse_num)
    df["anchor_num"] = df["anchor_value"].apply(parse_num)

    b = df[df["condition"] == "target_only"].set_index("sample_instance_id")
    a = df[df["condition"] == "target_plus_irrelevant_number_S1"].set_index("sample_instance_id")
    common = a.index.intersection(b.index)
    print(f"Paired sids: {len(common)}")

    # Match original B3 sample selection: bin by b-arm wrong-base + valid b-arm
    # confidence; df numerator only fires on sids with valid a-arm parse (M2
    # denominator-style — n_numeric_anchor is the inner denominator).
    b_filt = b.loc[common]
    pb = b_filt["pred_num"]
    gt_b = b_filt["gt_num"]
    conf = b_filt["answer_span_cross_entropy"]
    valid_b = pb.notna() & gt_b.notna() & conf.notna()
    wrong_b = valid_b & (pb != gt_b)
    sids_wrong = b_filt.index[wrong_b]
    n_w = len(sids_wrong)
    print(f"b-arm wrong-base sids (matches original B3 n=193 selection): {n_w}")

    pa = a.loc[sids_wrong, "pred_num"]
    pb = b_filt.loc[sids_wrong, "pred_num"]
    anchor = a.loc[sids_wrong, "anchor_num"]
    gt = a.loc[sids_wrong, "gt_num"]
    conf = b_filt.loc[sids_wrong, "answer_span_cross_entropy"]

    pa_w, pb_w, anc_w, conf_w = pa, pb, anchor, conf
    n_w = len(pa_w)

    # 6-bin: sort by conf ascending (lower entropy = more confident = B1)
    df_bin = pd.DataFrame({
        "pa": pa_w.values, "pb": pb_w.values, "anchor": anc_w.values, "conf": conf_w.values
    }).sort_values("conf").reset_index(drop=True)
    df_bin["bin"] = ((np.arange(n_w) * 6) // n_w).clip(max=5)

    print(f"\n=== B3 max=16 InfoVQA wrong-base 6-bin (n={n_w}, M2 denominator) ===")
    print(f"{'bin':5s}  n   n_numeric  df")
    for k in range(6):
        cell = df_bin[df_bin["bin"] == k]
        n = len(cell)
        # M2 df: denominator = numeric pair (anchor not null AND pa not null)
        n_num = int((cell["pa"].notna() & cell["anchor"].notna()).sum())
        if n_num == 0:
            print(f"B{k+1:<3d} {n:3d}  {n_num:7d}  NaN")
            continue
        sub = cell.dropna(subset=["pa", "pb", "anchor"])
        sign_ok = ((sub["pa"] - sub["pb"]) * (sub["anchor"] - sub["pb"])) > 0
        pa_ne_pb = sub["pa"] != sub["pb"]
        df_rate = float((sign_ok & pa_ne_pb).sum() / n_num)
        print(f"B{k+1:<3d} {n:3d}  {n_num:7d}  {df_rate:.4f}")

    # Compute B1 and B6 explicitly
    def _df(cell):
        n_num = int((cell["pa"].notna() & cell["anchor"].notna()).sum())
        if n_num == 0:
            return float("nan")
        sub = cell.dropna(subset=["pa", "pb", "anchor"])
        sign_ok = ((sub["pa"] - sub["pb"]) * (sub["anchor"] - sub["pb"])) > 0
        pa_ne_pb = sub["pa"] != sub["pb"]
        return float((sign_ok & pa_ne_pb).sum() / n_num)
    df_b1 = _df(df_bin[df_bin["bin"] == 0])
    df_b6 = _df(df_bin[df_bin["bin"] == 5])
    print(f"\nB6 - B1 = {df_b6 - df_b1:+.4f}  (B1={df_b1:.4f}, B6={df_b6:.4f})")


if __name__ == "__main__":
    main()
