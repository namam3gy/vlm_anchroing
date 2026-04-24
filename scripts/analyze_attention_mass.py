"""Analyze attention-mass output and test the E1 mechanism prediction.

Addresses four gotchas flagged by review:
  1. Analyse at the **answer-digit generation step**, not step 0. Step 0 is usually the
     opening brace `{`; the answer digit lands several steps later.
  2. Do **not** join with outputs/experiment/<model>/predictions.csv for base_correct —
     the attention run's set00 anchor image differs from the predictions run's set00
     (variants_per_sample=1 vs 5 changes the RNG trajectory). Derive base_correct from
     the attention run's OWN target_only condition instead.
  3. Stratify by susceptibility_stratum (from _data/susceptibility_strata.csv). A7
     predicts the mechanism should be strongest on universally-susceptible questions.
  4. Bootstrap 95 % CI on every delta that we might report.

Usage:
    uv run python scripts/analyze_attention_mass.py \\
        --attention-jsonl outputs/attention_analysis/<model>/<run>/per_step_attention.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-jsonl", type=str, required=True)
    parser.add_argument("--susceptibility-csv", type=str,
                        default="docs/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    return parser.parse_args()


_DIGIT_RE = re.compile(r"\d")


def _find_answer_digit_step(per_step_tokens: list[dict]) -> int | None:
    """Return the index of the first step whose token_text contains a digit.

    For the JSON-only prompt the digits come after `{"result": `, typically step 4–6.
    If no digit token is found (e.g., model fails to emit JSON), return None.
    """
    for rec in per_step_tokens:
        if _DIGIT_RE.search(rec.get("token_text", "") or ""):
            return int(rec["step"])
    return None


def _derive_base_correct(att_df: pd.DataFrame) -> pd.DataFrame:
    """For each sample_instance, read base_correct off the attention run's own target_only row."""
    base_only = att_df.loc[att_df["condition"] == "target_only", ["sample_instance_id", "decoded", "ground_truth"]].copy()
    base_only["pred_int"] = base_only["decoded"].astype(str).str.extract(r"(-?\d+)")[0]
    base_only["base_correct_att"] = (base_only["pred_int"].astype("string") == base_only["ground_truth"].astype("string")).astype(int)
    return base_only[["sample_instance_id", "base_correct_att"]]


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = rng.choice(values, size=values.size, replace=True).mean()
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _mass_at_step(record: dict, step: int, region: str) -> float:
    per_step = record.get("per_step", [])
    if step is None or step >= len(per_step):
        return float("nan")
    layers = per_step[step].get(region, [])
    return float(np.mean(layers)) if layers else float("nan")


def _build_analysis_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if "error" in r or not r.get("per_step"):
            continue
        step_answer = _find_answer_digit_step(r.get("per_step_tokens", []))
        rows.append({
            "sample_instance_id": r["sample_instance_id"],
            "question_id": int(r["question_id"]),
            "image_id": int(r["image_id"]),
            "condition": r["condition"],
            "irrelevant_type": r["irrelevant_type"],
            "anchor_value": r.get("anchor_value"),
            "ground_truth": r["ground_truth"],
            "decoded": r["decoded"],
            "n_layers": r["n_layers"],
            "n_steps": r["n_steps"],
            "step_answer": step_answer,
            "image_target_mass_step0": _mass_at_step(r, 0, "image_target"),
            "image_anchor_mass_step0": _mass_at_step(r, 0, "image_anchor"),
            "image_target_mass_answer": _mass_at_step(r, step_answer, "image_target") if step_answer is not None else float("nan"),
            "image_anchor_mass_answer": _mass_at_step(r, step_answer, "image_anchor") if step_answer is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def _run_test(
    triplet: pd.DataFrame,
    delta_col: str,
    label: str,
    bootstrap_n: int,
    rng_seed: int,
) -> None:
    """Print a self-contained block of stats for one (num − neut) delta column."""
    vals = triplet[delta_col].dropna().to_numpy()
    if vals.size == 0:
        print(f"  {label}: no valid rows")
        return
    mean = float(vals.mean())
    median = float(np.median(vals))
    pos_share = float((vals > 0).mean())
    ci_low, ci_high = _bootstrap_mean_ci(vals, bootstrap_n, rng_seed)
    print(f"  {label:45s} n={vals.size:4d}  mean={mean:+.5f}  95% CI=[{ci_low:+.5f}, {ci_high:+.5f}]  median={median:+.5f}  share>0={pos_share:.3f}")


def main() -> None:
    args = _parse_args()
    att_path = Path(args.attention_jsonl).resolve()

    records = [json.loads(l) for l in att_path.read_text().splitlines() if l.strip()]
    print(f"[load] {len(records)} attention records from {att_path.name}")
    att_df = _build_analysis_frame(records)
    print(f"[load] {len(att_df)} valid rows; {att_df['condition'].nunique()} conditions")

    # Sanity: per-step tokens present?
    has_tokens = any("per_step_tokens" in r for r in records)
    if not has_tokens:
        print("[warn] per_step_tokens missing — answer-step analysis will be nan")

    # base_correct from attention-run's own target_only condition
    base = _derive_base_correct(att_df)
    att_df = att_df.merge(base, on="sample_instance_id", how="left")
    print(f"[join] derived base_correct from att run: coverage = {att_df['base_correct_att'].notna().sum()}/{len(att_df)}")

    # Susceptibility stratum
    susc_path = PROJECT_ROOT / args.susceptibility_csv
    susc = pd.read_csv(susc_path)[["question_id", "susceptibility_stratum", "mean_moved_closer"]]
    att_df = att_df.merge(susc, on="question_id", how="left")
    print(f"[join] susceptibility coverage = {att_df['susceptibility_stratum'].notna().sum()}/{len(att_df)}")

    # Build triplets: one row per sample_instance_id with num/neut masses + base_correct + stratum
    base_info = (
        att_df.loc[att_df["condition"] == "target_only", ["sample_instance_id", "question_id", "base_correct_att", "susceptibility_stratum"]]
        .drop_duplicates(subset=["sample_instance_id"])
    )
    num = (
        att_df.loc[att_df["condition"] == "target_plus_irrelevant_number",
                   ["sample_instance_id", "image_anchor_mass_step0", "image_anchor_mass_answer", "step_answer"]]
        .rename(columns={
            "image_anchor_mass_step0": "num_anchor_mass_step0",
            "image_anchor_mass_answer": "num_anchor_mass_answer",
            "step_answer": "num_step_answer",
        })
    )
    neut = (
        att_df.loc[att_df["condition"] == "target_plus_irrelevant_neutral",
                   ["sample_instance_id", "image_anchor_mass_step0", "image_anchor_mass_answer", "step_answer"]]
        .rename(columns={
            "image_anchor_mass_step0": "neut_anchor_mass_step0",
            "image_anchor_mass_answer": "neut_anchor_mass_answer",
            "step_answer": "neut_step_answer",
        })
    )
    triplet = base_info.merge(num, on="sample_instance_id", how="inner").merge(neut, on="sample_instance_id", how="inner")
    triplet["delta_step0"] = triplet["num_anchor_mass_step0"] - triplet["neut_anchor_mass_step0"]
    triplet["delta_answer"] = triplet["num_anchor_mass_answer"] - triplet["neut_anchor_mass_answer"]

    print(f"\n[triplet] {len(triplet)} pairs")
    print(f"  answer-step availability: num={triplet['num_step_answer'].notna().sum()}, neut={triplet['neut_step_answer'].notna().sum()}")

    # ─── Test 1: overall (num − neut) delta ───
    print("\n=== Test 1: image_anchor_mass(number) vs (neutral), overall ===")
    _run_test(triplet, "delta_step0", "step 0 (opening brace)", args.bootstrap_n, args.rng_seed)
    _run_test(triplet, "delta_answer", "answer-digit step", args.bootstrap_n, args.rng_seed)

    # ─── Test 2: stratified by base_correct (H2 mechanistic test) ───
    print("\n=== Test 2: stratified by base_correct_att ===")
    for corr_label, sub in triplet.groupby(triplet["base_correct_att"].fillna(-1).astype(int)):
        name = {0: "wrong", 1: "correct", -1: "unknown"}.get(int(corr_label), str(corr_label))
        if len(sub) == 0:
            continue
        print(f"  --- base_correct = {name} (n={len(sub)}) ---")
        _run_test(sub, "delta_step0", "  step 0", args.bootstrap_n, args.rng_seed)
        _run_test(sub, "delta_answer", "  answer-digit step", args.bootstrap_n, args.rng_seed)

    # ─── Test 3: stratified by susceptibility stratum (A7 prediction) ───
    print("\n=== Test 3: stratified by susceptibility_stratum ===")
    for stratum, sub in triplet.groupby("susceptibility_stratum"):
        if len(sub) == 0:
            continue
        print(f"  --- {stratum} (n={len(sub)}) ---")
        _run_test(sub, "delta_step0", "  step 0", args.bootstrap_n, args.rng_seed)
        _run_test(sub, "delta_answer", "  answer-digit step", args.bootstrap_n, args.rng_seed)

    # ─── Test 4: combined base_correct × stratum ───
    print("\n=== Test 4: combined stratification (wrong × top-decile = strongest prediction) ===")
    for (corr, stratum), sub in triplet.groupby([triplet["base_correct_att"].fillna(-1).astype(int), "susceptibility_stratum"]):
        corr_name = {0: "wrong", 1: "correct", -1: "unknown"}.get(int(corr), str(corr))
        if len(sub) < 10:
            continue
        print(f"  --- {corr_name} × {stratum} (n={len(sub)}) ---")
        _run_test(sub, "delta_answer", "  answer-digit step", args.bootstrap_n, args.rng_seed)

    # ─── Outputs ───
    out_dir = Path(args.out_dir) if args.out_dir else att_path.parent
    triplet.to_csv(out_dir / "analysis_triplets.csv", index=False)
    print(f"\n[done] triplets saved -> {out_dir / 'analysis_triplets.csv'}")


if __name__ == "__main__":
    main()
