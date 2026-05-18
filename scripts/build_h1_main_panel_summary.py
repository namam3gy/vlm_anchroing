"""Aggregate H1 baseline predictions into the §4 main-panel summary CSVs.

Walks `outputs/paper2/cross_model_cross_dataset/predictions/<dataset>/<model>/
predictions.csv` (one file per cell), computes per-cell metrics under the
eps=0 DF form, and emits two artifacts:

  outputs/paper2/cross_model_cross_dataset/summary/main_panel_per_cell.csv
    long format consumed by `notebooks/paper_section_4_figures.ipynb`
    (cross-check vs the notebook's own inline aggregator)

  docs/insights/_data/main_panel_5dataset_per_cell.csv
    same schema, lives next to the legacy gitignored audit artifacts.
    Schema columns: dataset, model, cond_class, stratum, base_correct,
    n, adopt_M2, direction_follow_M2, direction_follow_M2_legacy,
    exact_match.

Run AFTER `scripts/launch_h1_baseline.py` completes all 30 cells.

Usage:
  uv run python scripts/build_h1_main_panel_summary.py
  uv run python scripts/build_h1_main_panel_summary.py --print
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED_ROOT = REPO / "outputs" / "paper2" / "cross_model_cross_dataset" / "predictions"
SUMMARY_DIR = REPO / "outputs" / "paper2" / "cross_model_cross_dataset" / "summary"
DATA_DIR = REPO / "docs" / "insights" / "_data"

# Slug → display name (paper §4 panel order).
DATASETS = [
    ("tallyqa",        "TallyQA"),
    ("chartqa",        "ChartQA"),
    ("mathvista",      "MathVista"),
    ("plotqa",         "PlotQA"),
    ("infographicvqa", "InfographicVQA"),
]
MODELS = [
    "llava-onevision-qwen2-7b-ov",
    "llava-next-interleaved-7b",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "gemma3-4b-it",
    "gemma3-27b-it",
]
# Long-format `cond_class` → predictions.csv `condition` mapping.
ARM_CONDITIONS = {
    "a": "target_plus_irrelevant_number_S1",
    "m": "target_plus_irrelevant_number_masked_S1",
}

LOAD_COLS = [
    "sample_instance_id",
    "condition",
    "exact_match",
    "anchor_adopted",
    "anchor_direction_followed",
    "anchor_direction_followed_moved",
    "pred_b_equal_anchor",
    "numeric_distance_to_anchor",
]


def _arm_metrics(arm_rows: pd.DataFrame) -> dict[str, float | int]:
    """Per-(arm) metrics under eps=0 DF + paired adopt denominator."""
    n = len(arm_rows)
    n_pb_ne_anc = int((arm_rows["pred_b_equal_anchor"] == 0).sum())
    n_num_anchor = int(arm_rows["numeric_distance_to_anchor"].notna().sum())
    eps0_mask = arm_rows["numeric_distance_to_anchor"].notna() & (arm_rows["pred_b_equal_anchor"] == 0)
    n_df_eps0 = int(eps0_mask.sum())
    adopt_num = int(arm_rows["anchor_adopted"].sum())
    df_num = int(arm_rows["anchor_direction_followed_moved"].sum())
    em_num = int(arm_rows["exact_match"].sum())
    return {
        "n": n,
        "adopt_M2": adopt_num / n_pb_ne_anc if n_pb_ne_anc else np.nan,
        "direction_follow_M2": df_num / n_df_eps0 if n_df_eps0 else np.nan,
        "direction_follow_M2_legacy": df_num / n_num_anchor if n_num_anchor else np.nan,
        "exact_match": em_num / n if n else np.nan,
    }


def _cell_rows(dataset_slug: str, dataset_name: str, model: str, pred_csv: Path) -> list[dict]:
    """Long-format rows (cond_class × base_correct) for one (dataset, model) cell.

    Per outline §3.3 the paper headline uses the **base-wrong** cohort
    (b-arm exact_match == 0); the broad / base-correct rows are emitted
    alongside for audit-trail completeness.
    """
    df = pd.read_csv(pred_csv, usecols=LOAD_COLS, low_memory=False)

    base = df[df["condition"] == "target_only"]
    base_wrong_sids = set(base.loc[base["exact_match"] == 0, "sample_instance_id"])
    base_correct_sids = set(base.loc[base["exact_match"] == 1, "sample_instance_id"])

    rows: list[dict] = []
    for cond_class, cond_label in ARM_CONDITIONS.items():
        arm_all = df[df["condition"] == cond_label]
        for base_correct, sid_set in ((False, base_wrong_sids), (True, base_correct_sids)):
            arm_sub = arm_all[arm_all["sample_instance_id"].isin(sid_set)]
            metrics = _arm_metrics(arm_sub)
            rows.append({
                "dataset": dataset_name,
                "model": model,
                "cond_class": cond_class,
                "stratum": "S1",
                "base_correct": base_correct,
                **metrics,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="Print summary to stdout in addition to CSV emit.")
    args = ap.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cell_rows: list[dict] = []
    missing: list[str] = []
    for slug, name in DATASETS:
        for model in MODELS:
            pred_csv = PRED_ROOT / slug / model / "predictions.csv"
            if not pred_csv.exists():
                missing.append(f"{name}/{model}")
                continue
            cell_rows.extend(_cell_rows(slug, name, model, pred_csv))

    if not cell_rows:
        sys.exit(f"No predictions found under {PRED_ROOT}. Re-run launch_h1_baseline.py first.")
    if missing:
        print(f"[warn] {len(missing)} cells missing predictions.csv:")
        for m in missing:
            print(f"  - {m}")

    df = pd.DataFrame(cell_rows)

    out_summary = SUMMARY_DIR / "main_panel_per_cell.csv"
    out_canon = DATA_DIR / "main_panel_5dataset_per_cell.csv"
    df.to_csv(out_summary, index=False)
    df.to_csv(out_canon, index=False)
    print(f"Wrote {out_summary.relative_to(REPO)}  ({len(df)} rows)")
    print(f"Wrote {out_canon.relative_to(REPO)}  ({len(df)} rows)")

    if args.print:
        wb = df[(df["base_correct"] == False) & (df["cond_class"] == "a")]
        wb_pivot = wb.pivot(index="model", columns="dataset", values="direction_follow_M2") * 100
        print("\nDF_a (eps=0, base-wrong, %) by model × dataset:")
        print(wb_pivot.round(1))


if __name__ == "__main__":
    main()
