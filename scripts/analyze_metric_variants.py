"""M2 — metric-variant analysis.

Enumerate adopt and direction-follow variants on existing ``predictions.jsonl``
outputs and report which (numerator, denominator) combination preserves known
qualitative signals most cleanly.

Signals checked:
  - wrong-base > correct-base  (Phase A A1)
  - distance decay S1 > S5     (E5b/E5c/E5d)
  - anchor > masked > neutral  (E5c)

Variants:
  adopt              numerator                                              denominator
  -----              ---------                                              -----------
  A_raw              pa == anchor                                           D_all       — all sample-pairs
  A_paired           pa == anchor AND pb != anchor                          D_paired    — pb != anchor
  A_clean            pa == anchor AND pb != anchor AND gt != anchor         D_clean     — pb != anchor AND gt != anchor

  direction-follow   numerator                                              denominator
  ----------------   ---------                                              -----------
  DF_raw             (pb-gt)*(px-gt) > 0                                    DD_all      — numeric pair
  DF_moved           DF_raw AND px != pb                                    DD_moved    — numeric pair AND px != pb
  DF_clean           DF_raw AND px != pb AND gt != anchor                   DD_clean    — numeric pair AND px != pb AND gt != anchor

Reads only; no re-inference.

Outputs (under ``docs/insights/_data/``):
  - M2_metric_variants_long.csv        long-form: one row per (cell, variant)
  - M2_metric_variants_wide.csv        wide-form: one row per cell, all 18 variants
  - M2_signal_preservation.csv         per-variant signal-survival score
  - M2_inputs_manifest.csv             which predictions.jsonl files were read

All boolean flags use the canonical naming: pb / pa / pm / pd / anchor / gt.
The arm being compared to baseline is generically ``px`` (= pa, pm, or pd).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from vlm_anchor.utils import extract_first_number, normalize_numeric_text


# ---------- Inputs ----------

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "docs" / "insights" / "_data"


@dataclass(frozen=True)
class InputSpec:
    """One predictions.jsonl source."""
    experiment: str
    dataset: str
    model: str
    run: str
    path: Path


def discover_inputs() -> list[InputSpec]:
    """Walk ``outputs/`` and pick up every relevant predictions.jsonl.

    Skips: smoke runs, partial files (no summary.json sibling), pilot runs,
    ablation/mitigation jsonl (different schema)."""
    specs: list[InputSpec] = []

    layouts: list[tuple[str, str, str]] = [
        # (experiment_dir, dataset_label, what)
        ("experiment", "VQAv2", "main"),
        ("experiment_anchor_strengthen_prompt", "VQAv2", "strengthen"),
        ("experiment_distance_vqa", "VQAv2", "e5b"),
        ("experiment_distance_tally", "TallyQA", "e5b"),
        ("experiment_e5c_vqa", "VQAv2", "e5c"),
        ("experiment_e5c_tally", "TallyQA", "e5c"),
        ("experiment_e5e_chartqa_full", "ChartQA", "e5e"),
        ("experiment_e5e_tallyqa_full", "TallyQA", "e5e"),
    ]

    for exp_dir, dataset, _ in layouts:
        root = OUTPUTS / exp_dir
        if not root.is_dir():
            continue
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != "analysis"):
                jsonl = run_dir / "predictions.jsonl"
                summary = run_dir / "summary.json"
                if not jsonl.is_file() or not summary.is_file():
                    continue
                specs.append(InputSpec(
                    experiment=exp_dir,
                    dataset=dataset,
                    model=model_dir.name,
                    run=run_dir.name,
                    path=jsonl,
                ))
    return specs


# ---------- Condition canonicalization ----------

COND_RE = re.compile(
    r"target_plus_irrelevant_(?P<kind>number_masked|number|neutral)(?:_(?P<stratum>S\d))?$"
)


def normalize_condition(cond: str) -> tuple[str, str | None]:
    """target_only -> ('b', None); '*_number_S1' -> ('a', 'S1'); etc.

    Returns (cond_class, stratum). cond_class in {'b','a','m','d','?'}.
    """
    if cond == "target_only":
        return ("b", None)
    m = COND_RE.match(cond)
    if not m:
        return ("?", None)
    kind = m.group("kind")
    stratum = m.group("stratum")
    if kind == "number":
        return ("a", stratum)
    if kind == "number_masked":
        return ("m", stratum)
    if kind == "neutral":
        return ("d", stratum)
    return ("?", None)


def is_int_str(s: str | None) -> bool:
    return bool(s) and s.lstrip("-").isdigit()


# ---------- Pair flags ----------

@dataclass
class PairFlags:
    pred_b: str
    pred_x: str
    gt: str
    anchor: str | None
    pb_eq_a: bool
    px_eq_a: bool
    gt_eq_a: bool
    px_ne_pb: bool
    pb_eq_gt: bool
    base_correct: bool
    df_raw: bool
    numeric_ok: bool


def compute_flags(b_row: dict, x_row: dict) -> PairFlags:
    pred_b = normalize_numeric_text(extract_first_number(b_row.get("prediction", "") or ""))
    pred_x = normalize_numeric_text(extract_first_number(x_row.get("prediction", "") or ""))
    gt = normalize_numeric_text(extract_first_number(x_row.get("ground_truth", "") or ""))
    anchor_raw = x_row.get("anchor_value")
    anchor = normalize_numeric_text(str(anchor_raw)) if anchor_raw is not None else None

    pb_eq_a = bool(anchor) and pred_b == anchor
    px_eq_a = bool(anchor) and pred_x == anchor
    gt_eq_a = bool(anchor) and gt == anchor
    px_ne_pb = pred_x != pred_b
    pb_eq_gt = pred_b == gt and is_int_str(pred_b)
    base_correct = pb_eq_gt

    numeric_ok = (
        is_int_str(pred_b)
        and is_int_str(pred_x)
        and is_int_str(gt)
        and (anchor is None or is_int_str(anchor))
    )

    df_raw = False
    if numeric_ok and anchor is not None:
        pb_i, px_i, gt_i = int(pred_b), int(pred_x), int(gt)
        df_raw = (pb_i - gt_i) * (px_i - gt_i) > 0

    return PairFlags(
        pred_b=pred_b,
        pred_x=pred_x,
        gt=gt,
        anchor=anchor,
        pb_eq_a=pb_eq_a,
        px_eq_a=px_eq_a,
        gt_eq_a=gt_eq_a,
        px_ne_pb=px_ne_pb,
        pb_eq_gt=pb_eq_gt,
        base_correct=base_correct,
        df_raw=df_raw,
        numeric_ok=numeric_ok,
    )


# ---------- Variant predicates ----------

ADOPT_NUM: dict[str, callable] = {
    "A_raw":    lambda f: f.px_eq_a,
    "A_paired": lambda f: f.px_eq_a and not f.pb_eq_a,
    "A_clean":  lambda f: f.px_eq_a and not f.pb_eq_a and not f.gt_eq_a,
}
ADOPT_DEN: dict[str, callable] = {
    "D_all":    lambda f: True,
    "D_paired": lambda f: not f.pb_eq_a,
    "D_clean":  lambda f: (not f.pb_eq_a) and (not f.gt_eq_a),
}
DF_NUM: dict[str, callable] = {
    "DF_raw":   lambda f: f.df_raw,
    "DF_moved": lambda f: f.df_raw and f.px_ne_pb,
    "DF_clean": lambda f: f.df_raw and f.px_ne_pb and not f.gt_eq_a,
}
DF_DEN: dict[str, callable] = {
    "DD_all":   lambda f: f.numeric_ok and f.anchor is not None,
    "DD_moved": lambda f: f.numeric_ok and f.anchor is not None and f.px_ne_pb,
    "DD_clean": lambda f: f.numeric_ok and f.anchor is not None and f.px_ne_pb and not f.gt_eq_a,
}


# ---------- Pair iteration ----------

def iter_pairs(spec: InputSpec) -> Iterator[tuple[str, str | None, dict, dict]]:
    """Yield (cond_class, stratum, b_row, x_row) tuples for every (b, x) pair
    sharing the same sample_instance_id.

    Skips pairs where the b row is missing or the cond_class is unknown.
    """
    by_sid: dict[str, dict[tuple[str, str | None], dict]] = defaultdict(dict)
    with spec.path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_instance_id")
            if sid is None:
                continue
            cls, stratum = normalize_condition(row.get("condition", ""))
            if cls == "?":
                continue
            by_sid[sid][(cls, stratum)] = row

    for sid, conds in by_sid.items():
        b_row = conds.get(("b", None))
        if b_row is None:
            continue
        for (cls, stratum), x_row in conds.items():
            if cls == "b":
                continue
            yield cls, stratum, b_row, x_row


# ---------- Aggregation ----------

CELL_KEYS = ["experiment", "dataset", "model", "cond_class", "stratum", "base_correct"]


def emit_long(spec: InputSpec) -> Iterator[dict]:
    """Yield one event per (sample, variant). Heavy; aggregator collapses it."""
    for cls, stratum, b_row, x_row in iter_pairs(spec):
        flags = compute_flags(b_row, x_row)
        # base_correct filtering only meaningful when pb is parseable
        bc = "all"  # always emit "all"
        for bc_label in ("all", "wrong", "correct"):
            if bc_label == "wrong" and (not is_int_str(flags.pred_b) or flags.base_correct):
                continue
            if bc_label == "correct" and (not is_int_str(flags.pred_b) or not flags.base_correct):
                continue
            cell = {
                "experiment": spec.experiment,
                "dataset": spec.dataset,
                "model": spec.model,
                "cond_class": cls,
                "stratum": stratum or "S0",
                "base_correct": bc_label,
            }
            for n_id, n_pred in ADOPT_NUM.items():
                for d_id, d_pred in ADOPT_DEN.items():
                    cell[f"{n_id}__{d_id}_num"] = int(bool(n_pred(flags) and d_pred(flags)))
                    cell[f"{n_id}__{d_id}_den"] = int(bool(d_pred(flags)))
            for n_id, n_pred in DF_NUM.items():
                for d_id, d_pred in DF_DEN.items():
                    cell[f"{n_id}__{d_id}_num"] = int(bool(n_pred(flags) and d_pred(flags)))
                    cell[f"{n_id}__{d_id}_den"] = int(bool(d_pred(flags)))
            yield cell


def aggregate(specs: list[InputSpec]) -> pd.DataFrame:
    """Aggregate per-cell sums of numerator and denominator across pairs."""
    rows: list[dict] = []
    for spec in specs:
        for cell in emit_long(spec):
            rows.append(cell)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sum_cols = [c for c in df.columns if c.endswith("_num") or c.endswith("_den")]
    grouped = df.groupby(CELL_KEYS, dropna=False)[sum_cols].sum().reset_index()
    grouped["n_pairs"] = (
        df.groupby(CELL_KEYS, dropna=False).size().reset_index(name="n_pairs")["n_pairs"]
    )
    return grouped


def to_long(grouped: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-cell wide table into one row per (cell, variant)."""
    variant_pairs: list[tuple[str, str, str]] = []
    for n_id in ADOPT_NUM:
        for d_id in ADOPT_DEN:
            variant_pairs.append(("adopt", n_id, d_id))
    for n_id in DF_NUM:
        for d_id in DF_DEN:
            variant_pairs.append(("df", n_id, d_id))

    out: list[dict] = []
    for _, row in grouped.iterrows():
        for metric, n_id, d_id in variant_pairs:
            num_col = f"{n_id}__{d_id}_num"
            den_col = f"{n_id}__{d_id}_den"
            num = int(row[num_col])
            den = int(row[den_col])
            rate = (num / den) if den > 0 else None
            out.append({
                **{k: row[k] for k in CELL_KEYS},
                "n_pairs": int(row["n_pairs"]),
                "metric": metric,
                "numerator": n_id,
                "denominator": d_id,
                "variant_id": f"{n_id}__{d_id}",
                "num": num,
                "den": den,
                "rate": rate,
            })
    return pd.DataFrame(out)


# ---------- Signal preservation ----------

def signal_preservation(long_df: pd.DataFrame) -> pd.DataFrame:
    """For each variant, score how often it preserves the three known signals
    on the a-arm cells (adopt and df only — d-arm not applicable):

      S_wc:    wrong-base rate > correct-base rate within (experiment, dataset, model, stratum)
      S_dist:  S1 rate > S5 rate within (experiment, dataset, model, base_correct=wrong)
      S_amd:   on E5c (experiment in {experiment_e5c_vqa, experiment_e5c_tally}):
               anchor rate > masked rate AND masked rate >= neutral rate
               within (dataset, stratum, base_correct=wrong)

    For each, returns (n_eligible, n_preserved, fraction).
    """
    out: list[dict] = []

    # we score adopt and df on a-arm (and df also on m-arm for amd contrast)
    for metric in ("adopt", "df"):
        for variant_id, sub in long_df[long_df["metric"] == metric].groupby("variant_id", dropna=False):
            n_id, d_id = variant_id.split("__")

            # S_wc: wrong > correct on a-arm cells
            arm_a = sub[(sub["cond_class"] == "a") & (sub["base_correct"].isin(["wrong", "correct"]))]
            wc_pivot = arm_a.pivot_table(
                index=["experiment", "dataset", "model", "stratum"],
                columns="base_correct",
                values="rate",
                aggfunc="first",
            ).dropna(subset=["wrong", "correct"])
            wc_n = len(wc_pivot)
            wc_preserved = int((wc_pivot["wrong"] > wc_pivot["correct"]).sum())

            # S_dist: S1 > S5 on a-arm wrong-base cells (E5b/E5c/E5d/E5e have strata)
            dist_src = sub[
                (sub["cond_class"] == "a")
                & (sub["base_correct"] == "wrong")
                & (sub["stratum"].isin(["S1", "S5"]))
            ]
            dist_pivot = dist_src.pivot_table(
                index=["experiment", "dataset", "model"],
                columns="stratum",
                values="rate",
                aggfunc="first",
            ).dropna(subset=["S1", "S5"])
            dist_n = len(dist_pivot)
            dist_preserved = int((dist_pivot["S1"] > dist_pivot["S5"]).sum())

            # S_amd: anchor > masked >= neutral on E5c wrong-base cells
            # neutral has no anchor → adopt/df undefined; only meaningful for adopt-on-anchor minus adopt-on-masked
            # For symmetry we just check anchor > masked (within wrong-base, e5c)
            amd_src = sub[
                sub["experiment"].isin(["experiment_e5c_vqa", "experiment_e5c_tally", "experiment_e5e_chartqa_full", "experiment_e5e_tallyqa_full"])
                & (sub["base_correct"] == "wrong")
                & sub["cond_class"].isin(["a", "m"])
            ]
            amd_pivot = amd_src.pivot_table(
                index=["experiment", "dataset", "model", "stratum"],
                columns="cond_class",
                values="rate",
                aggfunc="first",
            ).dropna(subset=["a", "m"])
            amd_n = len(amd_pivot)
            amd_preserved = int((amd_pivot["a"] > amd_pivot["m"]).sum())

            out.append({
                "metric": metric,
                "numerator": n_id,
                "denominator": d_id,
                "variant_id": variant_id,
                "S_wc_n": wc_n,
                "S_wc_preserved": wc_preserved,
                "S_wc_frac": wc_preserved / wc_n if wc_n else None,
                "S_dist_n": dist_n,
                "S_dist_preserved": dist_preserved,
                "S_dist_frac": dist_preserved / dist_n if dist_n else None,
                "S_amd_n": amd_n,
                "S_amd_preserved": amd_preserved,
                "S_amd_frac": amd_preserved / amd_n if amd_n else None,
                "score": (
                    (wc_preserved / wc_n if wc_n else 0)
                    + (dist_preserved / dist_n if dist_n else 0)
                    + (amd_preserved / amd_n if amd_n else 0)
                ) / 3.0,
            })
    return pd.DataFrame(out)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_inputs()
    print(f"[discover] {len(specs)} predictions.jsonl files")
    manifest = pd.DataFrame([
        {"experiment": s.experiment, "dataset": s.dataset, "model": s.model, "run": s.run, "path": str(s.path.relative_to(REPO_ROOT))}
        for s in specs
    ])
    manifest.to_csv(args.out_dir / "M2_inputs_manifest.csv", index=False)

    grouped = aggregate(specs)
    if grouped.empty:
        print("[aggregate] no records")
        return

    long_df = to_long(grouped)
    long_df.to_csv(args.out_dir / "M2_metric_variants_long.csv", index=False)
    grouped.to_csv(args.out_dir / "M2_metric_variants_wide.csv", index=False)

    sig = signal_preservation(long_df)
    sig = sig.sort_values(["metric", "score"], ascending=[True, False])
    sig.to_csv(args.out_dir / "M2_signal_preservation.csv", index=False)

    if args.print_summary:
        print()
        print(f"# Generated {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
        print(f"# Inputs: {len(specs)} predictions.jsonl from {manifest['experiment'].nunique()} experiments")
        print(f"# Total cells aggregated: {len(grouped)}")
        print()
        print("=== adopt variants ranked by mean signal-preservation ===")
        print(sig[sig["metric"] == "adopt"][[
            "variant_id", "S_wc_frac", "S_dist_frac", "S_amd_frac", "score"
        ]].to_string(index=False))
        print()
        print("=== df variants ranked by mean signal-preservation ===")
        print(sig[sig["metric"] == "df"][[
            "variant_id", "S_wc_frac", "S_dist_frac", "S_amd_frac", "score"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
