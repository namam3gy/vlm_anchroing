"""L1-L5 — confidence-modulated anchoring (§6 of the paper).

Test H7: ``direction_follow_rate`` and ``adopt_rate`` on the anchor arm
are monotonic with the *base* (target_only) prediction's answer-token
confidence — i.e. the wrong/correct binary in Phase A A1 is a coarse
projection of a continuous confidence-modulation effect.

Inputs: every ``predictions.jsonl`` that carries per-token logit capture
(post-commit ``5f925b2`` runs — E5b, E5c, E5d, E5e). The pre-logit VQAv2
main / strengthen runs are excluded; running them again under the new
runner would refresh the captured-logit coverage if §6 needs it.

Confidence proxies on the target_only row:
  - ``softmax_top1_prob``      = ``answer_token_probability``
  - ``top1_minus_top2_margin`` = top-1 logit − top-2 logit (from ``token_info``)
  - ``entropy_top_k``          = − Σ p log p over the captured top-k

For each (model, dataset, anchor stratum, condition class) cell we:
  1. Pair every anchor / mask row with its target_only row by sample_instance_id.
  2. Compute the three confidence proxies on the target_only side.
  3. Bucket records into quartiles within the cell (Q1 = most confident,
     Q4 = least).
  4. Report ``adopt_rate`` (M2 paired) and ``direction_follow_rate`` (M2)
     per quartile.

Outputs (under ``docs/insights/_data/``):
  - L1_confidence_quartile_long.csv     long-form one row per (cell, proxy, quartile)
  - L1_confidence_pair_records.csv      one row per (sample_instance, condition) with proxies
  - L1_proxy_monotonicity.csv           per-proxy monotonicity score (Spearman / sign-test)

The evidence narrative + recommendation lives in
``docs/insights/L1-confidence-modulation-evidence.md``.

No re-inference; reads only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterator

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "docs" / "insights" / "_data"


COND_RE = re.compile(
    r"target_plus_irrelevant_(?P<kind>number_masked|number|neutral)(?:_(?P<stratum>S\d))?$"
)


def normalize_condition(cond: str) -> tuple[str, str | None]:
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


@dataclass(frozen=True)
class InputSpec:
    experiment: str
    dataset: str
    model: str
    run: str
    path: Path


def discover_inputs() -> list[InputSpec]:
    layouts: list[tuple[str, str]] = [
        ("experiment_distance_vqa", "VQAv2"),
        ("experiment_distance_tally", "TallyQA"),
        ("experiment_e5c_vqa", "VQAv2"),
        ("experiment_e5c_tally", "TallyQA"),
        ("experiment_e5d_chartqa_validation", "ChartQA"),
        ("experiment_e5d_mathvista_validation", "MathVista"),
        ("experiment_e5e_chartqa_full", "ChartQA"),
        ("experiment_e5e_tallyqa_full", "TallyQA"),
        ("experiment_e5e_mathvista_full", "MathVista"),
        ("experiment_e7_plotqa_full", "PlotQA"),
        ("experiment_e7_infographicvqa_full", "InfographicVQA"),
    ]
    specs: list[InputSpec] = []
    for exp, ds in layouts:
        root = OUTPUTS / exp
        if not root.is_dir():
            continue
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
            # Pick largest run per (exp, model) — avoids pilot/smoke runs
            # under outputs/<exp>/<model>/ shadowing the canonical full run.
            candidates: list[tuple[int, Path, str]] = []
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != "analysis"):
                jsonl = run_dir / "predictions.jsonl"
                if not jsonl.is_file():
                    continue
                with jsonl.open() as f:
                    line = f.readline()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "answer_token_logit" not in sample:
                    continue
                # Count records to pick the canonical largest run.
                with jsonl.open() as f:
                    n_records = sum(1 for _ in f)
                candidates.append((n_records, jsonl, run_dir.name))
            if not candidates:
                continue
            candidates.sort()
            n_records, jsonl, run_name = candidates[-1]
            specs.append(InputSpec(experiment=exp, dataset=ds,
                                   model=model_dir.name, run=run_name, path=jsonl))
    return specs


# Proxy registry: (proxy_name, jsonl_field, direction)
# direction = "higher_more_confident" → Q1 = top quartile by descending sort
# direction = "lower_more_confident"  → Q1 = top quartile by ascending sort
# Fields with prefix "answer_span_*" are written by
# scripts/recompute_answer_span_confidence.py (post-hoc on token_info).
# Run that script before this one for the multi-proxy view to populate.
PROXIES: list[tuple[str, str, str]] = [
    # length-normalised entropy / cross-entropy (paper-clean: monotonic, length-invariant)
    ("cross_entropy",         "answer_span_cross_entropy",  "lower_more_confident"),
    ("geo_mean_prob",         "answer_span_geo_mean_prob",  "higher_more_confident"),
    ("log_prob_mean",         "answer_span_log_prob_mean",  "higher_more_confident"),
    # alternate-formulation proxies
    ("min_prob",              "answer_span_min_prob",       "higher_more_confident"),
    ("min_logit",             "answer_span_min_logit",      "higher_more_confident"),
    ("mean_logit",            "answer_span_mean_logit",     "higher_more_confident"),
    ("first_token_prob",      "answer_span_first_prob",     "higher_more_confident"),
    ("first_token_logit",     "answer_span_first_logit",    "higher_more_confident"),
    # length-biased / peak@1/e legacy formulations (kept for back-compat / comparison)
    ("entropy_top_k_legacy",  "answer_span_entropy_sum",    "lower_more_confident"),
    ("entropy_per_token",     "answer_span_entropy_mean",   "lower_more_confident"),
    ("log_prob_sum",          "answer_span_log_prob_sum",   "higher_more_confident"),
    # pre-recompute fallback (single-token only — present in older predictions
    # before recompute_answer_span_confidence.py was run)
    ("softmax_top1_prob_legacy", "answer_token_probability", "higher_more_confident"),
]
PROXY_NAMES = tuple(p[0] for p in PROXIES)


def confidence_proxies(row: dict) -> dict[str, float | None]:
    """Read multi-proxy answer-span confidence values from a preprocessed row.

    Expects predictions.jsonl to have been processed by
    `scripts/recompute_answer_span_confidence.py`. For records that
    pre-date that processing, only `softmax_top1_prob_legacy` (from the
    original `answer_token_probability` field) will be non-None.
    """
    out: dict[str, float | None] = {}
    for name, field, _direction in PROXIES:
        v = row.get(field)
        out[name] = float(v) if v is not None else None
    return out


def iter_paired_rows(spec: InputSpec) -> Iterator[dict]:
    """Yield one row per (sample_instance, anchor/mask arm) with target_only
    confidence proxies attached."""
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
        proxies = confidence_proxies(b_row)
        if all(v is None for v in proxies.values()):
            continue
        for (cls, stratum), x_row in conds.items():
            if cls in ("b", "d"):
                continue
            row = {
                "experiment": spec.experiment,
                "dataset": spec.dataset,
                "model": spec.model,
                "sample_instance_id": sid,
                "cond_class": cls,
                "stratum": stratum or "S0",
                "anchor_adopted": int(x_row.get("anchor_adopted") or 0),
                "anchor_direction_followed_moved": int(x_row.get("anchor_direction_followed_moved") or 0),
                "pred_b_equal_anchor": int(x_row.get("pred_b_equal_anchor") or 0),
                "numeric_distance_to_anchor": x_row.get("numeric_distance_to_anchor"),
                "anchor_value": x_row.get("anchor_value"),
                "ground_truth": x_row.get("ground_truth"),
                "prediction_b": b_row.get("prediction"),
                "prediction_x": x_row.get("prediction"),
                "exact_match_b": int(b_row.get("exact_match") or 0),
            }
            row.update({name: proxies[name] for name in PROXY_NAMES})
            yield row


def quartile_label(rank: int, n: int) -> str:
    if n == 0:
        return "Q?"
    q = (rank * 4) // n
    q = min(q, 3)
    return f"Q{q + 1}"  # Q1..Q4 with Q1 = highest confidence


def per_quartile_table(records: pd.DataFrame) -> pd.DataFrame:
    """Compute adopt and df rates per (cell, proxy, quartile).

    Cell = (experiment, dataset, model, cond_class, stratum). Quartiles
    are assigned along each proxy's confidence direction within each
    cell — Q1 is the most-confident quarter (so the proxy's `direction`
    determines whether to sort ascending or descending).
    """
    rows = []
    cell_keys = ["experiment", "dataset", "model", "cond_class", "stratum"]
    proxy_directions = {name: direction for name, _field, direction in PROXIES}

    for cell, sub in records.groupby(cell_keys, dropna=False):
        for proxy in PROXY_NAMES:
            if proxy not in sub.columns or sub[proxy].isna().all():
                continue
            direction = proxy_directions[proxy]
            # Q1 = most confident quarter; sort to put most-confident first.
            ascending = (direction == "lower_more_confident")
            sorted_sub = sub.sort_values(proxy, ascending=ascending, na_position="last").reset_index(drop=True)
            n = int(sorted_sub[proxy].notna().sum())
            if n == 0:
                continue
            sorted_sub["_quartile"] = None
            for i in range(n):
                sorted_sub.loc[i, "_quartile"] = quartile_label(i, n)
            for q in ("Q1", "Q2", "Q3", "Q4"):
                cell_q = sorted_sub[sorted_sub["_quartile"] == q]
                if cell_q.empty:
                    continue
                pb_ne_a = cell_q[cell_q["pred_b_equal_anchor"] == 0]
                num_anchor = cell_q[cell_q["numeric_distance_to_anchor"].notna()]
                row = dict(zip(cell_keys, cell))
                row.update({
                    "proxy": proxy,
                    "quartile": q,
                    "n": int(len(cell_q)),
                    "n_pb_ne_anchor": int(len(pb_ne_a)),
                    "n_numeric_anchor": int(len(num_anchor)),
                    "adopt_rate": (pb_ne_a["anchor_adopted"].sum() / len(pb_ne_a)) if len(pb_ne_a) else None,
                    "direction_follow_rate": (num_anchor["anchor_direction_followed_moved"].sum() / len(num_anchor))
                                             if len(num_anchor) else None,
                    "exact_match_b_in_quartile": cell_q["exact_match_b"].mean(),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def monotonicity_score(quartile_df: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, proxy) compute the trend across Q1 → Q4.

    Score: count of monotonic-decrease pairs out of (Q1>Q2, Q2>Q3, Q3>Q4)
    for both adopt_rate and direction_follow_rate. Q1 = most confident →
    expect *lower* anchor effect (anchor doesn't pull confident base);
    Q4 = least confident → expect *higher* anchor effect.

    A clean monotone increase from Q1 to Q4 = score 3/3.
    """
    rows = []
    cell_keys = ["experiment", "dataset", "model", "cond_class", "stratum", "proxy"]
    for cell, sub in quartile_df.groupby(cell_keys, dropna=False):
        sub = sub.sort_values("quartile")
        if len(sub) < 4:
            continue
        adopt = sub.set_index("quartile")["adopt_rate"].to_dict()
        df = sub.set_index("quartile")["direction_follow_rate"].to_dict()
        order = ["Q1", "Q2", "Q3", "Q4"]
        if not all(q in adopt for q in order):
            continue

        adopt_pairs = [(adopt[order[i]], adopt[order[i + 1]]) for i in range(3)]
        df_pairs = [(df[order[i]], df[order[i + 1]]) for i in range(3)]
        adopt_inc = sum(1 for a, b in adopt_pairs if a is not None and b is not None and b > a)
        df_inc = sum(1 for a, b in df_pairs if a is not None and b is not None and b > a)

        adopt_q1 = adopt.get("Q1")
        adopt_q4 = adopt.get("Q4")
        df_q1 = df.get("Q1")
        df_q4 = df.get("Q4")

        rows.append({
            **dict(zip(cell_keys, cell)),
            "adopt_increases": adopt_inc,
            "df_increases": df_inc,
            "adopt_q1": adopt_q1,
            "adopt_q4": adopt_q4,
            "adopt_q4_minus_q1": (adopt_q4 - adopt_q1) if (adopt_q4 is not None and adopt_q1 is not None) else None,
            "df_q1": df_q1,
            "df_q4": df_q4,
            "df_q4_minus_q1": (df_q4 - df_q1) if (df_q4 is not None and df_q1 is not None) else None,
        })
    return pd.DataFrame(rows)


def proxy_comparison_table(mono: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monotonicity statistics per proxy, anchor-arm only.

    Used to choose the §6 primary proxy for the paper: highest mean
    adopt/df Q4-Q1 gap + most cells fully-monotone (3/3).
    """
    rows = []
    for proxy in PROXY_NAMES:
        sub = mono[(mono["proxy"] == proxy) & (mono["cond_class"] == "a")]
        n = len(sub)
        if n == 0:
            rows.append({"proxy": proxy, "n_cells": 0})
            continue
        rows.append({
            "proxy": proxy,
            "n_cells": n,
            "mean_adopt_q4_minus_q1": float(sub["adopt_q4_minus_q1"].mean()),
            "mean_df_q4_minus_q1": float(sub["df_q4_minus_q1"].mean()),
            "adopt_fully_monotone_3of3": int((sub["adopt_increases"] == 3).sum()),
            "df_fully_monotone_3of3": int((sub["df_increases"] == 3).sum()),
            "adopt_at_least_2of3": int((sub["adopt_increases"] >= 2).sum()),
            "df_at_least_2of3": int((sub["df_increases"] >= 2).sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--primary-proxy", default="cross_entropy",
                        help="Primary proxy used in --print-summary headline.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_inputs()
    print(f"[discover] {len(specs)} predictions.jsonl with logit fields")

    pair_rows: list[dict] = []
    for spec in specs:
        for row in iter_paired_rows(spec):
            pair_rows.append(row)

    if not pair_rows:
        print("[no pairs] aborting")
        return

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(args.out_dir / "L1_confidence_pair_records.csv", index=False)
    print(f"[pair] {len(pair_df)} (sample × arm) records")

    quartile_df = per_quartile_table(pair_df)
    quartile_df.to_csv(args.out_dir / "L1_confidence_quartile_long.csv", index=False)

    mono = monotonicity_score(quartile_df)
    mono.to_csv(args.out_dir / "L1_proxy_monotonicity.csv", index=False)

    proxy_cmp = proxy_comparison_table(mono)
    proxy_cmp.to_csv(args.out_dir / "L1_proxy_comparison.csv", index=False)

    if args.print_summary:
        print()
        print("=== Per-proxy monotonicity score (anchor-arm cells; Q4 = least confident) ===")
        for _, r in proxy_cmp.sort_values("mean_df_q4_minus_q1", ascending=False).iterrows():
            n = int(r.get("n_cells") or 0)
            if n == 0:
                print(f"  {r['proxy']:28s}: no cells")
                continue
            print(f"  {r['proxy']:28s}: n={n:3d}  "
                  f"adopt(Q4-Q1)={r['mean_adopt_q4_minus_q1']:+.4f}  "
                  f"df(Q4-Q1)={r['mean_df_q4_minus_q1']:+.4f}  "
                  f"adopt 3/3 monotone={int(r['adopt_fully_monotone_3of3'])}/{n}  "
                  f"df 3/3 monotone={int(r['df_fully_monotone_3of3'])}/{n}")

        primary = args.primary_proxy
        print()
        print(f"=== Per-cell adopt Q1 vs Q4 on primary proxy ({primary}) ===")
        sub = mono[(mono["proxy"] == primary) & (mono["cond_class"] == "a")]
        for _, r in sub.iterrows():
            print(f"  {r['experiment']:42s}  {r['dataset']:14s}  {r['model']:30s}  "
                  f"{r['stratum']:3s}  adopt Q1={_fmt(r['adopt_q1'])} Q4={_fmt(r['adopt_q4'])} (Δ={_fmt_signed(r['adopt_q4_minus_q1'])})  "
                  f"df Q1={_fmt(r['df_q1'])} Q4={_fmt(r['df_q4'])} (Δ={_fmt_signed(r['df_q4_minus_q1'])})")


def _fmt(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.4f}"


def _fmt_signed(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:+.4f}"


if __name__ == "__main__":
    main()
