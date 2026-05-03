"""Build the 5-dataset main-matrix headline table (§3.3 + §5).

Reads per_cell.csv files for tallyqa, chartqa, mathvista (E5e) and plotqa,
infographicvqa (E7) and emits two artifacts:

  docs/insights/_data/main_panel_5dataset_per_cell.csv
    long format: model × dataset × {a-arm, m-arm} × {adopt, df, em}
  docs/insights/_data/main_panel_5dataset_summary.md
    markdown table for paper headline (3 models × 5 datasets, S1
    wrong-base; columns adopt(a)/adopt(m)/df(a)/df(m)/em(a)/em(m)).

Wrong-base S1 cells only (cond_class in {a,m}, stratum='S1',
base_correct=False) — paper canonical per references/project.md §0.4.3.

Usage:
  uv run python scripts/build_e5e_e7_5dataset_summary.py
  uv run python scripts/build_e5e_e7_5dataset_summary.py --print
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs" / "insights" / "_data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

DATASET_SOURCES = [
    ("TallyQA",         "experiment_e5e_tallyqa_full"),
    ("ChartQA",         "experiment_e5e_chartqa_full"),
    ("MathVista",       "experiment_e5e_mathvista_full"),
    ("PlotQA",          "experiment_e7_plotqa_full"),
    ("InfographicVQA",  "experiment_e7_infographicvqa_full"),
]
MODEL_DISPLAY = {
    "llava-next-interleaved-7b": "llava-interleave-7b",
    "qwen2.5-vl-7b-instruct":    "qwen2.5-vl-7b",
    "gemma3-27b-it":             "gemma3-27b",
}
MODEL_ORDER = list(MODEL_DISPLAY)


def _load_per_cell(exp_dir: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{exp_dir}_per_cell.csv"
    if not p.exists():
        print(f"[warn] {p.name} missing; run analyze_e5e_wrong_correct.py --exp-dir {exp_dir}")
        return None
    return pd.read_csv(p)


def _pick_canonical_run(df: pd.DataFrame) -> pd.DataFrame:
    """For each model, keep rows from the run with the largest total n.

    Phase A scripts: pick largest run, not alphabetically-latest.
    """
    keep_runs: dict[str, str] = {}
    for model, sub in df.groupby("model", sort=False):
        # Sum n across all (cond, stratum, base_correct) rows per run; pick max.
        per_run_totals = sub.groupby("run")["n"].sum().sort_values()
        keep_runs[model] = per_run_totals.index[-1]
    mask = df.apply(lambda r: r["run"] == keep_runs.get(r["model"]), axis=1)
    return df[mask].copy()


def _wrong_base_s1(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["stratum"] == "S1")
        & (df["base_correct"] == False)
        & (df["cond_class"].isin(["a", "m"]))
    ].copy()


def _supplementary_acc_b(exp_dir: str, model: str, run: str) -> dict:
    """Mean PlotQA-relaxed (5%) and InfoVQA-ANLS (>=0.5) on target_only.

    Returns {} if the predictions.jsonl is missing or doesn't have the
    fields (i.e., recompute_answer_span_confidence.py hasn't been run).
    """
    p = OUTPUTS_ROOT / exp_dir / model / run / "predictions.jsonl"
    if not p.exists():
        return {}
    out: dict = {}
    relaxed_vals: list[float] = []
    anls_vals: list[float] = []
    em_strict_vals: list[int] = []
    n_b = 0
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("condition") != "target_only":
            continue
        n_b += 1
        em_strict_vals.append(int(r.get("exact_match") or 0))
        rv = r.get("plotqa_relaxed_correct")
        if rv is not None:
            relaxed_vals.append(float(rv))
        av = r.get("infovqa_anls")
        if av is not None:
            anls_vals.append(float(av))
    if n_b == 0:
        return {}
    out["acc_b_strict"] = sum(em_strict_vals) / max(len(em_strict_vals), 1)
    if relaxed_vals:
        out["acc_b_plotqa_relaxed_5pct"] = sum(relaxed_vals) / len(relaxed_vals)
    if anls_vals:
        out["acc_b_infovqa_anls_05"] = sum(anls_vals) / len(anls_vals)
    out["n_b"] = n_b
    return out


def build_long_table() -> pd.DataFrame:
    rows: list[dict] = []
    for ds_name, exp_dir in DATASET_SOURCES:
        df = _load_per_cell(exp_dir)
        if df is None or df.empty:
            continue
        df = _pick_canonical_run(df)
        df = _wrong_base_s1(df)
        for _, r in df.iterrows():
            row = {
                "dataset":      ds_name,
                "exp_dir":      exp_dir,
                "model":        r["model"],
                "run":          r["run"],
                "cond_class":   r["cond_class"],
                "n":            int(r["n"]),
                "n_pb_ne_anchor": int(r.get("n_pb_ne_anchor") or 0),
                "n_numeric_anchor": int(r.get("n_numeric_anchor") or 0),
                "adopt_M2":     float(r["adopt_M2"]),
                "direction_follow_M2": float(r["direction_follow_M2"]),
                "exact_match":  float(r["exact_match"]),
            }
            # Attach base-condition supplementary accuracies (computed from
            # predictions.jsonl, requires recompute_answer_span_confidence.py
            # to have run first for plotqa_relaxed_correct / infovqa_anls.)
            row.update(_supplementary_acc_b(exp_dir, r["model"], r["run"]))
            rows.append(row)
    long_df = pd.DataFrame(rows)
    return long_df


def build_summary_md(long_df: pd.DataFrame) -> str:
    pieces: list[str] = []
    pieces.append("# 5-dataset main-matrix headline (S1 wrong-base, C-form)")
    pieces.append("")
    pieces.append("Auto-generated by `scripts/build_e5e_e7_5dataset_summary.py`.")
    pieces.append("Per-cell source: `docs/insights/_data/{exp}_per_cell.csv`.")
    pieces.append("")
    pieces.append("Strict exact_match is the canonical em metric across all 5 datasets")
    pieces.append("(cross-dataset comparability + clean alignment with adopt/df definitions).")
    pieces.append("PlotQA-relaxed (5%) and InfoVQA-ANLS (>=0.5) supplementary acc(b) values")
    pieces.append("are reported in parentheses for reference where applicable, matching the")
    pieces.append("dataset's own official metric (Methani 2020 / Mathew 2022).")
    pieces.append("")
    pieces.append("| dataset | model | n | acc(b) strict | acc(b) supp. | adopt(a) | df(a) | adopt(m) | df(m) | em(a) | em(m) |")
    pieces.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds_name, _ in DATASET_SOURCES:
        sub = long_df[long_df["dataset"] == ds_name]
        if sub.empty:
            continue
        for model in MODEL_ORDER:
            ms = sub[sub["model"] == model]
            if ms.empty:
                continue
            a = ms[ms["cond_class"] == "a"].iloc[0] if not ms[ms["cond_class"] == "a"].empty else None
            m = ms[ms["cond_class"] == "m"].iloc[0] if not ms[ms["cond_class"] == "m"].empty else None
            n = int(a["n"]) if a is not None else (int(m["n"]) if m is not None else 0)
            adopt_a = f"{a['adopt_M2']:.3f}" if a is not None else "—"
            df_a    = f"{a['direction_follow_M2']:.3f}" if a is not None else "—"
            em_a    = f"{a['exact_match']:.3f}" if a is not None else "—"
            adopt_m = f"{m['adopt_M2']:.3f}" if m is not None else "—"
            df_m    = f"{m['direction_follow_M2']:.3f}" if m is not None else "—"
            em_m    = f"{m['exact_match']:.3f}" if m is not None else "—"
            # Base-cond accuracy (strict + supplementary); pull from a row
            # since both a and m share the same target_only b records.
            ref = a if a is not None else m
            acc_b_strict = ref.get("acc_b_strict") if ref is not None else None
            acc_b_strict_str = f"{acc_b_strict:.3f}" if acc_b_strict is not None and not pd.isna(acc_b_strict) else "—"
            relaxed = ref.get("acc_b_plotqa_relaxed_5pct") if ref is not None else None
            anls = ref.get("acc_b_infovqa_anls_05") if ref is not None else None
            supp_str = "—"
            if relaxed is not None and not pd.isna(relaxed):
                supp_str = f"{relaxed:.3f} (relaxed-5%)"
            elif anls is not None and not pd.isna(anls):
                supp_str = f"{anls:.3f} (ANLS≥0.5)"
            pieces.append(f"| {ds_name} | {MODEL_DISPLAY[model]} | {n} | "
                          f"{acc_b_strict_str} | {supp_str} | "
                          f"{adopt_a} | {df_a} | {adopt_m} | {df_m} | {em_a} | {em_m} |")
        pieces.append("")
    return "\n".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true",
                    help="Print the markdown summary to stdout in addition to writing.")
    ap.add_argument("--out-csv",
                    default=str(DATA_DIR / "main_panel_5dataset_per_cell.csv"))
    ap.add_argument("--out-md",
                    default=str(DATA_DIR / "main_panel_5dataset_summary.md"))
    args = ap.parse_args()

    long_df = build_long_table()
    if long_df.empty:
        print("[err] no rows produced — check that per_cell.csv files exist for all 5 datasets")
        sys.exit(1)
    long_df.to_csv(args.out_csv, index=False)
    print(f"[wrote] {args.out_csv} ({len(long_df)} rows)")

    md = build_summary_md(long_df)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"[wrote] {args.out_md}")

    if args.print:
        print()
        print(md)


if __name__ == "__main__":
    main()
