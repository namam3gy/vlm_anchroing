"""E5e-style 4-condition (b/a/m/d) wrong-base / correct-base split.

For an E5e run dir (single-stratum b/a/m/d on a single dataset × N models),
walks predictions.jsonl and computes, per (model, condition, base_correct),
M2 adopt + direction-follow + accuracy. Anchor − masked − neutral gaps
are also reported.

Inputs are auto-discovered: every `predictions.jsonl` under
`outputs/<exp_dir>/<model>/<run>/`. Pass `--exp-dir` to restrict
(e.g. only γ-α MathVista).

Outputs: a single CSV under `docs/insights/_data/<exp_label>_per_cell.csv`
and a summary table printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    return (
        {"number": "a", "number_masked": "m", "neutral": "d"}.get(m.group("kind"), "?"),
        m.group("stratum"),
    )


def is_int_str(s: str | None) -> bool:
    return bool(s) and str(s).lstrip("-").isdigit()


def aggregate_run(jsonl_path: Path) -> pd.DataFrame:
    """One row per (cond_class, stratum, base_correct): n + adopt/df/em rates."""
    by_sid: dict[str, dict] = defaultdict(dict)
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("sample_instance_id")
            if sid is None:
                continue
            cls, stratum = normalize_condition(r.get("condition", ""))
            if cls == "?":
                continue
            by_sid[sid][(cls, stratum)] = r

    rows = []
    for sid, conds in by_sid.items():
        b_row = conds.get(("b", None))
        if b_row is None:
            continue
        pb = (b_row.get("prediction") or "").strip()
        gt = (b_row.get("ground_truth") or "").strip()
        # base_correct using normalized comparison (best-effort)
        base_correct = False
        if is_int_str(pb) and is_int_str(gt) and pb == gt:
            base_correct = True
        for (cls, stratum), x_row in conds.items():
            if cls == "b":
                continue
            rows.append({
                "cond_class": cls,
                "stratum": stratum or "S0",
                "base_correct": base_correct,
                "anchor_adopted": int(x_row.get("anchor_adopted") or 0),
                "anchor_direction_followed_moved": int(x_row.get("anchor_direction_followed_moved") or 0),
                "exact_match": int(x_row.get("exact_match") or 0),
                "pred_b_equal_anchor": int(x_row.get("pred_b_equal_anchor") or 0),
                "anchor_present": x_row.get("anchor_value") is not None,
                "numeric_pair_with_anchor": x_row.get("numeric_distance_to_anchor") is not None,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Aggregate per (cls, stratum, base_correct)
    grouped = (
        df.groupby(["cond_class", "stratum", "base_correct"], dropna=False)
          .apply(lambda sub: pd.Series({
              "n": len(sub),
              "n_pb_ne_anchor": int((sub["pred_b_equal_anchor"] == 0).sum()),
              "n_numeric_anchor": int(sub["numeric_pair_with_anchor"].sum()),
              "adopt_M2": (
                  sub["anchor_adopted"].sum() / max(1, (sub["pred_b_equal_anchor"] == 0).sum())
              ),
              "direction_follow_M2": (
                  sub["anchor_direction_followed_moved"].sum() /
                  max(1, int(sub["numeric_pair_with_anchor"].sum()))
              ),
              "exact_match": sub["exact_match"].mean(),
          }))
          .reset_index()
    )
    return grouped


def discover_runs(exp_dir: str | None) -> list[Path]:
    root = REPO_ROOT / "outputs"
    if exp_dir:
        roots = [root / exp_dir]
    else:
        roots = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("experiment_e5e")]
    out: list[Path] = []
    for r in roots:
        for path in r.rglob("predictions.jsonl"):
            if "analysis" in path.parts or "_logs" in path.parts:
                continue
            out.append(path)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=str, default="experiment_e5e_mathvista_full")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    runs = discover_runs(args.exp_dir)
    if not runs:
        print(f"[discover] no runs under outputs/{args.exp_dir}/")
        return
    print(f"[discover] {len(runs)} run dirs")

    out_rows: list[pd.DataFrame] = []
    for run in runs:
        model = run.parts[-3]
        run_id = run.parts[-2]
        df = aggregate_run(run)
        if df.empty:
            continue
        df["model"] = model
        df["run"] = run_id
        df["exp_dir"] = args.exp_dir
        out_rows.append(df)

    if not out_rows:
        print("[aggregate] no records")
        return
    final = pd.concat(out_rows, ignore_index=True)

    out_csv = args.out_csv or (DATA_DIR / f"{args.exp_dir}_per_cell.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}")

    if args.print_summary:
        print()
        print("=== Per (model, condition × stratum × base_correct) summary ===")
        cols = ["model", "cond_class", "stratum", "base_correct", "n", "adopt_M2", "direction_follow_M2", "exact_match"]
        print(final[cols].to_string(index=False))

        print()
        print("=== Anchor − masked − neutral gaps (adopt_M2, on wrong-base subset) ===")
        wrong = final[final["base_correct"] == False]
        for model, mb in wrong.groupby("model"):
            piv = mb.pivot_table(index=["stratum"], columns="cond_class", values="adopt_M2", aggfunc="first")
            print(f"--- {model} ---")
            print(piv.to_string())
            if "a" in piv.columns and "m" in piv.columns:
                print(f"  a − m: {(piv['a'] - piv['m']).to_dict()}")


if __name__ == "__main__":
    main()
