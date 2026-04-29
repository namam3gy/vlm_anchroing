"""analyze_e6_subspace.py — analyze sweep-subspace predictions (Method 1).

Computes M2 metrics (adopt_rate, direction_follow_rate, exact_match) per
(layer, K, alpha) cell vs. baseline. Applies 5% relative df reduction
threshold for Method 1 selection (looser than Phase 1's 10%).

Usage:
  uv run python scripts/analyze_e6_subspace.py \\
      --sweep-dir outputs/e6_steering/llava-next-interleaved-7b/sweep_subspace_tally_pooled

  # with explicit output dir
  uv run python scripts/analyze_e6_subspace.py \\
      --sweep-dir <path> --out-dir <outpath>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DF_REL_THRESHOLD = 0.05   # ≥5% relative df reduction to qualify
EM_PP_TOLERANCE = 2.0     # exact_match must stay within ±2pp


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="Directory containing predictions.jsonl from sweep-subspace.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir for CSVs. Defaults to <sweep-dir>/_analysis.")
    return ap.parse_args()


def _load_records(sweep_dir: Path) -> list[dict]:
    p = sweep_dir / "predictions.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"predictions.jsonl not found at {p}")
    records = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[load] {len(records)} records from {p}")
    return records


def _try_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _compute_metrics(group: list[dict]) -> dict:
    """M2 canonical metrics over a group sharing the same (cell, dataset).

    Pairs sids on target_only (b) and target_plus_irrelevant_number_S1 (a).
    C-form direction_follow: (pa-pb)·(anchor-pb) > 0 AND pa != pb.
    """
    by_sid: dict[str, dict] = defaultdict(dict)
    for r in group:
        by_sid[r["sample_instance_id"]][r["condition"]] = r

    adopt_num = adopt_den = 0
    df_num = df_den = 0
    em_num = em_den = 0

    for _sid, conds in by_sid.items():
        r_b = conds.get("target_only")
        r_a = conds.get("target_plus_irrelevant_number_S1")
        if r_b is None or r_a is None:
            continue

        pb = _try_float(r_b.get("parsed_number"))
        pa = _try_float(r_a.get("parsed_number"))
        anchor = _try_float(r_a.get("anchor_value"))
        gt = _try_float(r_b.get("ground_truth"))

        if pb is not None and pa is not None and anchor is not None:
            # adopt_rate: pa == anchor AND pb != anchor
            if pb != anchor:
                adopt_den += 1
                if pa == anchor:
                    adopt_num += 1
            # direction_follow (C-form): (pa-pb)·(anchor-pb) > 0 AND pa != pb
            df_den += 1
            if pa != pb and (pa - pb) * (anchor - pb) > 0:
                df_num += 1

        # exact_match on a-arm
        if pa is not None and gt is not None:
            em_den += 1
            try:
                if int(pa) == int(gt):
                    em_num += 1
            except (ValueError, TypeError):
                pass

    return {
        "adopt_rate": adopt_num / adopt_den if adopt_den else None,
        "direction_follow_rate": df_num / df_den if df_den else None,
        "exact_match": em_num / em_den if em_den else None,
        "adopt_num": adopt_num, "adopt_den": adopt_den,
        "df_num": df_num, "df_den": df_den,
        "em_num": em_num, "em_den": em_den,
    }


def main():
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = PROJECT_ROOT / sweep_dir
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(sweep_dir)

    # Group by cell label
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cell[r["cell_label"]].append(r)

    baseline_metrics: dict | None = None
    cell_summaries: list[dict] = []

    for label, group in sorted(by_cell.items()):
        if not group:
            continue
        r0 = group[0]
        L = int(r0.get("cell_layer", -1))
        K = int(r0.get("subspace_K", 0))
        alpha = float(r0.get("cell_alpha", 0.0))
        m = _compute_metrics(group)
        entry = {
            "cell_label": label, "layer": L, "K": K, "alpha": alpha,
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in m.items()},
            "n_records": len(group),
        }
        cell_summaries.append(entry)
        if label == "baseline":
            baseline_metrics = m

    if not cell_summaries:
        print("[warn] no cells found in predictions.jsonl")
        return

    if baseline_metrics is None:
        print("[warn] 'baseline' cell not found — skipping relative comparisons")
        baseline_df = None
        baseline_em = None
    else:
        baseline_df = baseline_metrics.get("direction_follow_rate")
        baseline_em = baseline_metrics.get("exact_match")

    passing = []
    for entry in cell_summaries:
        if entry["cell_label"] == "baseline":
            entry["df_rel_change"] = 0.0
            entry["em_pp_change"] = 0.0
            entry["passes"] = False
            continue
        df = entry.get("direction_follow_rate")
        em = entry.get("exact_match")
        if df is not None and baseline_df and baseline_df > 0:
            df_rel = (df - baseline_df) / baseline_df
        else:
            df_rel = None
        if em is not None and baseline_em is not None:
            em_pp = (em - baseline_em) * 100
        else:
            em_pp = None
        entry["df_rel_change"] = round(df_rel, 6) if df_rel is not None else None
        entry["em_pp_change"] = round(em_pp, 4) if em_pp is not None else None
        passes = (
            df_rel is not None and df_rel <= -DF_REL_THRESHOLD
            and em_pp is not None and abs(em_pp) <= EM_PP_TOLERANCE
        )
        entry["passes"] = passes
        if passes:
            passing.append(entry)

    # Save CSV
    fieldnames = [
        "cell_label", "layer", "K", "alpha",
        "direction_follow_rate", "adopt_rate", "exact_match",
        "df_rel_change", "em_pp_change", "passes",
        "adopt_num", "adopt_den", "df_num", "df_den", "em_num", "em_den",
        "n_records",
    ]
    summary_path = out_dir / "cell_summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(cell_summaries, key=lambda x: x["cell_label"]))
    print(f"[save] {summary_path}  ({len(cell_summaries)} cells)")

    # Print report
    if baseline_df is not None:
        print(f"\n=== Baseline: df={baseline_df:.4f}  em={baseline_em:.4f} ===")
    print(f"Cells passing (≥{DF_REL_THRESHOLD*100:.0f}% rel df drop, em ±{EM_PP_TOLERANCE}pp): "
          f"{len(passing)}")
    if passing:
        best = min(passing, key=lambda x: x.get("direction_follow_rate") or 1.0)
        print(f"Best: {best['cell_label']}  "
              f"df={best['direction_follow_rate']:.4f} "
              f"(Δ={best['df_rel_change']*100:.1f}%)  "
              f"em={best['exact_match']:.4f} "
              f"(Δ={best['em_pp_change']:+.2f}pp)")
        (out_dir / "best_cell.json").write_text(json.dumps(best, indent=2))
    else:
        ranked = sorted(
            [e for e in cell_summaries
             if e["cell_label"] != "baseline" and e.get("df_rel_change") is not None],
            key=lambda x: x["df_rel_change"]
        )[:5]
        print("No passing cells. Top 5 by df reduction:")
        for e in ranked:
            print(f"  {e['cell_label']}: df={e['direction_follow_rate']:.4f} "
                  f"(Δ={e['df_rel_change']*100:.1f}%)  "
                  f"em_pp={e['em_pp_change']:+.2f}")

    print(f"\n[done] analysis in {out_dir}")


if __name__ == "__main__":
    main()
