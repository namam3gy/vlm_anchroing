"""Aggregate sweep_subspace pilot grid → rank cells by (Δdf, Δem) trade-off.

Inputs:
  --plotqa-dir / --infovqa-dir: pilot grid sweep outputs (predictions.jsonl)
    each containing records of form
        { sample_instance_id, condition, cell_layer, subspace_K, cell_alpha,
          parsed_number, exact_match, anchor_value, ground_truth, ... }
  --baseline-plotqa / --baseline-infovqa: paths to the model's baseline
    run dir(s); baseline (target_only, target_plus_irrelevant_number_S1,
    target_plus_irrelevant_neutral) predictions are read from
    predictions.jsonl in the latest timestamped subdir.

Per cell × dataset metrics:
  - df(a)   sweep-arm direction-follow rate (C-form: (pa-pb)*(anchor-pb)>0 AND pa!=pb)
  - em(a)   sweep-arm exact-match rate vs ground_truth
  - acc(b)  baseline target_only exact-match (from baseline preds)
  Δdf(a) = sweep_df(a) − baseline_df(a)            (negative = mitigation works)
  Δem(a) = sweep_em(a) − baseline_em(a)            (positive = better)

Cell selection rule (per memory feedback_em_drop_dealbreaker):
  - Reject any cell with Δem(a) ≤ −em_drop_deal_breaker  on EITHER dataset
  - Of remaining, rank by combined |Δdf(a)| reduction (more negative = better)

Outputs CSV with all cells × datasets:
  cell_label, layer, K, alpha, dataset,
  n_eligible, df_baseline, df_sweep, dDF, em_baseline, em_sweep, dEM,
  rejected (bool), reason
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--plotqa-dir", required=True)
    p.add_argument("--infovqa-dir", required=True)
    p.add_argument("--baseline-plotqa", required=True,
                   help="outputs/experiment_e7_plotqa_full/<model>/")
    p.add_argument("--baseline-infovqa", required=True,
                   help="outputs/experiment_e7_infographicvqa_full/<model>/")
    p.add_argument("--em-drop-deal-breaker", type=float, default=0.06,
                   help="Reject cell if Δem(a) ≤ -threshold on either dataset.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def _latest_baseline_run(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    runs = []
    for ts in model_dir.iterdir():
        if not ts.is_dir():
            continue
        f = ts / "predictions.jsonl"
        if f.exists():
            runs.append((f.stat().st_size, ts))
    if not runs:
        return None
    return max(runs)[1]


def _load_baseline_metrics(baseline_run: Path) -> dict[str, dict]:
    """Compute baseline df(a), em(a), acc(b) from a model's predictions.jsonl."""
    if baseline_run is None:
        return {}
    rows = []
    for line in (baseline_run / "predictions.jsonl").open():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    by_sid_cond: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    a_conds = ("target_plus_irrelevant_number", "target_plus_irrelevant_number_S1")
    df_count = df_total = 0
    em_a_count = em_a_total = 0
    em_b_count = em_b_total = 0
    for sid, conds in by_sid_cond.items():
        b_rec = conds.get("target_only")
        if b_rec is None:
            continue
        em_b_total += 1
        if b_rec.get("exact_match") == 1:
            em_b_count += 1
        a_rec = next((conds[c] for c in a_conds if c in conds), None)
        if a_rec is None:
            continue
        em_a_total += 1
        if a_rec.get("exact_match") == 1:
            em_a_count += 1
        # df: (pa - pb) * (anchor - pb) > 0 AND pa != pb
        try:
            pa = a_rec.get("prediction") or a_rec.get("parsed_number")
            pb = b_rec.get("prediction") or b_rec.get("parsed_number")
            anchor = a_rec.get("anchor_value")
            pa_i = int(pa) if pa not in (None, "") else None
            pb_i = int(pb) if pb not in (None, "") else None
            a_i = int(anchor) if anchor not in (None, "") else None
        except (TypeError, ValueError):
            continue
        if pa_i is None or pb_i is None or a_i is None:
            continue
        df_total += 1
        if pa_i != pb_i and (pa_i - pb_i) * (a_i - pb_i) > 0:
            df_count += 1
    return {
        "df_baseline": df_count / df_total if df_total else 0.0,
        "em_a_baseline": em_a_count / em_a_total if em_a_total else 0.0,
        "em_b_baseline": em_b_count / em_b_total if em_b_total else 0.0,
        "n_df_eligible": df_total,
    }


def _load_sweep_metrics(pilot_dir: Path) -> dict[tuple[int, int, float], dict]:
    """Group sweep predictions by (cell_layer, subspace_K, cell_alpha) and
    compute df(a), em(a). Each sweep record corresponds to one (sid, cond,
    cell) intervention. We need per-cell df/em on the anchor arm.
    """
    if not pilot_dir.exists():
        return {}
    pred = pilot_dir / "predictions.jsonl"
    if not pred.exists():
        return {}
    rows = [json.loads(l) for l in pred.open() if l.strip()]
    # Group by (cell, sid) to pair conds within same cell.
    by_cell: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cell = (int(r["cell_layer"]), int(r["subspace_K"]), float(r["cell_alpha"]))
        by_cell[cell][r["sample_instance_id"]][r["condition"]] = r

    out: dict[tuple, dict] = {}
    for cell, by_sid in by_cell.items():
        em_a_total = em_a_count = 0
        df_total = df_count = 0
        a_conds = ("target_plus_irrelevant_number", "target_plus_irrelevant_number_S1")
        for sid, conds in by_sid.items():
            b_rec = conds.get("target_only")
            a_rec = next((conds[c] for c in a_conds if c in conds), None)
            if a_rec is None or b_rec is None:
                continue
            em_a_total += 1
            if a_rec.get("exact_match") == 1:
                em_a_count += 1
            try:
                pa = a_rec.get("parsed_number")
                pb = b_rec.get("parsed_number")
                anchor = a_rec.get("anchor_value")
                pa_i = int(pa) if pa not in (None, "") else None
                pb_i = int(pb) if pb not in (None, "") else None
                a_i = int(anchor) if anchor not in (None, "") else None
            except (TypeError, ValueError):
                continue
            if pa_i is None or pb_i is None or a_i is None:
                continue
            df_total += 1
            if pa_i != pb_i and (pa_i - pb_i) * (a_i - pb_i) > 0:
                df_count += 1
        out[cell] = {
            "df_sweep": df_count / df_total if df_total else 0.0,
            "em_a_sweep": em_a_count / em_a_total if em_a_total else 0.0,
            "n_df_eligible": df_total,
        }
    return out


def main() -> None:
    args = parse_args()

    plotqa_baseline = _load_baseline_metrics(
        _latest_baseline_run(Path(args.baseline_plotqa))
    )
    infovqa_baseline = _load_baseline_metrics(
        _latest_baseline_run(Path(args.baseline_infovqa))
    )

    plotqa_sweep = _load_sweep_metrics(Path(args.plotqa_dir))
    infovqa_sweep = _load_sweep_metrics(Path(args.infovqa_dir))

    rows: list[dict] = []
    for ds_label, sweep_dict, baseline in (
        ("plotqa", plotqa_sweep, plotqa_baseline),
        ("infographicvqa", infovqa_sweep, infovqa_baseline),
    ):
        for (L, K, alpha), sw in sweep_dict.items():
            d_df = sw["df_sweep"] - baseline.get("df_baseline", 0.0)
            d_em = sw["em_a_sweep"] - baseline.get("em_a_baseline", 0.0)
            rejected = d_em <= -args.em_drop_deal_breaker
            rows.append({
                "dataset": ds_label,
                "layer": L,
                "K": K,
                "alpha": alpha,
                "cell_label": f"L{L:02d}_K{K:02d}_a{alpha:.1f}",
                "n_eligible_df": sw["n_df_eligible"],
                "df_baseline": baseline.get("df_baseline"),
                "df_sweep": sw["df_sweep"],
                "dDF": d_df,
                "em_a_baseline": baseline.get("em_a_baseline"),
                "em_a_sweep": sw["em_a_sweep"],
                "dEM": d_em,
                "rejected": rejected,
                "reject_reason": ("em drop "
                                   f">{args.em_drop_deal_breaker:.0%}"
                                   if rejected else ""),
            })

    df = pd.DataFrame(rows).sort_values(["cell_label", "dataset"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[done] wrote {len(df)} rows to {out}")

    # ── Per-cell pooled summary (avg over datasets, with reject if either rejected) ──
    pooled = (
        df.groupby(["cell_label", "layer", "K", "alpha"])
        .agg(
            mean_dDF=("dDF", "mean"),
            mean_dEM=("dEM", "mean"),
            any_rejected=("rejected", "any"),
        )
        .reset_index()
        .sort_values("mean_dDF")  # most negative dDF first (best mitigation)
    )
    pooled_path = out.with_name(out.stem + "_pooled.csv")
    pooled.to_csv(pooled_path, index=False)
    print(f"[done] wrote pooled ranking to {pooled_path}")
    print()
    print("=== Pooled cell ranking (sorted by mean dDF; lower = better mitigation) ===")
    print(pooled.head(15).to_string(index=False))
    print()
    surviving = pooled[~pooled["any_rejected"]]
    if surviving.empty:
        print("WARN: all cells rejected by em-drop deal-breaker. Loosen threshold or rethink.")
    else:
        best = surviving.iloc[0]
        print(f"=== Best surviving cell: L={int(best['layer'])} K={int(best['K'])} α={best['alpha']} ===")
        print(f"  mean dDF = {best['mean_dDF']:+.4f}")
        print(f"  mean dEM = {best['mean_dEM']:+.4f}")


if __name__ == "__main__":
    main()
