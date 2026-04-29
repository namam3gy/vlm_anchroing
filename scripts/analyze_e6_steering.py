"""Analyze E6 Phase 1 sweep results.

Reads `outputs/e6_steering/<model>/sweep_n200/predictions.jsonl`, aggregates
per-cell metrics, prints the (L*, α*, v_var*) selection per the design-doc
rule, and writes Pareto data + chosen_cell.json.

Per-cell metrics (matching M2 / C-form):
  df_a    = direction_follow_rate on anchor-arm  (a-S1)
            = #( (pa - pb) · (anchor - pb) > 0  AND pa != pb )
              / #( numeric pair AND anchor != pb )
  adopt_a = adoption rate on anchor-arm
            = #( pa == anchor AND pb != anchor ) / #( pb != anchor )
  em_b    = exact_match on target_only (no-anchor invariance check)
  em_d    = exact_match on target_plus_irrelevant_neutral (no-digit check)
  em_a    = exact_match on anchor-arm
  em_m    = exact_match on masked-arm (sanity — should ≈ em_b)
  mdist   = mean numeric distance to anchor on anchor-arm (fluency monitor)

`pa` and `pb` are the cell's own predictions (within-cell pairing) so that
each cell's pull metric is internally consistent.

Selection rule (design doc):
  smallest |α| satisfying on a-S1:
    df_a(cell)  ≤  0.9 · df_a(baseline)             # ≥ 10 % rel reduction
    em_b(cell)  ≥  em_b(baseline) − 0.02            # no-anchor invariant
    em_d(cell)  ≥  em_d(baseline) − 0.02            # no-digit invariant
    em_a(cell)  ≥  em_a(baseline) − 0.02            # anchor arm not damaged
    mdist(cell) ≤  mdist(baseline) + 1.0            # fluency guard

Tiebreakers: smaller |α| → v_wrong over v_all → closer to mid-stack.

Usage:
    uv run python scripts/analyze_e6_steering.py \\
        --model llava-next-interleaved-7b
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap.parse_args()


def _to_int(x) -> int | None:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _index_by_cell_sid_cond(records: list[dict]) -> dict:
    """records → {(layer, alpha, v_var_idx): {(sid, cond): record}}"""
    out: dict = defaultdict(dict)
    for r in records:
        cell_key = (int(r["cell_layer"]), float(r["cell_alpha"]), int(r["cell_v_var_idx"]))
        sid_cond = (str(r["sample_instance_id"]), str(r["condition"]))
        out[cell_key][sid_cond] = r
    return out


def _cell_metrics(cell_records: dict) -> dict:
    """Compute df_a, adopt_a, em_b, em_d, em_a, em_m, mdist on a single cell's
    (sid, condition) → record map."""
    # Group by sid: collect b/a-S1/m-S1/d records present.
    by_sid: dict = defaultdict(dict)
    for (sid, cond), r in cell_records.items():
        by_sid[sid][cond] = r

    df_num, df_den = 0, 0
    adopt_num, adopt_den = 0, 0
    em_per_cond: dict = {
        "target_only": [0, 0],  # [match, total]
        "target_plus_irrelevant_neutral": [0, 0],
        "target_plus_irrelevant_number_S1": [0, 0],
        "target_plus_irrelevant_number_masked_S1": [0, 0],
    }
    mdist_sum, mdist_cnt = 0.0, 0

    for sid, conds in by_sid.items():
        # exact_match per condition (pre-recorded by sweep driver as int 0/1)
        for cond_name, em_acc in em_per_cond.items():
            r = conds.get(cond_name)
            if r is None:
                continue
            em_acc[1] += 1
            em_acc[0] += int(r.get("exact_match", 0))

        # anchor-arm metrics need pb (target_only) and pa (a-S1) within this cell
        b = conds.get("target_only")
        a = conds.get("target_plus_irrelevant_number_S1")
        if b is None or a is None:
            continue
        pa = _to_int(a.get("parsed_number"))
        pb = _to_int(b.get("parsed_number"))
        anchor = _to_int(a.get("anchor_value"))
        if pa is None or pb is None or anchor is None:
            continue

        # adoption: pa == anchor given pb != anchor
        if pb != anchor:
            adopt_den += 1
            if pa == anchor:
                adopt_num += 1

        # direction follow (C-form): (pa - pb) * (anchor - pb) > 0 AND pa != pb
        if anchor != pb:
            df_den += 1
            if (pa - pb) * (anchor - pb) > 0 and pa != pb:
                df_num += 1

        # numeric distance to anchor (mean over anchor-arm valid)
        mdist_sum += abs(pa - anchor)
        mdist_cnt += 1

    em = {k: (v[0] / v[1] if v[1] else None) for k, v in em_per_cond.items()}
    return {
        "df_a": df_num / df_den if df_den else None,
        "df_a_n": df_den,
        "adopt_a": adopt_num / adopt_den if adopt_den else None,
        "adopt_a_n": adopt_den,
        "em_b": em["target_only"],
        "em_d": em["target_plus_irrelevant_neutral"],
        "em_a": em["target_plus_irrelevant_number_S1"],
        "em_m": em["target_plus_irrelevant_number_masked_S1"],
        "mdist": mdist_sum / mdist_cnt if mdist_cnt else None,
        "mdist_n": mdist_cnt,
        "n_sids": len(by_sid),
    }


def _check_selection(cell_m: dict, baseline_m: dict) -> dict:
    """Return per-criterion pass/fail booleans + summary."""
    if cell_m["df_a"] is None or baseline_m["df_a"] is None:
        return {"valid": False, "reason": "missing metrics"}
    df_target = 0.9 * baseline_m["df_a"]
    pass_df = cell_m["df_a"] <= df_target
    pass_em_b = (cell_m["em_b"] is not None and baseline_m["em_b"] is not None
                 and cell_m["em_b"] >= baseline_m["em_b"] - 0.02)
    pass_em_d = (cell_m["em_d"] is not None and baseline_m["em_d"] is not None
                 and cell_m["em_d"] >= baseline_m["em_d"] - 0.02)
    pass_em_a = (cell_m["em_a"] is not None and baseline_m["em_a"] is not None
                 and cell_m["em_a"] >= baseline_m["em_a"] - 0.02)
    pass_mdist = (cell_m["mdist"] is not None and baseline_m["mdist"] is not None
                  and cell_m["mdist"] <= baseline_m["mdist"] + 1.0)
    return {
        "valid": True,
        "df_a_target": df_target,
        "pass_df": pass_df,
        "pass_em_b": pass_em_b,
        "pass_em_d": pass_em_d,
        "pass_em_a": pass_em_a,
        "pass_mdist": pass_mdist,
        "all_pass": pass_df and pass_em_b and pass_em_d and pass_em_a and pass_mdist,
    }


def main() -> None:
    args = _parse_args()
    base_dir = PROJECT_ROOT / "outputs" / "e6_steering" / args.model
    pred_path = base_dir / "sweep_n200" / "predictions.jsonl"
    out_dir = PROJECT_ROOT / "outputs" / "e6_steering" / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {pred_path}")
    records = _load_records(pred_path)
    print(f"[load] {len(records)} records")

    by_cell = _index_by_cell_sid_cond(records)
    print(f"[load] {len(by_cell)} cells")

    # Identify baseline cell (alpha == 0)
    baseline_keys = [k for k in by_cell if k[1] == 0.0]
    if not baseline_keys:
        raise RuntimeError("no baseline cell (alpha=0) found in records")
    baseline_key = baseline_keys[0]
    baseline_m = _cell_metrics(by_cell[baseline_key])
    print(f"[baseline] {baseline_m}")

    # Compute per-cell metrics
    rows = []
    for cell_key, cell_records in sorted(by_cell.items()):
        L, alpha, vv_idx = cell_key
        m = _cell_metrics(cell_records)
        sel = _check_selection(m, baseline_m) if alpha != 0 else {"valid": False}
        v_var_name = "v_wrong" if vv_idx == 0 else ("v_all" if vv_idx == 1 else "baseline")
        row = {
            "cell_layer": L,
            "cell_alpha": alpha,
            "cell_v_var_idx": vv_idx,
            "v_var": v_var_name,
            "label": ("baseline" if alpha == 0
                      else f"L{L:02d}_a{alpha}_{v_var_name}"),
            **m,
            "df_rel_reduction": (
                None if (m["df_a"] is None or baseline_m["df_a"] is None
                         or baseline_m["df_a"] == 0)
                else (m["df_a"] - baseline_m["df_a"]) / baseline_m["df_a"]
            ),
            "all_pass": sel.get("all_pass", False),
        }
        rows.append(row)

    # CSV
    csv_path = out_dir / "sweep_pareto.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[save] {csv_path}")

    # Print top candidates (passing selection rule)
    candidates = [r for r in rows if r["all_pass"]]
    print(f"\n[selection] {len(candidates)} cells pass all criteria:")
    for r in candidates:
        print(f"  {r['label']:30s}  df_a={r['df_a']:.4f} ({r['df_rel_reduction']*100:+.1f}% rel)  "
              f"em_b={r['em_b']:.3f} em_d={r['em_d']:.3f} em_a={r['em_a']:.3f}")

    # Read n_layers from v_meta for principled mid-stack tiebreaker
    v_meta_path = base_dir / "calibration" / "v_meta.json"
    n_layers = 32
    if v_meta_path.exists():
        n_layers = int(json.loads(v_meta_path.read_text())["n_layers"])

    # Apply tiebreakers: smallest |α| → v_wrong over v_all → closer to mid-stack
    if candidates:
        def tiebreak(r):
            return (
                abs(r["cell_alpha"]),
                r["cell_v_var_idx"],  # 0 (v_wrong) before 1 (v_all)
                abs(r["cell_layer"] - n_layers // 2),
            )
        chosen = min(candidates, key=tiebreak)
    else:
        chosen = None

    if chosen:
        print(f"\n[chosen] {chosen['label']}")
        chosen_path = out_dir / "chosen_cell.json"
        chosen_path.write_text(json.dumps({
            "model": args.model,
            "cell_layer": chosen["cell_layer"],
            "cell_alpha": chosen["cell_alpha"],
            "cell_v_var_idx": chosen["cell_v_var_idx"],
            "v_var": chosen["v_var"],
            "metrics": {k: chosen[k] for k in (
                "df_a", "adopt_a", "em_b", "em_d", "em_a", "em_m", "mdist",
                "df_rel_reduction"
            )},
            "baseline_metrics": {k: baseline_m[k] for k in (
                "df_a", "adopt_a", "em_b", "em_d", "em_a", "em_m", "mdist"
            )},
        }, indent=2))
        print(f"[save] {chosen_path}")
    else:
        print("\n[chosen] NONE — no cell passes all criteria. Failure-escalation paths:")
        print("  (a) try projection h ← h − (h·v̂)·v̂ instead of subtraction")
        print("  (b) per-layer-pair v[L1] − v[L2] differences")
        print("  (c) multi-layer steering (still residual-stream, label-free)")

    # Print full table sorted by df_rel_reduction
    def _fmt(val, spec, na_width):
        if val is None:
            return f"{'n/a':>{na_width}s}"
        return format(val, spec)

    print(f"\n[full table] sorted by df relative reduction:")
    print(f"  {'label':30s}  {'df_a':>7s}  {'rel%':>7s}  {'em_b':>5s}  "
          f"{'em_d':>5s}  {'em_a':>5s}  {'mdist':>7s}  pass")
    for r in sorted(rows, key=lambda x: (
            x["df_rel_reduction"] if x["df_rel_reduction"] is not None else 999)):
        rel = r["df_rel_reduction"]
        rel_str = f"{rel * 100:+6.1f}%" if rel is not None else f"{'n/a':>7s}"
        line = (
            f"  {r['label']:30s}  "
            f"{_fmt(r['df_a'], '>7.4f', 7)}  "
            f"{rel_str:>7s}  "
            f"{_fmt(r['em_b'], '>5.3f', 5)}  "
            f"{_fmt(r['em_d'], '>5.3f', 5)}  "
            f"{_fmt(r['em_a'], '>5.3f', 5)}  "
            f"{_fmt(r['mdist'], '>7.2f', 7)}  "
            f"{'✓' if r['all_pass'] else ''}"
        )
        print(line)


if __name__ == "__main__":
    main()
