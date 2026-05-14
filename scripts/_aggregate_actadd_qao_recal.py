"""Per-dataset paired-bootstrap CI aggregation for ActAdd + QAO recalibration.

Mirrors build_e6_stage4_bootstrap_ci.py logic (sid-paired bootstrap on
adopt / df / em_a / em_b) but parameterised over sweep-dir family. Produces
one CSV + one MD table per family with rows = (dataset × cell).

Families:
  - actadd: outputs/e6_steering/<MODEL>/tiebreaker_<ds>__from_plotqa_infovqa_pooled/
            (3 cells: L26_a0.5/1.0/2.0, baseline auto)
  - qao:    outputs/e6_steering/<MODEL>/sweep_qao_<ds>_from_plotqa_infovqa_pooled_n200_pooled/
            (16 cells: Lq{20,24,26,27} × Lt26 × α{0.5,1,2,4})

Usage:
  uv run python scripts/_aggregate_actadd_qao_recal.py --family actadd
  uv run python scripts/_aggregate_actadd_qao_recal.py --family qao
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava-onevision-qwen2-7b-ov"

DATASETS = [
    ("TallyQA", "tallyqa"),
    ("PlotQA", "plotqa"),
    ("InfoVQA", "infographicvqa"),
    ("ChartQA", "chartqa"),
    ("MathVista", "mathvista"),
]

METRICS = ("adopt", "df", "em_a", "em_b")

FAMILY_DIR = {
    "actadd": "tiebreaker_{ds}__from_plotqa_infovqa_pooled",
    "qao":    "sweep_qao_{ds}_from_plotqa_infovqa_pooled_n200_pooled",
}


def _try_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_by_cell_sid(jsonl_path: Path):
    """Returns {cell_label: {sid: {condition: record}}} (latest record wins on dup keys)."""
    by_cell: dict = defaultdict(lambda: defaultdict(dict))
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_cell[r["cell_label"]][r["sample_instance_id"]][r["condition"]] = r
    return by_cell


def _paired_sids(base: dict, mit: dict):
    out = []
    for sid in set(base) & set(mit):
        b_b = base[sid].get("target_only")
        b_a = base[sid].get("target_plus_irrelevant_number_S1")
        m_b = mit[sid].get("target_only")
        m_a = mit[sid].get("target_plus_irrelevant_number_S1")
        if not (b_b and b_a and m_b and m_a):
            continue
        try:
            float(b_b["parsed_number"])
            float(b_a["parsed_number"])
            float(b_a["anchor_value"])
            float(m_b["parsed_number"])
            float(m_a["parsed_number"])
            out.append(sid)
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def _per_sid_indicators(arm_data: dict, sids: list[str]):
    n = len(sids)
    out = {f"{m}_{k}": np.zeros(n, dtype=np.int8)
           for m in METRICS for k in ("num", "den")}
    for i, sid in enumerate(sids):
        b = arm_data[sid].get("target_only")
        a = arm_data[sid].get("target_plus_irrelevant_number_S1")
        if not (b and a):
            continue
        pb = _try_float(b["parsed_number"])
        pa = _try_float(a["parsed_number"])
        anchor = _try_float(a.get("anchor_value"))
        if pb is None or pa is None or anchor is None:
            continue
        if not (math.isfinite(pb) and math.isfinite(pa) and math.isfinite(anchor)):
            continue
        gt_b = _try_int(b.get("ground_truth"))
        gt_a = _try_int(a.get("ground_truth"))
        # adopt
        if pb != anchor:
            out["adopt_den"][i] = 1
            if pa == anchor:
                out["adopt_num"][i] = 1
        # df (C-form)
        out["df_den"][i] = 1
        if pa != pb and (pa - pb) * (anchor - pb) > 0:
            out["df_num"][i] = 1
        # em(a)
        if gt_a is not None:
            out["em_a_den"][i] = 1
            if int(pa) == gt_a:
                out["em_a_num"][i] = 1
        # em(b)
        if gt_b is not None:
            out["em_b_den"][i] = 1
            if int(pb) == gt_b:
                out["em_b_num"][i] = 1
    return out


def _point(ind):
    res = {}
    for m in METRICS:
        num = ind[f"{m}_num"].sum()
        den = ind[f"{m}_den"].sum()
        res[m] = float(num) / float(den) if den > 0 else float("nan")
    return res


def _bootstrap_deltas(base_ind, mit_ind, B: int, seed: int):
    n = base_ind["adopt_num"].shape[0]
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(B, dtype=np.float64) for m in METRICS}
    base_arr = {k: v.astype(np.int32) for k, v in base_ind.items()}
    mit_arr = {k: v.astype(np.int32) for k, v in mit_ind.items()}
    BATCH = max(1, min(B, 2000))
    written = 0
    while written < B:
        b = min(BATCH, B - written)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        for m in METRICS:
            num_b = base_arr[f"{m}_num"][idx].sum(axis=1, dtype=np.int64)
            den_b = base_arr[f"{m}_den"][idx].sum(axis=1, dtype=np.int64)
            num_m = mit_arr[f"{m}_num"][idx].sum(axis=1, dtype=np.int64)
            den_m = mit_arr[f"{m}_den"][idx].sum(axis=1, dtype=np.int64)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_b = np.where(den_b > 0, num_b / den_b, np.nan)
                rate_m = np.where(den_m > 0, num_m / den_m, np.nan)
            deltas[m][written:written + b] = rate_m - rate_b
        written += b
    return deltas


def _ci(draws, alpha):
    lo = float(np.nanpercentile(draws, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("actadd", "qao"), required=True)
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "docs" / "insights" / "_data"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_template = FAMILY_DIR[args.family]

    rows = []
    raw_draws = {}
    for label, ds_tag in DATASETS:
        sweep_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / MODEL
                     / fam_template.format(ds=ds_tag))
        pred_path = sweep_dir / "predictions.jsonl"
        if not pred_path.exists():
            print(f"  [skip] {label}: missing {pred_path}")
            continue
        by_cell = _load_by_cell_sid(pred_path)
        if "baseline" not in by_cell:
            print(f"  [skip] {label}: no 'baseline' cell")
            continue
        baseline = by_cell["baseline"]
        cell_labels = sorted(c for c in by_cell if c != "baseline")
        print(f"  [{label}] cells={len(cell_labels)}  base_sids={len(baseline)}")
        for cell in cell_labels:
            mit = by_cell[cell]
            sids = _paired_sids(baseline, mit)
            if not sids:
                print(f"    [{cell}] no paired sids — skip")
                continue
            base_ind = _per_sid_indicators(baseline, sids)
            mit_ind = _per_sid_indicators(mit, sids)
            point_base = _point(base_ind)
            point_mit = _point(mit_ind)
            point_delta = {m: point_mit[m] - point_base[m] for m in METRICS}
            deltas = _bootstrap_deltas(base_ind, mit_ind,
                                        args.bootstrap, args.seed)
            row = {
                "dataset": label,
                "cell": cell,
                "n_paired": len(sids),
            }
            for m in METRICS:
                row[f"{m}_base"] = point_base[m]
                row[f"{m}_mit"] = point_mit[m]
                row[f"{m}_delta"] = point_delta[m]
                lo95, hi95 = _ci(deltas[m], 0.05)
                row[f"{m}_ci95_lo"] = lo95
                row[f"{m}_ci95_hi"] = hi95
                # Bonferroni-20: 5 datasets × 4 metrics = 20 family
                lo_bf, hi_bf = _ci(deltas[m], 0.05 / 20)
                row[f"{m}_ci_bf20_lo"] = lo_bf
                row[f"{m}_ci_bf20_hi"] = hi_bf
            rows.append(row)
            key = f"{ds_tag}__{cell}"
            raw_draws[key] = np.stack([deltas[m] for m in METRICS], axis=0)

    # write CSV
    csv_path = out_dir / f"{args.family}_recal_per_dataset_ci.csv"
    fieldnames = ["dataset", "cell", "n_paired"]
    for m in METRICS:
        fieldnames += [f"{m}_base", f"{m}_mit", f"{m}_delta",
                       f"{m}_ci95_lo", f"{m}_ci95_hi",
                       f"{m}_ci_bf20_lo", f"{m}_ci_bf20_hi"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] CSV: {csv_path}  ({len(rows)} rows)")

    # write compact MD table — Δdf + Δem(b) headline, CI95
    md_path = out_dir / f"{args.family}_recal_per_dataset_ci.md"
    md_lines = [
        f"# {args.family.upper()} recalibration — per-dataset Δ table",
        "",
        f"Source: `{csv_path}`  Bootstrap B={args.bootstrap:,}  seed={args.seed}",
        "Sample-instance paired bootstrap (sid resampling with replacement; both arms recomputed per draw).",
        "",
        "## Δdf(a) [95% CI]  —  primary anchor-mitigation metric",
        "",
        "| Dataset | Cell | n | base | mit | Δ | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['cell']} | {r['n_paired']} | "
            f"{r['df_base']:.3f} | {r['df_mit']:.3f} | "
            f"{r['df_delta']:+.4f} | "
            f"[{r['df_ci95_lo']:+.4f}, {r['df_ci95_hi']:+.4f}] |"
        )
    md_lines += [
        "",
        "## Δem(b) [95% CI]  —  capability-preservation on non-anchored arm",
        "",
        "| Dataset | Cell | n | base | mit | Δ | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['cell']} | {r['n_paired']} | "
            f"{r['em_b_base']:.3f} | {r['em_b_mit']:.3f} | "
            f"{r['em_b_delta']:+.4f} | "
            f"[{r['em_b_ci95_lo']:+.4f}, {r['em_b_ci95_hi']:+.4f}] |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[done] MD: {md_path}")

    # save raw draws
    npz_path = out_dir / f"{args.family}_recal_bootstrap_draws.npz"
    np.savez_compressed(npz_path, **raw_draws,
                        metrics=np.array(list(METRICS)))
    print(f"[done] NPZ: {npz_path}  ({len(raw_draws)} (ds×cell) entries)")


if __name__ == "__main__":
    main()
