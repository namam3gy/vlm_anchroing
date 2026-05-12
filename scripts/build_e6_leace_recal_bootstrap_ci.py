"""Paired-bootstrap CI on LEACE re-calibration sweep — 5 datasets × {α=0.5, 1.0, 2.0}.

Reads predictions.jsonl from outputs/e6_steering/<MODEL>/sweep_leace_<ds>_recal_pooled/
and computes per-(dataset × cell) Δadopt, Δdf, Δem(a), Δem(b) vs baseline with
paired-bootstrap 95% CI (B=10,000) + Bonferroni-N correction (N = 5 datasets ×
3 cells × 4 metrics = 60).

Output: docs/insights/_data/leace_recal_per_dataset_ci.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava-onevision-qwen2-7b-ov"
SCOPE = "recal"

DATASETS = [
    ("PlotQA", "plotqa"),
    ("InfoVQA", "infographicvqa"),
    ("TallyQA", "tallyqa"),
    ("ChartQA", "chartqa"),
    ("MathVista", "mathvista"),
]

METRICS = ("adopt", "df", "em_a", "em_b")

CELL_LABELS = ("L26_a0.5", "L26_a1.0", "L26_a2.0")


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
    by_cell: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_cell[r["cell_label"]][r["sample_instance_id"]][r["condition"]] = r
    return by_cell


def _paired_sids(base, mit):
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
            float(m_a["anchor_value"])
            out.append(sid)
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


def _per_sid_indicators(arm_data, sids):
    n = len(sids)
    out = {f"{m}_{k}": np.zeros(n, dtype=np.int8) for m in METRICS for k in ("num", "den")}
    import math
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
        if pb != anchor:
            out["adopt_den"][i] = 1
            if pa == anchor:
                out["adopt_num"][i] = 1
        out["df_den"][i] = 1
        if pa != pb and (pa - pb) * (anchor - pb) > 0:
            out["df_num"][i] = 1
        if gt_a is not None:
            out["em_a_den"][i] = 1
            if int(pa) == gt_a:
                out["em_a_num"][i] = 1
        if gt_b is not None:
            out["em_b_den"][i] = 1
            if int(pb) == gt_b:
                out["em_b_num"][i] = 1
    return out


def _point_metrics(ind):
    res = {}
    for m in METRICS:
        num = ind[f"{m}_num"].sum()
        den = ind[f"{m}_den"].sum()
        res[m] = float(num) / float(den) if den > 0 else float("nan")
    return res


def _paired_bootstrap_deltas(base_ind, mit_ind, B, seed):
    n = base_ind["adopt_num"].shape[0]
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(B, dtype=np.float64) for m in METRICS}
    base_arr = {k: v.astype(np.int32) for k, v in base_ind.items()}
    mit_arr = {k: v.astype(np.int32) for k, v in mit_ind.items()}

    BATCH = max(1, min(B, 2000))
    written = 0
    while written < B:
        b_chunk = min(BATCH, B - written)
        idx = rng.integers(0, n, size=(b_chunk, n), dtype=np.int32)
        for m in METRICS:
            num_b = base_arr[f"{m}_num"][idx].sum(axis=1, dtype=np.int64)
            den_b = base_arr[f"{m}_den"][idx].sum(axis=1, dtype=np.int64)
            num_m = mit_arr[f"{m}_num"][idx].sum(axis=1, dtype=np.int64)
            den_m = mit_arr[f"{m}_den"][idx].sum(axis=1, dtype=np.int64)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate_b = np.where(den_b > 0, num_b / den_b, np.nan)
                rate_m = np.where(den_m > 0, num_m / den_m, np.nan)
            deltas[m][written : written + b_chunk] = rate_m - rate_b
        written += b_chunk
    return deltas


def _ci(draws, alpha):
    lo = float(np.nanpercentile(draws, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "docs" / "insights" / "_data"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N_FAMILY = len(DATASETS) * len(CELL_LABELS) * len(METRICS)  # 60
    bonf_alpha = 0.05 / N_FAMILY

    rows: list[dict] = []
    raw_draws: dict[str, np.ndarray] = {}

    for label, ds_tag in DATASETS:
        sweep_dir = (
            PROJECT_ROOT
            / "outputs"
            / "e6_steering"
            / MODEL
            / f"sweep_leace_{ds_tag}_{SCOPE}_pooled"
        )
        pred_path = sweep_dir / "predictions.jsonl"
        if not pred_path.exists():
            print(f"[skip] {label}: missing {pred_path}")
            continue
        by_cell = _load_by_cell_sid(pred_path)
        if "baseline" not in by_cell:
            print(f"[skip] {label}: no baseline")
            continue
        base = by_cell["baseline"]
        for cell_label in CELL_LABELS:
            if cell_label not in by_cell:
                print(f"[skip] {label}/{cell_label}: missing cell")
                continue
            mit = by_cell[cell_label]
            sids = _paired_sids(base, mit)
            n = len(sids)
            if n == 0:
                print(f"[skip] {label}/{cell_label}: 0 paired sids")
                continue
            base_ind = _per_sid_indicators(base, sids)
            mit_ind = _per_sid_indicators(mit, sids)
            bm = _point_metrics(base_ind)
            mm = _point_metrics(mit_ind)
            deltas = _paired_bootstrap_deltas(
                base_ind, mit_ind, B=args.bootstrap, seed=args.seed
            )
            for m in METRICS:
                raw_draws[f"{ds_tag}__{cell_label}__{m}"] = deltas[m].astype(np.float32)

            row = {
                "dataset": label,
                "ds_tag": ds_tag,
                "cell_label": cell_label,
                "n_paired": n,
            }
            for m in METRICS:
                d_pt = mm[m] - bm[m]
                ci95 = _ci(deltas[m], alpha=0.05)
                ci_bonf = _ci(deltas[m], alpha=bonf_alpha)
                row.update({
                    f"{m}_baseline": bm[m],
                    f"{m}_mitigation": mm[m],
                    f"delta_{m}": d_pt,
                    f"delta_{m}_ci95_lo": ci95[0],
                    f"delta_{m}_ci95_hi": ci95[1],
                    f"delta_{m}_ci_bonf{N_FAMILY}_lo": ci_bonf[0],
                    f"delta_{m}_ci_bonf{N_FAMILY}_hi": ci_bonf[1],
                    f"delta_{m}_se": float(np.nanstd(deltas[m], ddof=1)),
                })
            rows.append(row)
            df_str = (f"Δdf={row['delta_df']:+.4f} "
                      f"95%[{row['delta_df_ci95_lo']:+.4f},{row['delta_df_ci95_hi']:+.4f}]")
            print(f"  {label:<11} {cell_label:<12} n={n:>4}  {df_str}")

    if not rows:
        raise SystemExit("No data found.")

    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "leace_recal_per_dataset_ci.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}")

    npz_path = out_dir / "leace_recal_bootstrap_draws.npz"
    np.savez_compressed(npz_path, **raw_draws)
    print(f"wrote {npz_path}")

    md_path = out_dir / "leace_recal_per_dataset_ci.md"
    with md_path.open("w") as f:
        f.write("# LEACE re-calibration — per-dataset paired-bootstrap CI\n\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write(f"- Eraser: `plotqa_infovqa_recal` (X_neg = h^b, X_pos = h^b + (h^a - h^m); PlotQA + InfoVQA pool)\n")
        f.write(f"- Cells: baseline + L=26 × α∈{{0.5, 1.0, 2.0}}\n")
        f.write(f"- Bootstrap: B={args.bootstrap}, seed={args.seed}\n")
        f.write(f"- Bonferroni family: N={N_FAMILY} (5 datasets × 3 cells × 4 metrics)\n\n")
        f.write("## Δdf(a) cross-dataset\n\n")
        f.write("| Dataset | Cell | n | Δdf | 95% CI | Bonferroni CI |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            cif = f"[{r['delta_df_ci95_lo']:+.4f}, {r['delta_df_ci95_hi']:+.4f}]"
            bcif = f"[{r[f'delta_df_ci_bonf{N_FAMILY}_lo']:+.4f}, {r[f'delta_df_ci_bonf{N_FAMILY}_hi']:+.4f}]"
            f.write(f"| {r['dataset']} | {r['cell_label']} | {r['n_paired']} | "
                    f"{r['delta_df']:+.4f} | {cif} | {bcif} |\n")
        f.write("\n## Δem(b) cross-dataset (capability preservation)\n\n")
        f.write("| Dataset | Cell | n | Δem(b) | 95% CI | Bonferroni CI |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            cif = f"[{r['delta_em_b_ci95_lo']:+.4f}, {r['delta_em_b_ci95_hi']:+.4f}]"
            bcif = f"[{r[f'delta_em_b_ci_bonf{N_FAMILY}_lo']:+.4f}, {r[f'delta_em_b_ci_bonf{N_FAMILY}_hi']:+.4f}]"
            f.write(f"| {r['dataset']} | {r['cell_label']} | {r['n_paired']} | "
                    f"{r['delta_em_b']:+.4f} | {cif} | {bcif} |\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
