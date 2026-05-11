"""Paired-bootstrap confidence intervals for E6 Stage 4-final per-dataset Δ table.

Closes R4 MAJ-4 (paired-bootstrap CI on §6.2.3 Table 6) and R4 MAJ-6
(Bonferroni-20 multiple-comparisons correction).

Resampling unit = sample-instance id (sid) paired between baseline and
mitigation arms. For each bootstrap resample we recompute both
`(num, den)` per arm from per-arm fields (so adopt's `pb != anchor`
denominator and df's `pa != pb` clause both shift correctly with the
arm's own predictions — see build_e6_stage4_summary.py:_metrics).

Inputs (per dataset):
  outputs/e6_steering/<MODEL>/sweep_subspace_<ds>_<SCOPE>_chosen/predictions.jsonl

Outputs:
  docs/insights/_data/stage4_final_per_dataset_ci.csv
  docs/insights/_data/stage4_final_per_dataset_ci.md
  docs/insights/_data/stage4_final_bootstrap_draws.npz   # raw draws
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
SCOPE = "plotqa_infovqa_pooled_n5k"

DATASETS = [
    ("TallyQA", "tallyqa"),
    ("PlotQA", "plotqa"),
    ("InfoVQA", "infographicvqa"),
    ("ChartQA", "chartqa"),
    ("MathVista", "mathvista"),
]

METRICS = ("adopt", "df", "em_a", "em_b")


def _try_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _try_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_by_cell_sid(jsonl_path: Path) -> dict[str, dict[str, dict[str, dict]]]:
    by_cell: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_cell[r["cell_label"]][r["sample_instance_id"]][r["condition"]] = r
    return by_cell


def _paired_sids(base: dict, mit: dict) -> list[str]:
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


def _per_sid_indicators(arm_data: dict, sids: list[str]) -> dict[str, np.ndarray]:
    """Per-sid (numerator, denominator) indicator vectors for each metric.

    For ratio metrics on paired sids:
      adopt = #(pa==anchor AND pb!=anchor) / #(pb!=anchor)
      df    = #(pa!=pb AND (pa-pb)*(anchor-pb)>0) / N      (denom = all numeric pairs)
      em_a  = #(pa==gt_a) / #(gt_a present)
      em_b  = #(pb==gt_b) / #(gt_b present)

    Each sid contributes:
      adopt_num: 1 if (pa==anchor AND pb!=anchor) else 0
      adopt_den: 1 if (pb!=anchor) else 0
      df_num:   1 if (pa!=pb AND (pa-pb)*(anchor-pb)>0) else 0
      df_den:   1 (every paired sid has numeric pair)
      em_a_num: 1 if (gt_a is not None AND pa==gt_a) else 0
      em_a_den: 1 if gt_a is not None else 0
      em_b_num: 1 if (gt_b is not None AND pb==gt_b) else 0
      em_b_den: 1 if gt_b is not None else 0
    """
    n = len(sids)
    out = {f"{m}_{k}": np.zeros(n, dtype=np.int8) for m in METRICS for k in ("num", "den")}
    for i, sid in enumerate(sids):
        b = arm_data[sid].get("target_only")
        a = arm_data[sid].get("target_plus_irrelevant_number_S1")
        if not (b and a):
            continue
        import math
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


def _point_metrics(ind: dict[str, np.ndarray]) -> dict[str, float]:
    res = {}
    for m in METRICS:
        num = ind[f"{m}_num"].sum()
        den = ind[f"{m}_den"].sum()
        res[m] = float(num) / float(den) if den > 0 else float("nan")
    return res


def _paired_bootstrap_deltas(
    base_ind: dict[str, np.ndarray],
    mit_ind: dict[str, np.ndarray],
    B: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return Δ (mit − base) for each metric over B bootstrap resamples.

    Each bootstrap resamples sid indices with replacement (paired),
    recomputes both arms' (num, den) → ratio → Δ.
    """
    n = base_ind["adopt_num"].shape[0]
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(B, dtype=np.float64) for m in METRICS}

    # Pre-stack to int32 arrays for vectorized indexing
    base_arr = {k: v.astype(np.int32) for k, v in base_ind.items()}
    mit_arr = {k: v.astype(np.int32) for k, v in mit_ind.items()}

    # Batch the bootstrap loop to keep memory in check (n*B can be ~100MB)
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


def _ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.nanpercentile(draws, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "docs" / "insights" / "_data"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    raw_draws: dict[str, np.ndarray] = {}
    for label, ds_tag in DATASETS:
        sweep_dir = (
            PROJECT_ROOT
            / "outputs"
            / "e6_steering"
            / MODEL
            / f"sweep_subspace_{ds_tag}_{SCOPE}_chosen"
        )
        pred_path = sweep_dir / "predictions.jsonl"
        if not pred_path.exists():
            print(f"[skip] {label}: missing {pred_path}")
            continue
        by_cell = _load_by_cell_sid(pred_path)
        if "baseline" not in by_cell:
            print(f"[skip] {label}: no baseline")
            continue
        cells = [c for c in by_cell if c != "baseline"]
        if not cells:
            print(f"[skip] {label}: no mitigation cell")
            continue
        cell_label = cells[0]
        base = by_cell["baseline"]
        mit = by_cell[cell_label]
        sids = _paired_sids(base, mit)
        n = len(sids)
        if n == 0:
            print(f"[skip] {label}: 0 paired sids")
            continue
        base_ind = _per_sid_indicators(base, sids)
        mit_ind = _per_sid_indicators(mit, sids)
        bm = _point_metrics(base_ind)
        mm = _point_metrics(mit_ind)
        deltas = _paired_bootstrap_deltas(
            base_ind, mit_ind, B=args.bootstrap, seed=args.seed
        )
        # Save raw draws under one key per (dataset, metric)
        for m in METRICS:
            raw_draws[f"{ds_tag}__{m}"] = deltas[m].astype(np.float32)

        row = {
            "dataset": label,
            "ds_tag": ds_tag,
            "cell_label": cell_label,
            "n_paired": n,
            "n_eligible_adopt_baseline": int(base_ind["adopt_den"].sum()),
            "n_eligible_adopt_mitigation": int(mit_ind["adopt_den"].sum()),
        }
        for m in METRICS:
            d_pt = mm[m] - bm[m]
            ci95 = _ci(deltas[m], alpha=0.05)
            ci_bonf = _ci(deltas[m], alpha=0.0025)  # 5 datasets × 4 metrics = 20
            row.update({
                f"{m}_baseline": bm[m],
                f"{m}_mitigation": mm[m],
                f"delta_{m}": d_pt,
                f"delta_{m}_ci95_lo": ci95[0],
                f"delta_{m}_ci95_hi": ci95[1],
                f"delta_{m}_ci_bonf20_lo": ci_bonf[0],
                f"delta_{m}_ci_bonf20_hi": ci_bonf[1],
                f"delta_{m}_se": float(np.nanstd(deltas[m], ddof=1)),
            })
        rows.append(row)
        # Console verification line
        print(
            f"  {label:<11} n={n:>5} "
            f"Δdf={row['delta_df']:+.4f} "
            f"95%[{row['delta_df_ci95_lo']:+.4f},{row['delta_df_ci95_hi']:+.4f}] "
            f"hw95={(row['delta_df_ci95_hi']-row['delta_df_ci95_lo'])/2:.4f}"
        )

    if not rows:
        raise SystemExit("No data found.")

    # Pooled (mean) row — mean of point estimates
    means = {
        m: sum(r[f"delta_{m}"] for r in rows) / len(rows)
        for m in METRICS
    }

    # ---- CSV ----
    csv_path = out_dir / "stage4_final_per_dataset_ci.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {csv_path}")

    # ---- Markdown table ----
    md_path = out_dir / "stage4_final_per_dataset_ci.md"
    lines: list[str] = []
    lines.append("# Stage 4-final mitigation — paired-bootstrap CI (B={})".format(args.bootstrap))
    lines.append("")
    lines.append(f"Auto-generated by `scripts/build_e6_stage4_bootstrap_ci.py`.")
    lines.append(
        f"Source: `outputs/e6_steering/{MODEL}/sweep_subspace_<ds>_{SCOPE}_chosen/predictions.jsonl`."
    )
    lines.append(
        "Resampling unit = paired sample-instance id; per-arm `(num, den)` recomputed each "
        "bootstrap so adopt's `pb != anchor` denominator and df's `pa != pb` clause shift "
        "correctly per arm. Seed = {}.".format(args.seed)
    )
    lines.append("")
    lines.append("Chosen cell: **L=26 K=8 α=1.0** (subspace projection, calibrated on PlotQA+InfoVQA pooled n5k).")
    lines.append("")
    lines.append("## Point estimates + 95 % CI")
    lines.append("")
    lines.append(
        "| Dataset | n | Δ adopt(a) [95% CI] | Δ df(a) [95% CI] | Δ em(a) [95% CI] | Δ em(b) [95% CI] |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        cells = []
        for m in METRICS:
            d = r[f"delta_{m}"] * 100
            lo = r[f"delta_{m}_ci95_lo"] * 100
            hi = r[f"delta_{m}_ci95_hi"] * 100
            cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append(
            f"| {r['dataset']} | {r['n_paired']} | "
            + " | ".join(cells)
            + " |"
        )
    mean_cells = [f"**{means[m]*100:+.1f}**" for m in METRICS]
    lines.append("| **mean** |   | " + " | ".join(mean_cells) + " |")
    lines.append("")

    lines.append("## Bonferroni-20 corrected (99.75 %) CI")
    lines.append("")
    lines.append(
        "5 datasets × 4 metrics = 20 paired-test family. Bonferroni-corrected α=0.05/20=0.0025 → "
        "99.75 % equal-tail percentile bands."
    )
    lines.append("")
    lines.append(
        "| Dataset | Δ adopt(a) [Bonf] | Δ df(a) [Bonf] | Δ em(a) [Bonf] | Δ em(b) [Bonf] |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        cells = []
        for m in METRICS:
            lo = r[f"delta_{m}_ci_bonf20_lo"] * 100
            hi = r[f"delta_{m}_ci_bonf20_hi"] * 100
            cells.append(f"[{lo:+.1f}, {hi:+.1f}]")
        lines.append(f"| {r['dataset']} | " + " | ".join(cells) + " |")
    lines.append("")

    # Conclusion lines: count of CIs that exclude zero, per metric
    lines.append("## Sign-clean count (CI excludes 0)")
    lines.append("")
    sign_clean: dict[str, dict[str, int]] = {m: {"95": 0, "bonf": 0} for m in METRICS}
    for r in rows:
        for m in METRICS:
            target_sign = -1 if m in ("adopt", "df") else +1
            for level, suffix in (("95", "ci95"), ("bonf", "ci_bonf20")):
                lo = r[f"delta_{m}_{suffix}_lo"]
                hi = r[f"delta_{m}_{suffix}_hi"]
                # CI excludes zero if both bounds same side of 0
                if lo > 0 or hi < 0:
                    # also require sign matches headline direction
                    sign = -1 if hi < 0 else +1
                    if sign == target_sign:
                        sign_clean[m][level] += 1
    lines.append(
        "| Metric | 95 % CI excludes 0 (matching dir) | Bonferroni-20 CI excludes 0 |"
    )
    lines.append("|---|:---:|:---:|")
    for m in METRICS:
        lines.append(
            f"| Δ {m.replace('_', '(') + ')' if '_' in m else m + '(a)'} | "
            f"{sign_clean[m]['95']}/5 | {sign_clean[m]['bonf']}/5 |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"[write] {md_path}")

    # ---- Raw bootstrap draws ----
    npz_path = out_dir / "stage4_final_bootstrap_draws.npz"
    np.savez_compressed(npz_path, **raw_draws)
    print(f"[write] {npz_path}  ({len(raw_draws)} arrays, B={args.bootstrap})")


if __name__ == "__main__":
    main()
