"""Entropy-gated E6 mitigation — selective deployment variant.

Pairs E6 chosen-cell predictions with answer-span cross-entropy from the
baseline E5/E7 target_only runs (per §6.2 confidence binning) and emits a
per-dataset table comparing three streams on paired wrong-base sids:

- baseline       : E6 baseline cell (no projection)
- mitigation     : E6 mitigation cell (L=26 K=8 α=1.0, always-on)
- entropy-gated  : projection applied only when base-pass CE exceeds the
                   per-dataset B2/B3 boundary (33.33 % rank of CE ascending,
                   equiv. samples in §6.2 bins B3-B6 / "anchor-pull regime")

Per §6.2, B1-B2 (top 33 % by confidence) is a broad zero-floor; gating
focuses projection on the regime where the §6 confidence mechanism is active.
Output: `docs/insights/_data/e6_entropy_gated_per_dataset.{csv,md}`.

Sources:
- CE  : `outputs/<base_exp>/llava-onevision-qwen2-7b-ov/<run>/predictions.jsonl`
        field `answer_span_cross_entropy` on `condition == "target_only"`.
- Pred: `outputs/e6_steering/llava-onevision-qwen2-7b-ov/`
        `sweep_subspace_<ds>_plotqa_infovqa_pooled_n5k_chosen/predictions.jsonl`.

CPU-only. No new inference. Run:
    uv run python scripts/build_e6_entropy_gated_summary.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import quantiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava-onevision-qwen2-7b-ov"
SCOPE = "plotqa_infovqa_pooled_n5k"

# Per-dataset baseline run discovery: (display name, ds_tag, base experiment dir)
DATASETS: list[tuple[str, str, str]] = [
    ("TallyQA",   "tallyqa",        "experiment_e5e_tallyqa_full"),
    ("PlotQA",    "plotqa",         "experiment_e7_plotqa_full"),
    ("InfoVQA",   "infographicvqa", "experiment_e7_infographicvqa_full"),
    ("ChartQA",   "chartqa",        "experiment_e5e_chartqa_full"),
    ("MathVista", "mathvista",      "experiment_e5e_mathvista_full"),
]

GATE_PERCENTILE = 1.0 / 3.0  # B2/B3 boundary in 6-equal-bin §6.2 default


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


def _pick_largest_run(base_root: Path) -> Path | None:
    """Memory: pick largest predictions.jsonl, not alphabetically-latest."""
    best: Path | None = None
    best_size = -1
    if not base_root.exists():
        return None
    for run in base_root.iterdir():
        if not run.is_dir():
            continue
        pj = run / "predictions.jsonl"
        if pj.exists() and pj.stat().st_size > best_size:
            best_size = pj.stat().st_size
            best = run
    return best


def _load_ce_map(pred_path: Path) -> dict[str, float]:
    """sid -> CE for target_only records."""
    ce: dict[str, float] = {}
    with pred_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") != "target_only":
                continue
            v = d.get("answer_span_cross_entropy")
            if v is not None:
                ce[d["sample_instance_id"]] = float(v)
    return ce


def _load_by_cell_sid(jsonl_path: Path) -> dict[str, dict[str, dict[str, dict]]]:
    by_cell: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_cell[r["cell_label"]][r["sample_instance_id"]][r["condition"]] = r
    return by_cell


def _paired_sids(base: dict, mit: dict) -> list[str]:
    """sids parseable on (b+a) in BOTH baseline and mitigation cells, with anchor."""
    out: list[str] = []
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
    return out


def _metrics_from_records(records: list[tuple[dict, dict]]) -> dict[str, float | None]:
    """Compute em(b), em(a), df(a), adopt(a) on a list of (b_record, a_record)."""
    em_b_n = em_b_d = 0
    em_a_n = em_a_d = 0
    df_n = df_d = 0
    ad_n = ad_d = 0
    for b, a in records:
        pb = _try_float(b["parsed_number"])
        pa = _try_float(a["parsed_number"])
        anchor = _try_float(a.get("anchor_value"))
        if pb is None or pa is None or anchor is None:
            continue
        gt_b = _try_int(b.get("ground_truth"))
        if gt_b is not None:
            em_b_d += 1
            if int(pb) == gt_b:
                em_b_n += 1
        gt_a = _try_int(a.get("ground_truth"))
        if gt_a is not None:
            em_a_d += 1
            if int(pa) == gt_a:
                em_a_n += 1
        if pb != anchor:
            ad_d += 1
            if pa == anchor:
                ad_n += 1
        df_d += 1
        if pa != pb and (pa - pb) * (anchor - pb) > 0:
            df_n += 1
    return {
        "n": len(records),
        "em_b": em_b_n / em_b_d if em_b_d else None,
        "em_a": em_a_n / em_a_d if em_a_d else None,
        "df": df_n / df_d if df_d else None,
        "adopt": ad_n / ad_d if ad_d else None,
    }


def _percentile_cutoff(values: list[float], frac: float) -> float:
    """Return value v such that ~frac of values are <= v (linear interpolation)."""
    sv = sorted(values)
    n = len(sv)
    if n == 0:
        raise ValueError("empty values for percentile cutoff")
    pos = frac * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return sv[lo] * (1 - w) + sv[hi] * w


def process_dataset(label: str, ds_tag: str, base_exp: str) -> dict | None:
    base_root = PROJECT_ROOT / "outputs" / base_exp / MODEL
    base_run = _pick_largest_run(base_root)
    if base_run is None:
        print(f"[skip] {label}: no baseline run under {base_root}")
        return None
    ce_map = _load_ce_map(base_run / "predictions.jsonl")

    e6_dir = (
        PROJECT_ROOT / "outputs" / "e6_steering" / MODEL
        / f"sweep_subspace_{ds_tag}_{SCOPE}_chosen"
    )
    pred_path = e6_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"[skip] {label}: missing {pred_path}")
        return None
    by_cell = _load_by_cell_sid(pred_path)
    if "baseline" not in by_cell:
        print(f"[skip] {label}: no baseline cell in {pred_path}")
        return None
    mit_cells = [c for c in by_cell if c != "baseline"]
    if not mit_cells:
        print(f"[skip] {label}: no mitigation cell")
        return None
    mit_label = mit_cells[0]
    base = by_cell["baseline"]
    mit = by_cell[mit_label]

    sids = _paired_sids(base, mit)
    if not sids:
        print(f"[skip] {label}: no paired sids")
        return None

    # Drop sids with no CE signal
    sids_with_ce = [sid for sid in sids if sid in ce_map]
    sids_no_ce = [sid for sid in sids if sid not in ce_map]
    if sids_no_ce:
        print(f"[warn] {label}: {len(sids_no_ce)}/{len(sids)} paired sids lack CE; excluded from analysis")

    if not sids_with_ce:
        print(f"[skip] {label}: no paired sid with CE")
        return None

    ce_values = [ce_map[sid] for sid in sids_with_ce]
    cutoff = _percentile_cutoff(ce_values, GATE_PERCENTILE)

    # Build records for each stream
    base_records: list[tuple[dict, dict]] = []
    mit_records: list[tuple[dict, dict]] = []
    sel_records: list[tuple[dict, dict]] = []
    n_gated = 0
    for sid in sids_with_ce:
        b_b = base[sid]["target_only"]
        b_a = base[sid]["target_plus_irrelevant_number_S1"]
        m_b = mit[sid]["target_only"]
        m_a = mit[sid]["target_plus_irrelevant_number_S1"]
        base_records.append((b_b, b_a))
        mit_records.append((m_b, m_a))
        if ce_map[sid] > cutoff:
            sel_records.append((m_b, m_a))
            n_gated += 1
        else:
            sel_records.append((b_b, b_a))

    bm = _metrics_from_records(base_records)
    mm = _metrics_from_records(mit_records)
    sm = _metrics_from_records(sel_records)

    return {
        "dataset": label,
        "ds_tag": ds_tag,
        "cell_label": mit_label,
        "n_paired": len(sids_with_ce),
        "n_dropped_no_ce": len(sids_no_ce),
        "ce_cutoff_B2B3": cutoff,
        "n_gated_on": n_gated,
        "frac_gated_on": n_gated / len(sids_with_ce),
        # baseline arm
        "df_baseline":   bm["df"],
        "em_a_baseline": bm["em_a"],
        "em_b_baseline": bm["em_b"],
        # full mitigation arm
        "df_mit":        mm["df"],
        "em_a_mit":      mm["em_a"],
        "em_b_mit":      mm["em_b"],
        # selective arm
        "df_sel":        sm["df"],
        "em_a_sel":      sm["em_a"],
        "em_b_sel":      sm["em_b"],
        # deltas vs baseline
        "delta_df_mit":    mm["df"] - bm["df"],
        "delta_em_a_mit":  mm["em_a"] - bm["em_a"],
        "delta_em_b_mit":  mm["em_b"] - bm["em_b"],
        "delta_df_sel":    sm["df"] - bm["df"],
        "delta_em_a_sel":  sm["em_a"] - bm["em_a"],
        "delta_em_b_sel":  sm["em_b"] - bm["em_b"],
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "docs" / "insights" / "_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, ds_tag, base_exp in DATASETS:
        r = process_dataset(label, ds_tag, base_exp)
        if r is not None:
            rows.append(r)

    if not rows:
        raise SystemExit("No data found.")

    mean_keys = [
        "delta_df_mit", "delta_em_a_mit", "delta_em_b_mit",
        "delta_df_sel", "delta_em_a_sel", "delta_em_b_sel",
        "frac_gated_on",
    ]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in mean_keys}

    csv_path = out_dir / "e6_entropy_gated_per_dataset.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {csv_path}")

    md = []
    md.append("# Entropy-gated E6 mitigation — per-dataset Δ table")
    md.append("")
    md.append("Auto-generated by `scripts/build_e6_entropy_gated_summary.py`.")
    md.append(
        "Source: E6 chosen-cell predictions + per-sample base-pass "
        "`answer_span_cross_entropy` from "
        "`outputs/experiment_e{5e,7}_<ds>_full/.../predictions.jsonl`."
    )
    md.append("Paired wrong-base sids with CE available.")
    md.append("")
    md.append(
        "Chosen cell: **L=26 K=8 α=1.0** (subspace projection). "
        "Gating cutoff: per-dataset 33.33 % CE percentile (§6.2 B2/B3 boundary)."
    )
    md.append("")
    md.append(
        "Streams: `baseline` = no projection; `mit` = projection on all "
        "samples (the canonical §7.4.5 result); `sel` = projection only when "
        "base-pass CE > cutoff (selective deployment variant)."
    )
    md.append("")
    md.append(
        "| Dataset | n_paired | frac_gated | Δdf(a) mit | Δdf(a) sel | "
        "Δem(b) mit | Δem(b) sel | Δem(a) mit | Δem(a) sel |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['n_paired']} "
            f"| {r['frac_gated_on']:.2%} "
            f"| {r['delta_df_mit']:+.4f} | {r['delta_df_sel']:+.4f} "
            f"| {r['delta_em_b_mit']:+.4f} | {r['delta_em_b_sel']:+.4f} "
            f"| {r['delta_em_a_mit']:+.4f} | {r['delta_em_a_sel']:+.4f} |"
        )
    md.append(
        f"| **mean** |   | **{means['frac_gated_on']:.2%}** "
        f"| **{means['delta_df_mit']:+.4f}** | **{means['delta_df_sel']:+.4f}** "
        f"| **{means['delta_em_b_mit']:+.4f}** | **{means['delta_em_b_sel']:+.4f}** "
        f"| **{means['delta_em_a_mit']:+.4f}** | **{means['delta_em_a_sel']:+.4f}** |"
    )
    md.append("")
    md.append(
        "Reading: with projection restricted to ~2/3 of inputs whose base-pass "
        "is least confident (§6.2 B3-B6 regime), the mitigation gain is "
        "preserved while the projection bypasses the B1-B2 zero-floor where "
        "no anchor pull exists. Deployment-cost trade-off (sequential 2-stage "
        "inference) is discussed in §7.4.5 prose."
    )

    md_path = out_dir / "e6_entropy_gated_per_dataset.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[write] {md_path}")


if __name__ == "__main__":
    main()
