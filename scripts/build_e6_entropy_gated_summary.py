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


# ---- dataset-official accuracy (mirrors scripts/recompute_answer_span_confidence.py)
# em (exact_match) = strict int match used in M2 canonical metrics.
# acc (dataset-official) = the metric the dataset's leaderboard uses:
#   TallyQA, MathVista  : exact int match (same as em)
#   PlotQA, ChartQA     : PlotQA-relaxed (|p-g|/|g| < 5 %; |g|=0 → exact)
#   InfoVQA             : ANLS >= 0.5 (binarised)

def _plotqa_relaxed_correct(pred, gt, tol: float = 0.05) -> int | None:
    p = _try_int(pred)
    g = _try_int(gt)
    if p is None or g is None:
        return None
    if g == 0:
        return int(p == 0)
    return int(abs(p - g) / abs(g) < tol)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def _infovqa_anls(pred, gt, threshold: float = 0.5) -> float | None:
    if pred is None or gt is None:
        return None
    s1 = str(pred).strip().lower()
    s2 = str(gt).strip().lower()
    if not s1 or not s2:
        return None
    m = max(len(s1), len(s2))
    nld = _levenshtein(s1, s2) / max(m, 1)
    sim = 1.0 - nld
    return sim if sim >= threshold else 0.0


def _dataset_acc(dataset: str, pred, gt) -> int | None:
    if dataset in ("TallyQA", "MathVista"):
        p = _try_int(pred)
        g = _try_int(gt)
        if p is None or g is None:
            return None
        return int(p == g)
    if dataset in ("PlotQA", "ChartQA"):
        return _plotqa_relaxed_correct(pred, gt, tol=0.05)
    if dataset == "InfoVQA":
        v = _infovqa_anls(pred, gt, threshold=0.5)
        return None if v is None else int(v > 0)
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


def _metrics_from_records(records: list[tuple[dict, dict, str]]) -> dict[str, float | None]:
    """Compute em(b/a), acc(b/a), df(a), adopt(a) on a list of (b, a, dataset).

    em  = strict integer exact_match (canonical M2 definition)
    acc = dataset-official accuracy (PlotQA/ChartQA 5 % relative tolerance,
          InfoVQA ANLS >= 0.5, TallyQA/MathVista = em)
    """
    em_b_n = em_b_d = 0
    em_a_n = em_a_d = 0
    acc_b_n = acc_b_d = 0
    acc_a_n = acc_a_d = 0
    df_n = df_d = 0
    ad_n = ad_d = 0
    for b, a, dataset in records:
        pb = _try_float(b["parsed_number"])
        pa = _try_float(a["parsed_number"])
        anchor = _try_float(a.get("anchor_value"))
        if pb is None or pa is None or anchor is None:
            continue
        gt_b_int = _try_int(b.get("ground_truth"))
        if gt_b_int is not None:
            em_b_d += 1
            if int(pb) == gt_b_int:
                em_b_n += 1
        gt_a_int = _try_int(a.get("ground_truth"))
        if gt_a_int is not None:
            em_a_d += 1
            if int(pa) == gt_a_int:
                em_a_n += 1
        # dataset-official acc on raw parsed_number / ground_truth strings
        acc_b = _dataset_acc(dataset, b.get("parsed_number"), b.get("ground_truth"))
        if acc_b is not None:
            acc_b_d += 1
            acc_b_n += acc_b
        acc_a = _dataset_acc(dataset, a.get("parsed_number"), a.get("ground_truth"))
        if acc_a is not None:
            acc_a_d += 1
            acc_a_n += acc_a
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
        "acc_b": acc_b_n / acc_b_d if acc_b_d else None,
        "acc_a": acc_a_n / acc_a_d if acc_a_d else None,
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

    # Build records for each stream. Each entry is (b_record, a_record, dataset).
    base_records: list[tuple[dict, dict, str]] = []
    mit_records: list[tuple[dict, dict, str]] = []
    sel_records: list[tuple[dict, dict, str]] = []
    n_gated = 0
    for sid in sids_with_ce:
        b_b = base[sid]["target_only"]
        b_a = base[sid]["target_plus_irrelevant_number_S1"]
        m_b = mit[sid]["target_only"]
        m_a = mit[sid]["target_plus_irrelevant_number_S1"]
        base_records.append((b_b, b_a, label))
        mit_records.append((m_b, m_a, label))
        if ce_map[sid] > cutoff:
            sel_records.append((m_b, m_a, label))
            n_gated += 1
        else:
            sel_records.append((b_b, b_a, label))

    bm = _metrics_from_records(base_records)
    mm = _metrics_from_records(mit_records)
    sm = _metrics_from_records(sel_records)

    metric_keys = ("df", "adopt", "em_b", "em_a", "acc_b", "acc_a")
    summary: dict[str, float | int | str | None] = {
        "dataset": label,
        "ds_tag": ds_tag,
        "cell_label": mit_label,
        "n_paired": len(sids_with_ce),
        "n_dropped_no_ce": len(sids_no_ce),
        "ce_cutoff_B2B3": cutoff,
        "n_gated_on": n_gated,
        "frac_gated_on": n_gated / len(sids_with_ce),
    }
    for stream, m in (("baseline", bm), ("mit", mm), ("sel", sm)):
        for k in metric_keys:
            summary[f"{k}_{stream}"] = m[k]
    for k in metric_keys:
        b = bm[k]
        if b is None:
            summary[f"delta_{k}_mit"] = None
            summary[f"delta_{k}_sel"] = None
            continue
        summary[f"delta_{k}_mit"] = (mm[k] - b) if mm[k] is not None else None
        summary[f"delta_{k}_sel"] = (sm[k] - b) if sm[k] is not None else None
    return {
        "summary": summary,
        "records": {
            "baseline": base_records,
            "mit": mit_records,
            "sel": sel_records,
        },
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "docs" / "insights" / "_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ds: list[dict] = []
    pooled = {"baseline": [], "mit": [], "sel": []}
    for label, ds_tag, base_exp in DATASETS:
        r = process_dataset(label, ds_tag, base_exp)
        if r is None:
            continue
        per_ds.append(r["summary"])
        for stream in pooled:
            pooled[stream].extend(r["records"][stream])

    if not per_ds:
        raise SystemExit("No data found.")

    # Pooled metrics (5-dataset union, paired-sid weighted by dataset size)
    pooled_metrics = {stream: _metrics_from_records(pooled[stream]) for stream in pooled}
    metric_keys = ("df", "adopt", "em_b", "em_a", "acc_b", "acc_a")
    bm = pooled_metrics["baseline"]
    mm = pooled_metrics["mit"]
    sm = pooled_metrics["sel"]
    n_total = sum(r["n_paired"] for r in per_ds)
    n_gated_total = sum(r["n_gated_on"] for r in per_ds)
    pooled_row = {
        "dataset": "POOLED (5-ds union)",
        "n_paired": n_total,
        "frac_gated_on": n_gated_total / n_total if n_total else 0.0,
    }
    for stream, m in (("baseline", bm), ("mit", mm), ("sel", sm)):
        for k in metric_keys:
            pooled_row[f"{k}_{stream}"] = m[k]
    for k in metric_keys:
        b = bm[k]
        if b is None:
            pooled_row[f"delta_{k}_mit"] = None
            pooled_row[f"delta_{k}_sel"] = None
            continue
        pooled_row[f"delta_{k}_mit"] = (mm[k] - b) if mm[k] is not None else None
        pooled_row[f"delta_{k}_sel"] = (sm[k] - b) if sm[k] is not None else None

    # Unweighted per-dataset means (kept for reference; pooled is the headline)
    mean_keys = [
        "delta_df_mit", "delta_df_sel",
        "delta_em_b_mit", "delta_em_b_sel",
        "delta_em_a_mit", "delta_em_a_sel",
        "delta_acc_b_mit", "delta_acc_b_sel",
        "delta_acc_a_mit", "delta_acc_a_sel",
        "delta_adopt_mit", "delta_adopt_sel",
        "frac_gated_on",
    ]
    means = {
        k: (sum(r[k] for r in per_ds if r.get(k) is not None)
            / max(1, sum(1 for r in per_ds if r.get(k) is not None)))
        for k in mean_keys
    }

    # CSV: pooled row + per-dataset rows (same column set; per-dataset adds ds_tag/cell_label/cutoff)
    fieldnames = list(per_ds[0].keys())
    csv_path = out_dir / "e6_entropy_gated_per_dataset.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        # Pooled first
        pooled_csv = {k: pooled_row.get(k) for k in fieldnames}
        pooled_csv["dataset"] = pooled_row["dataset"]
        w.writerow(pooled_csv)
        for r in per_ds:
            w.writerow(r)
    print(f"[write] {csv_path}")

    md = []
    md.append("# Entropy-gated E6 mitigation — Δ table (pooled + per-dataset)")
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
        "samples (canonical §7.4.5 result); `sel` = projection only when "
        "base-pass CE > cutoff. `inputs_touched` is the fraction of inputs "
        "where the projection fires (= `frac_gated_on`); on `baseline` "
        "all inputs are passed through unchanged."
    )
    md.append("")
    md.append(
        "Metrics: **em** = strict integer exact_match (canonical M2 "
        "definition, `int(pred) == int(gt)`). **acc** = dataset-official "
        "accuracy (TallyQA/MathVista exact int; PlotQA/ChartQA 5 % relative "
        "tolerance; InfoVQA ANLS ≥ 0.5). **df** = direction_follow_rate "
        "`(pa-pb)·(anchor-pb)>0 AND pa!=pb`. **adopt** = #(pa==anchor "
        "AND pb!=anchor) / #(pb!=anchor)."
    )
    md.append("")
    md.append("## Pooled (5-dataset paired-sid union; headline)")
    md.append("")
    md.append(
        "Total paired sids `N_pool = " + str(n_total) + "` "
        f"(weighted by dataset size: TallyQA dominates at ~64 %)."
    )
    md.append("")
    md.append(
        "| Stream | inputs touched | df(a) | adopt(a) | em(b) | acc(b) | em(a) | acc(a) |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.append(
        f"| baseline | 0 % | {bm['df']:.4f} | {bm['adopt']:.4f} "
        f"| {bm['em_b']:.4f} | {bm['acc_b']:.4f} | {bm['em_a']:.4f} | {bm['acc_a']:.4f} |"
    )
    md.append(
        f"| mit (unconditional) | 100 % | {mm['df']:.4f} | {mm['adopt']:.4f} "
        f"| {mm['em_b']:.4f} | {mm['acc_b']:.4f} | {mm['em_a']:.4f} | {mm['acc_a']:.4f} |"
    )
    md.append(
        f"| sel (entropy-gated) | {pooled_row['frac_gated_on']:.2%} | {sm['df']:.4f} | {sm['adopt']:.4f} "
        f"| {sm['em_b']:.4f} | {sm['acc_b']:.4f} | {sm['em_a']:.4f} | {sm['acc_a']:.4f} |"
    )
    md.append("")
    md.append(
        "**Δ vs baseline (pooled)**: "
        f"Δdf(a) mit `{pooled_row['delta_df_mit']:+.4f}` sel `{pooled_row['delta_df_sel']:+.4f}`; "
        f"Δadopt mit `{pooled_row['delta_adopt_mit']:+.4f}` sel `{pooled_row['delta_adopt_sel']:+.4f}`; "
        f"Δem(b) mit `{pooled_row['delta_em_b_mit']:+.4f}` sel `{pooled_row['delta_em_b_sel']:+.4f}`; "
        f"Δacc(b) mit `{pooled_row['delta_acc_b_mit']:+.4f}` sel `{pooled_row['delta_acc_b_sel']:+.4f}`; "
        f"Δem(a) mit `{pooled_row['delta_em_a_mit']:+.4f}` sel `{pooled_row['delta_em_a_sel']:+.4f}`; "
        f"Δacc(a) mit `{pooled_row['delta_acc_a_mit']:+.4f}` sel `{pooled_row['delta_acc_a_sel']:+.4f}`."
    )
    md.append("")
    md.append("## Per-dataset breakdown")
    md.append("")
    md.append(
        "| Dataset | n | gated | Δdf(a) mit / sel | Δem(b) mit / sel | "
        "Δacc(b) mit / sel | Δem(a) mit / sel | Δacc(a) mit / sel |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_ds:
        md.append(
            f"| {r['dataset']} | {r['n_paired']} | {r['frac_gated_on']:.2%} "
            f"| {r['delta_df_mit']:+.4f} / {r['delta_df_sel']:+.4f} "
            f"| {r['delta_em_b_mit']:+.4f} / {r['delta_em_b_sel']:+.4f} "
            f"| {r['delta_acc_b_mit']:+.4f} / {r['delta_acc_b_sel']:+.4f} "
            f"| {r['delta_em_a_mit']:+.4f} / {r['delta_em_a_sel']:+.4f} "
            f"| {r['delta_acc_a_mit']:+.4f} / {r['delta_acc_a_sel']:+.4f} |"
        )
    md.append(
        f"| **5-ds mean (unweighted)** |   | **{means['frac_gated_on']:.2%}** "
        f"| **{means['delta_df_mit']:+.4f}** / **{means['delta_df_sel']:+.4f}** "
        f"| **{means['delta_em_b_mit']:+.4f}** / **{means['delta_em_b_sel']:+.4f}** "
        f"| **{means['delta_acc_b_mit']:+.4f}** / **{means['delta_acc_b_sel']:+.4f}** "
        f"| **{means['delta_em_a_mit']:+.4f}** / **{means['delta_em_a_sel']:+.4f}** "
        f"| **{means['delta_acc_a_mit']:+.4f}** / **{means['delta_acc_a_sel']:+.4f}** |"
    )
    md.append("")
    md.append(
        "Reading: the *pooled* result (paired-sid weighted) is the cleanest "
        "single-number summary because dataset sizes differ by ~30×. The "
        "5-dataset unweighted mean is preserved as a per-dataset average for "
        "readers who want equal dataset weight."
    )

    md_path = out_dir / "e6_entropy_gated_per_dataset.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[write] {md_path}")


if __name__ == "__main__":
    main()
