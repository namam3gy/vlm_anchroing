"""Per-(model, dataset) peak-layer comparison for §7.1-7.3.

Disentangles model-specific vs dataset-shift in attention peak-layer:
the standard `analyze_attention_per_layer.py` pools all run dirs per
model into one peak; this splits per dataset by cross-referencing each
run's question_ids against per-dataset susceptibility CSVs.

Output: docs/insights/_data/cross_dataset_peaks.csv with rows
  (model, dataset, n_records, n_layers, peak_layer, peak_layer_frac,
   peak_delta, peak_ci_low, peak_ci_high)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATT_ROOT = PROJECT_ROOT / "outputs" / "attention_analysis"
SUSC_DIR = PROJECT_ROOT / "docs" / "insights" / "_data"

# Per-dataset susceptibility CSV (provides question_id → dataset mapping).
DATASET_SUSC = {
    "tallyqa": SUSC_DIR / "susceptibility_tallyqa_onevision.csv",
    "plotqa": SUSC_DIR / "susceptibility_plotqa_onevision.csv",
    "infovqa": SUSC_DIR / "susceptibility_infovqa_onevision.csv",
    "vqav2": SUSC_DIR / "susceptibility_strata.csv",  # cross-model (covers VQAv2)
}


def _load_qid_set(path: Path) -> set[str]:
    qids: set[str] = set()
    if not path.exists():
        return qids
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q = row.get("question_id")
            if q:
                qids.add(str(q).strip())
    return qids


def _detect_dataset(run_dir: Path, dataset_qids: dict[str, set[str]]) -> str | None:
    """Read the first 20 records from the run, count how many question_ids
    fall into each dataset's qid set, return the dataset with most matches."""
    f = run_dir / "per_step_attention.jsonl"
    if not f.exists():
        return None
    counts: dict[str, int] = defaultdict(int)
    seen = 0
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                return None
            qid = str(r.get("question_id", "")).strip()
            for ds, qids in dataset_qids.items():
                if qid in qids:
                    counts[ds] += 1
            seen += 1
            if seen >= 20:
                break
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _load_records(run_dir: Path) -> list[dict]:
    f = run_dir / "per_step_attention.jsonl"
    if not f.exists():
        return []
    out = []
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a corrupt line but continue collecting valid ones.
                continue
    return out


def _compute_peak(records: list[dict], step_label: str = "answer") -> dict | None:
    """Replicates the peak-layer logic from analyze_attention_per_layer
    (number-vs-neutral delta on full population). Step label "answer"
    locks onto the first answer-digit step; "step0" uses the prefill."""
    # Group by (sid, condition) and compute per-layer anchor mass deltas.
    by_sid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_sid[r["sample_instance_id"]][r["condition"]] = r

    deltas_by_layer: dict[int, list[float]] = defaultdict(list)
    n_layers_seen: set[int] = set()
    n_triplets = 0
    a_conds = ("target_plus_irrelevant_number", "target_plus_irrelevant_number_S1")
    for sid, by_cond in by_sid.items():
        a_rec = next((by_cond[c] for c in a_conds if c in by_cond), None)
        d_rec = by_cond.get("target_plus_irrelevant_neutral")
        if a_rec is None or d_rec is None:
            continue
        # Pick step
        if step_label == "answer":
            from re import compile as _re
            digit = _re(r"\d")
            tokens = a_rec.get("per_step_tokens") or []
            step = next((int(t["step"]) for t in tokens if digit.search(t.get("token_text") or "")), None)
            if step is None:
                continue
        else:
            step = 0
        per_step_a = a_rec.get("per_step", [])
        per_step_d = d_rec.get("per_step", [])
        if step >= len(per_step_a) or step >= len(per_step_d):
            continue
        # per_step[step]["image_anchor"] is the per-layer anchor-region mass
        # (already aggregated over the bbox bands at extraction time).
        layers_a = per_step_a[step].get("image_anchor") or []
        layers_d = per_step_d[step].get("image_anchor") or []
        if not layers_a or not layers_d:
            continue
        L = min(len(layers_a), len(layers_d))
        n_layers_seen.add(L)
        for li in range(L):
            try:
                deltas_by_layer[li].append(float(layers_a[li]) - float(layers_d[li]))
            except (TypeError, ValueError):
                continue
        n_triplets += 1
    if not deltas_by_layer:
        return None
    means = []
    cis = []
    for li in sorted(deltas_by_layer.keys()):
        arr = np.array(deltas_by_layer[li])
        n = len(arr)
        if n < 5:
            continue
        m = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        means.append((li, m, m - 1.96 * sem, m + 1.96 * sem, n))
    if not means:
        return None
    peak = max(means, key=lambda x: x[1])
    n_layers = max(n_layers_seen) if n_layers_seen else None
    return {
        "n_records": n_triplets,
        "n_layers": n_layers,
        "peak_layer": peak[0],
        "peak_layer_frac": peak[0] / (n_layers - 1) if n_layers and n_layers > 1 else None,
        "peak_delta": peak[1],
        "peak_ci_low": peak[2],
        "peak_ci_high": peak[3],
        "n_at_peak": peak[4],
    }


def main() -> None:
    dataset_qids = {ds: _load_qid_set(p) for ds, p in DATASET_SUSC.items()}
    print(f"[info] loaded susceptibility CSVs:")
    for ds, qs in dataset_qids.items():
        print(f"  {ds}: {len(qs)} unique qids")

    rows: list[dict] = []
    if not ATT_ROOT.exists():
        raise SystemExit(f"missing {ATT_ROOT}")
    for model_dir in sorted(ATT_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        per_dataset_records: dict[str, list[dict]] = defaultdict(list)
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ds = _detect_dataset(run_dir, dataset_qids)
            if ds is None:
                continue
            per_dataset_records[ds].extend(_load_records(run_dir))
        for ds, recs in per_dataset_records.items():
            for step_label in ("answer", "step0"):
                peak = _compute_peak(recs, step_label=step_label)
                if peak is None:
                    continue
                rows.append({
                    "model": model_dir.name,
                    "dataset": ds,
                    "step": step_label,
                    **peak,
                })
                print(f"  {model_dir.name:<32} {ds:<10} {step_label:<6} "
                      f"n={peak['n_records']:>5} L={peak['peak_layer']}/{peak['n_layers']} "
                      f"frac={peak['peak_layer_frac']:.2f} delta={peak['peak_delta']:+.4f}")

    out_csv = SUSC_DIR / "cross_dataset_peaks.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {len(df)} rows to {out_csv}")
    if not df.empty:
        print("\n=== Pivot: peak_layer per (model, dataset) — answer step ===")
        ans = df[df["step"] == "answer"]
        print(ans.pivot_table(index="model", columns="dataset",
                               values="peak_layer", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
