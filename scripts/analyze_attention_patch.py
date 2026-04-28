"""E1-patch — digit-pixel attention concentration analysis.

Reads `per_step_attention.jsonl` files produced by
`extract_attention_mass.py --bbox-file ...` (which adds `image_anchor_digit`
and `image_anchor_background` fields per layer per step).

Headlines to evaluate per archetype:

  1. **Concentration above fair share** — per layer, the ratio
     `mass_digit / mass_anchor` should exceed the bbox area share
     (= `bbox_area / image_area`) for the answer step. If yes, attention
     is *concentrated* on the digit patch beyond what uniform-attention
     would predict.
  2. **Concentration peak vs. E1b peak** — does the digit-patch
     concentration profile peak at the same layer as the existing
     `image_anchor` total mass? Confirms or refutes "anchor attention
     is anchored on digit pixels" at the locus level.
  3. **Anchor − masked contrast** — the mask arm has the same anchor span
     but with the digit pixel inpainted out. Same bbox should hold but the
     digit is invisible. If `digit_mass(anchor) > digit_mass(masked)`
     consistently, the digit pixel is the cause of the concentration —
     not a position effect.

Outputs (under `docs/insights/_data/`):

  - E1_patch_per_layer.csv           one row per (model, run, layer, step, condition)
  - E1_patch_concentration_ratio.csv per (model, condition), digit/anchor ratio at peak
  - E1_patch_anchor_minus_masked.csv per-layer (anchor − masked) digit-mass diff
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs" / "insights" / "_data"


COND_RE = re.compile(
    r"target_plus_irrelevant_(?P<kind>number_masked|number|neutral)$"
)


def normalize_condition(cond: str) -> str:
    """Map full condition name to a short label: 'b' / 'a' / 'm' / 'd' / '?'.

    For E1 attention dumps the conditions are 1-stratum (no S? suffix); the
    bbox-enabled extraction is run on the same susceptibility-stratified set
    as E1.
    """
    if cond == "target_only":
        return "b"
    m = COND_RE.match(cond)
    if not m:
        return "?"
    kind = m.group("kind")
    return {"number": "a", "number_masked": "m", "neutral": "d"}.get(kind, "?")


def _find_answer_step(per_step_tokens: list[dict]) -> int:
    """First token where the decoded text contains a digit; fall back to step 0."""
    for rec in per_step_tokens:
        text = rec.get("token_text", "")
        if any(c.isdigit() for c in text):
            return rec.get("step", 0)
    return 0


def discover_runs() -> list[Path]:
    """Yield every per_step_attention.jsonl whose anchor records contain
    `image_anchor_digit` (= bbox-enabled extraction). target_only rows
    don't carry the field (no anchor span), so scan up to ~50 records."""
    out: list[Path] = []
    root = REPO_ROOT / "outputs" / "attention_analysis"
    if not root.is_dir():
        return out
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"_per_layer", "analysis"}):
        for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            jsonl = run_dir / "per_step_attention.jsonl"
            if not jsonl.is_file():
                continue
            has_bbox = False
            with jsonl.open() as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ps = row.get("per_step", [])
                    if ps and "image_anchor_digit" in ps[0]:
                        has_bbox = True
                        break
            if has_bbox:
                out.append(jsonl)
    return out


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def per_layer_concentration(records: list[dict]) -> pd.DataFrame:
    """One row per (sample, condition, layer) with digit / background / anchor mass
    and the concentration ratio at the answer step."""
    rows: list[dict] = []
    for r in records:
        if r.get("error"):
            continue
        cond_class = normalize_condition(r["condition"])
        if cond_class not in {"a", "m"}:
            continue
        ps = r.get("per_step")
        if not ps:
            continue
        step = _find_answer_step(r.get("per_step_tokens", []))
        step = min(step, len(ps) - 1)
        ps_step = ps[step]
        digit = ps_step.get("image_anchor_digit")
        anchor = ps_step.get("image_anchor")
        bg = ps_step.get("image_anchor_background")
        if digit is None or anchor is None:
            continue
        n_layers = len(digit)
        for layer_idx in range(n_layers):
            anc = float(anchor[layer_idx])
            dig = float(digit[layer_idx])
            ratio = (dig / anc) if anc > 1e-9 else 0.0
            rows.append({
                "model": r.get("model"),
                "sample_instance_id": r.get("sample_instance_id"),
                "anchor_value": r.get("anchor_value"),
                "condition_class": cond_class,
                "step_used": step,
                "layer": layer_idx,
                "mass_anchor": anc,
                "mass_digit": dig,
                "mass_background": float(bg[layer_idx]) if bg is not None else max(0.0, anc - dig),
                "digit_over_anchor": ratio,
            })
    return pd.DataFrame(rows)


def fair_share_lookup(bbox_file: Path) -> dict[str, float]:
    """anchor_value → bbox_fraction (used as 'fair share' baseline)."""
    if not bbox_file.is_file():
        return {}
    return {k: v.get("fraction", 0.0) for k, v in json.loads(bbox_file.read_text()).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bbox-file", type=Path,
                        default=REPO_ROOT / "inputs" / "irrelevant_number_bboxes.json")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    if not runs:
        print("[discover] no bbox-enabled attention dumps yet — run extract_attention_mass.py --bbox-file first")
        return
    print(f"[discover] {len(runs)} bbox-enabled attention dumps")

    all_rows: list[dict] = []
    for run in runs:
        records = load_records(run)
        df = per_layer_concentration(records)
        df["run"] = run.parent.name
        all_rows.append(df)

    long = pd.concat(all_rows, ignore_index=True)
    long.to_csv(args.out_dir / "E1_patch_per_layer.csv", index=False)

    fair_share = fair_share_lookup(args.bbox_file)
    long["fair_share"] = long["anchor_value"].astype(str).map(fair_share)
    long["concentration_above_fair"] = long["digit_over_anchor"] - long["fair_share"].fillna(0.0)

    # Per (model, condition_class, layer) mean with bootstrap-ish 95% CI on
    # digit_over_anchor.
    per_layer = (
        long.groupby(["model", "condition_class", "layer"])
            .agg(n=("mass_anchor", "size"),
                 mean_digit_over_anchor=("digit_over_anchor", "mean"),
                 mean_concentration_above_fair=("concentration_above_fair", "mean"),
                 std_digit_over_anchor=("digit_over_anchor", "std"))
            .reset_index()
    )
    per_layer.to_csv(args.out_dir / "E1_patch_concentration_per_layer.csv", index=False)

    # Anchor − masked digit-mass diff (paired by sample, layer)
    pivot = long.pivot_table(
        index=["model", "sample_instance_id", "layer"],
        columns="condition_class",
        values="mass_digit",
        aggfunc="first",
    ).reset_index()
    if "a" in pivot.columns and "m" in pivot.columns:
        pivot["a_minus_m"] = pivot["a"] - pivot["m"]
        gap = (
            pivot.groupby(["model", "layer"])
                 .agg(n=("a_minus_m", "size"),
                      mean_a_minus_m=("a_minus_m", "mean"),
                      std_a_minus_m=("a_minus_m", "std"))
                 .reset_index()
        )
        gap.to_csv(args.out_dir / "E1_patch_anchor_minus_masked.csv", index=False)

    if args.print_summary:
        print()
        print("=== Per-model peak concentration (layer with max mean digit_over_anchor on anchor arm) ===")
        for model, sub in per_layer[per_layer["condition_class"] == "a"].groupby("model"):
            peak = sub.loc[sub["mean_digit_over_anchor"].idxmax()]
            n = int(peak["n"])
            print(f"  {model:30s} peak L{int(peak['layer']):3d}/N: digit/anchor = {peak['mean_digit_over_anchor']:.3f} | concentration_above_fair = {peak['mean_concentration_above_fair']:+.3f} | n_samples_per_cell={n}")
        print()
        print("=== Anchor − masked digit-mass gap, per model peak layer ===")
        if "a" in pivot.columns and "m" in pivot.columns:
            gap_df = pd.read_csv(args.out_dir / "E1_patch_anchor_minus_masked.csv")
            for model, sub in gap_df.groupby("model"):
                peak = sub.loc[sub["mean_a_minus_m"].idxmax()]
                print(f"  {model:30s} peak L{int(peak['layer']):3d}: a_minus_m = {peak['mean_a_minus_m']:+.4f}  std={peak['std_a_minus_m']:.4f}  n={int(peak['n'])}")


if __name__ == "__main__":
    main()
