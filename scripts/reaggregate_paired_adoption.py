"""Re-compute anchor_adopted under the paired definition (M1) for every
existing predictions.jsonl under outputs/.

Idempotent. Backs up the marginal-era artefacts as `*.marginal.bak.*`
beside their replacement; if a backup already exists, the directory is
treated as already re-aggregated and skipped (unless --force).

Usage:
    uv run python scripts/reaggregate_paired_adoption.py             # dry-run, prints plan
    uv run python scripts/reaggregate_paired_adoption.py --apply     # actually rewrite
    uv run python scripts/reaggregate_paired_adoption.py --apply --force   # re-process even backed-up dirs
    uv run python scripts/reaggregate_paired_adoption.py --apply --root outputs/experiment    # subset

The script:
  1. For each predictions.jsonl found under --root (default `outputs/`),
     groups records by `sample_instance_id`.
  2. For each group, finds the `target_only` record's `raw_prediction`
     (preferred) or `prediction` field. That string is `base_prediction`.
  3. For every record in the group, calls `vlm_anchor.metrics.evaluate_sample`
     with `base_prediction=` set, and overwrites `anchor_adopted`,
     `anchor_direction_followed`, `numeric_distance_to_anchor`, and any
     `anchor_value` normalisation drift to match the live evaluator.
  4. Writes the rewritten records back to `predictions.jsonl` and a fresh
     `predictions.csv` (column order preserved) and recomputes
     `summary.json` via `summarize_experiment`.
  5. Backs up the originals as `predictions.marginal.bak.jsonl`,
     `predictions.marginal.bak.csv`, `summary.marginal.bak.json`.

Records missing `sample_instance_id`, missing a target_only row, or with
non-parseable `prediction` strings are passed through unchanged with
`anchor_adopted=0` (matching M1-T1's conservative behaviour) and counted
in a `skipped` tally for the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_anchor.metrics import evaluate_sample, summarize_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER_FIELDS = (
    "anchor_adopted",
    "anchor_direction_followed",
    "numeric_distance_to_anchor",
)


def find_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("predictions.jsonl"))


def already_processed(run_dir: Path) -> bool:
    return (run_dir / "predictions.marginal.bak.jsonl").exists()


def load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in jsonl_path.open()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    columns = list(records[0].keys())
    extra = sorted({k for r in records for k in r if k not in columns})
    columns.extend(extra)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def _display_path(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(run_dir)


def reaggregate_one(run_dir: Path, force: bool, apply: bool) -> dict[str, Any]:
    if already_processed(run_dir) and not force:
        return {"run_dir": _display_path(run_dir), "status": "skipped (already backed up)"}

    jsonl = run_dir / "predictions.jsonl"
    records = load_records(jsonl)
    if not records:
        return {"run_dir": _display_path(run_dir), "status": "empty"}

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_instance: list[dict[str, Any]] = []
    for r in records:
        sid = r.get("sample_instance_id")
        if sid is None:
            no_instance.append(r)
        else:
            by_instance[sid].append(r)

    rewrote = 0
    skipped_no_target = 0
    target_only_count = 0
    new_records: list[dict[str, Any]] = []

    for sid, group in by_instance.items():
        target_rows = [r for r in group if r.get("condition") == "target_only"]
        if not target_rows:
            for r in group:
                r["anchor_adopted"] = 0
                new_records.append(r)
            skipped_no_target += len(group)
            continue
        base_pred_raw = target_rows[0].get("raw_prediction") or target_rows[0].get("prediction") or ""
        for r in group:
            new_eval = evaluate_sample(
                prediction=r.get("raw_prediction") or r.get("prediction") or "",
                gt_answer=r.get("ground_truth", "") or "",
                all_answers=r.get("answers", []) or [],
                anchor_value=r.get("anchor_value"),
                base_prediction=base_pred_raw,
            )
            for f in MARKER_FIELDS:
                r[f] = getattr(new_eval, f)
            if r.get("condition") == "target_only":
                target_only_count += 1
                r["anchor_adopted"] = 0  # always zero for target_only by definition
            new_records.append(r)
            rewrote += 1

    for r in no_instance:
        r["anchor_adopted"] = 0
        new_records.append(r)

    new_records.sort(key=lambda r: (r.get("model", ""), r.get("sample_instance_id") or "", r.get("condition", "")))

    if not apply:
        return {
            "run_dir": _display_path(run_dir),
            "status": "would rewrite",
            "n_total": len(records),
            "n_rewrote": rewrote,
            "n_skipped_no_target": skipped_no_target,
            "n_no_instance_id": len(no_instance),
            "target_only_rows": target_only_count,
        }

    csv_path = run_dir / "predictions.csv"
    summary_path = run_dir / "summary.json"

    if jsonl.exists():
        shutil.copy2(jsonl, run_dir / "predictions.marginal.bak.jsonl")
    if csv_path.exists():
        shutil.copy2(csv_path, run_dir / "predictions.marginal.bak.csv")
    if summary_path.exists():
        shutil.copy2(summary_path, run_dir / "summary.marginal.bak.json")

    write_jsonl(new_records, jsonl)
    write_csv(new_records, csv_path)
    summary = summarize_experiment(new_records)
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "run_dir": _display_path(run_dir),
        "status": "rewrote",
        "n_total": len(records),
        "n_rewrote": rewrote,
        "n_skipped_no_target": skipped_no_target,
        "n_no_instance_id": len(no_instance),
        "target_only_rows": target_only_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs",
                        help="Where to walk for predictions.jsonl (default outputs/)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite files. Without this flag, dry-run only.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process directories that already have backups (overwrite previous re-aggregation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.root)
    print(f"found {len(run_dirs)} predictions.jsonl under {args.root}")
    if not args.apply:
        print("[dry run -- pass --apply to actually rewrite]")

    summary_rows = []
    for d in run_dirs:
        result = reaggregate_one(d, force=args.force, apply=args.apply)
        summary_rows.append(result)
        print(json.dumps(result))

    print(f"\nDONE. {sum(1 for r in summary_rows if r.get('status') == 'rewrote')} rewrote / "
          f"{sum(1 for r in summary_rows if r.get('status') == 'would rewrite')} would-rewrite / "
          f"{sum(1 for r in summary_rows if r.get('status') == 'skipped (already backed up)')} skipped / "
          f"{sum(1 for r in summary_rows if r.get('status') == 'empty')} empty")


if __name__ == "__main__":
    main()
