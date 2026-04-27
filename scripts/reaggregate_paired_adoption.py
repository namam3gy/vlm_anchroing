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
from vlm_anchor.utils import extract_first_number, normalize_numeric_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER_FIELDS = (
    "anchor_adopted",
    "anchor_direction_followed",
    "numeric_distance_to_anchor",
)


def _normalized_int_str(s: str | None) -> str | None:
    """Normalize a possibly-prose string to its first numeric token; return None
    if the result isn't a parseable integer."""
    if s is None:
        return None
    n = normalize_numeric_text(extract_first_number(str(s)))
    if not n or not n.lstrip("-").isdigit():
        return None
    return n


def paired_adoption(parsed_pred: str | None, anchor_value: str | None,
                    base_parsed_pred: str | None) -> int:
    """Paired anchor_adopted: 1 iff prediction equals anchor AND base differs from anchor."""
    a = _normalized_int_str(anchor_value)
    p = _normalized_int_str(parsed_pred)
    b = _normalized_int_str(base_parsed_pred)
    if a is None or p is None or b is None:
        return 0
    return int(p == a and b != a)


def direction_follow(parsed_pred: str | None, anchor_value: str | None,
                     ground_truth: str | None) -> int:
    a = _normalized_int_str(anchor_value)
    p = _normalized_int_str(parsed_pred)
    g = _normalized_int_str(ground_truth)
    if a is None or p is None or g is None:
        return 0
    return int((int(p) - int(g)) * (int(a) - int(g)) > 0)


def numeric_distance_to_anchor(parsed_pred: str | None,
                               anchor_value: str | None) -> float | None:
    a = _normalized_int_str(anchor_value)
    p = _normalized_int_str(parsed_pred)
    if a is None or p is None:
        return None
    return float(abs(int(p) - int(a)))


def detect_schema(records: list[dict]) -> str:
    """Return 'standard', 'causal_ablation', 'e4_mitigation', 'ablation_like', or 'unknown'."""
    if not records:
        return "unknown"
    sample = records[0]
    if "prediction" in sample and "answers" in sample:
        return "standard"
    if "mode" in sample and "ablate_layers" in sample:
        return "causal_ablation"
    if "mask_strength" in sample and "parsed_number" in sample:
        return "e4_mitigation"
    if "parsed_number" in sample:
        return "ablation_like"
    return "unknown"


def grouping_key(record: dict, schema: str) -> tuple:
    sid = record.get("sample_instance_id")
    if schema == "causal_ablation":
        return (sid, record.get("mode"))
    if schema == "e4_mitigation":
        return (sid, record.get("mask_strength"))
    return (sid, None)


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


def reaggregate_ablation_like(run_dir: Path, records: list[dict[str, Any]],
                              schema: str, force: bool, apply: bool) -> dict[str, Any]:
    """Add paired anchor_adopted (+ direction_follow + distance) fields in-place
    to every record. Group by (sample_instance_id, mode_or_strength).

    Skips summary.json regeneration (preserves existing schema-specific summary
    written by analyze_*.py).
    """
    by_group: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    no_instance: list[dict[str, Any]] = []
    for r in records:
        sid = r.get("sample_instance_id")
        if sid is None:
            no_instance.append(r)
            continue
        by_group[grouping_key(r, schema)].append(r)

    # Per-sid target_only fallback. e4_mitigation full_n* runs only collect
    # target_only at mask_strength=0.0 (target_only has no anchor span, so
    # masking is a no-op — verified empirically in sweep runs). Without this
    # fallback every non-zero-strength row would get anchor_adopted=0.
    target_only_by_sid: dict[Any, str | None] = {}
    for r in records:
        if r.get("condition") == "target_only":
            sid = r.get("sample_instance_id")
            if sid is None:
                continue
            # First write wins; assumes target_only is sid-invariant across
            # groupings (true for e4_mitigation; in causal_ablation each
            # (sid,mode) has its own target_only and the in-group lookup below
            # always succeeds, so the fallback is never hit).
            target_only_by_sid.setdefault(sid, r.get("parsed_number"))

    rewrote = 0
    skipped_no_target = 0
    fallback_used = 0
    target_only_count = 0
    new_records: list[dict[str, Any]] = []

    for key, group in by_group.items():
        target_rows = [r for r in group if r.get("condition") == "target_only"]
        if not target_rows:
            sid = key[0]
            if sid in target_only_by_sid:
                base_parsed = target_only_by_sid[sid]
                fallback_used += len(group)
                # Fall through to the normal compute path.
            else:
                for r in group:
                    r["anchor_adopted"] = 0
                    r["anchor_direction_followed"] = 0
                    r["numeric_distance_to_anchor"] = None
                    new_records.append(r)
                skipped_no_target += len(group)
                continue
        else:
            base_parsed = target_rows[0].get("parsed_number")
        for r in group:
            anchor = r.get("anchor_value")
            parsed = r.get("parsed_number")
            gt = r.get("ground_truth")
            if r.get("condition") == "target_only":
                r["anchor_adopted"] = 0
                r["anchor_direction_followed"] = 0
                r["numeric_distance_to_anchor"] = None
                target_only_count += 1
            else:
                r["anchor_adopted"] = paired_adoption(parsed, anchor, base_parsed)
                r["anchor_direction_followed"] = direction_follow(parsed, anchor, gt)
                r["numeric_distance_to_anchor"] = numeric_distance_to_anchor(parsed, anchor)
            new_records.append(r)
            rewrote += 1

    for r in no_instance:
        r["anchor_adopted"] = 0
        r["anchor_direction_followed"] = 0
        r["numeric_distance_to_anchor"] = None
        new_records.append(r)

    if not apply:
        return {
            "run_dir": _display_path(run_dir),
            "status": f"would rewrite (schema={schema})",
            "n_total": len(records),
            "n_rewrote": rewrote,
            "n_skipped_no_target": skipped_no_target,
            "n_fallback_target_only": fallback_used,
            "n_no_instance_id": len(no_instance),
            "target_only_rows": target_only_count,
        }

    jsonl = run_dir / "predictions.jsonl"
    csv_path = run_dir / "predictions.csv"
    csv_existed = csv_path.exists()

    if jsonl.exists():
        shutil.copy2(jsonl, run_dir / "predictions.marginal.bak.jsonl")
    if csv_existed:
        shutil.copy2(csv_path, run_dir / "predictions.marginal.bak.csv")

    write_jsonl(new_records, jsonl)
    # Only regenerate the csv if there was one before — mirror the original
    # state of the run directory rather than introducing new files.
    if csv_existed:
        write_csv(new_records, csv_path)

    # Explicitly do NOT regenerate summary.json — those dirs use schema-specific
    # summaries written by analyze_causal_ablation.py / analyze_e4_mitigation.py.

    return {
        "run_dir": _display_path(run_dir),
        "status": f"rewrote (schema={schema})",
        "n_total": len(records),
        "n_rewrote": rewrote,
        "n_skipped_no_target": skipped_no_target,
        "n_fallback_target_only": fallback_used,
        "n_no_instance_id": len(no_instance),
        "target_only_rows": target_only_count,
    }


def reaggregate_one(run_dir: Path, force: bool, apply: bool) -> dict[str, Any]:
    if already_processed(run_dir) and not force:
        return {"run_dir": _display_path(run_dir), "status": "skipped (already backed up)"}

    jsonl = run_dir / "predictions.jsonl"
    records = load_records(jsonl)
    if not records:
        return {"run_dir": _display_path(run_dir), "status": "empty"}

    schema = detect_schema(records)
    if schema in ("causal_ablation", "e4_mitigation", "ablation_like"):
        return reaggregate_ablation_like(run_dir, records, schema, force=force, apply=apply)
    if schema == "unknown":
        return {"run_dir": _display_path(run_dir), "status": "skipped (unknown schema)"}

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

    def _count(prefix: str) -> int:
        return sum(1 for r in summary_rows if str(r.get("status", "")).startswith(prefix))

    print(
        f"\nDONE. {_count('rewrote')} rewrote / "
        f"{_count('would rewrite')} would-rewrite / "
        f"{_count('skipped')} skipped / "
        f"{_count('empty')} empty"
    )


if __name__ == "__main__":
    main()
