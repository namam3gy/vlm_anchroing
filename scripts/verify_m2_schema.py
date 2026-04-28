"""CI-style M2 schema guard.

Walk every ``predictions.jsonl`` under ``--root`` (default ``outputs/``,
excluding ``outputs/before_C_form/``) and assert that the first row has
all six M2 flags. Exits non-zero on the first violation so it can be
chained into pre-commit / pre-push hooks.

Usage::

    uv run python scripts/verify_m2_schema.py
    uv run python scripts/verify_m2_schema.py --root outputs/experiment

Why this exists: between 2026-04-08 and 2026-04-28 the project drove
~60 dirs of `predictions.jsonl` while `run_experiment.py` silently
omitted three M2 flags (`anchor_direction_followed_moved`,
`pred_b_equal_anchor`, `pred_diff_from_base`) from each row dict.
``summarize_condition._flag()`` returns 0 on missing keys, so the
schema gap reported `df_M2 = 0` for ~3 weeks before being noticed.
This guard catches that class of regression at write time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FLAGS = (
    "anchor_adopted",
    "anchor_direction_followed",
    "anchor_direction_followed_moved",
    "pred_b_equal_anchor",
    "pred_diff_from_base",
    "numeric_distance_to_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["before_C_form"],
        help="Sub-trees to skip (default: before_C_form).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root
    if not root.exists():
        print(f"root not found: {root}", file=sys.stderr)
        return 0

    violations = []
    n_checked = 0
    for path in root.rglob("predictions.jsonl"):
        if any(part in args.exclude for part in path.parts):
            continue
        n_checked += 1
        try:
            with path.open() as f:
                first_line = next((l for l in f if l.strip()), None)
        except OSError as exc:
            violations.append((path, f"unreadable: {exc}"))
            continue
        if first_line is None:
            continue  # empty jsonl — skip
        try:
            row = json.loads(first_line)
        except json.JSONDecodeError as exc:
            violations.append((path, f"non-JSON first row: {exc}"))
            continue
        # `target_only` rows have anchor_value=None, so anchor-derived
        # flags can be 0 — but the keys must still exist.
        missing = [f for f in REQUIRED_FLAGS if f not in row]
        if missing:
            violations.append((path, f"missing flags: {missing}"))

    if violations:
        print(f"M2 SCHEMA FAIL — {len(violations)} of {n_checked} files violate the M2 schema:")
        for path, why in violations:
            print(f"  {path}: {why}")
        return 1
    print(f"M2 SCHEMA OK — {n_checked} predictions.jsonl files all carry the 6-flag schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
