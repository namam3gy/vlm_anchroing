"""Regression guard for `scripts/phase_a_data_mining.py::_resolve_model_runs`.

The script picks one canonical run dir per model under
`outputs/experiment/<model>/`. Before 2026-04-28, it used the
alphabetically-latest run dir, which silently selected a 45-record
verification smoke run from `20260428-140004/` over the canonical
53,190-record full run from `20260411-213927/`. The fix is to pick the
*largest* run with `n >= min_records` instead.

This test fixtures both a "smoke" and a "full" run side-by-side and
verifies the resolver picks the full one. If a future refactor reverts
to "latest by name", this test fires before any Phase A CSV gets
silently polluted.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "phase_a_data_mining.py"


def _load_resolver():
    spec = importlib.util.spec_from_file_location(
        "phase_a_data_mining", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_data_mining"] = module
    spec.loader.exec_module(module)
    return module._resolve_model_runs


def _write_csv(path: Path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_instance_id", "condition", "prediction"])
        for i in range(n_rows):
            writer.writerow([i, "target_only", "0"])


class ResolveModelRunsTest(unittest.TestCase):
    def test_picks_largest_run_not_alphabetically_latest(self) -> None:
        resolver = _load_resolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "qwen2.5-vl-7b-instruct"
            # Older but canonical full run
            full_run = model_dir / "20260411-213927"
            _write_csv(full_run / "predictions.csv", n_rows=53190)
            # Newer smoke / verification run, alphabetically later
            smoke_run = model_dir / "20260428-140004"
            _write_csv(smoke_run / "predictions.csv", n_rows=45)

            runs = resolver(root)

            self.assertIn("qwen2.5-vl-7b-instruct", runs)
            self.assertEqual(
                runs["qwen2.5-vl-7b-instruct"].name,
                "20260411-213927",
                "resolver must pick the largest run with n >= min_records, "
                "not the alphabetically-latest run dir",
            )

    def test_skips_runs_below_min_records_threshold(self) -> None:
        resolver = _load_resolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "tinytest-model"
            _write_csv(model_dir / "20260101-000001/predictions.csv", n_rows=42)
            _write_csv(model_dir / "20260101-000002/predictions.csv", n_rows=10)

            runs = resolver(root, min_records=100)

            self.assertNotIn("tinytest-model", runs,
                             "all runs below threshold => model excluded")

    def test_ignores_analysis_subdir(self) -> None:
        resolver = _load_resolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_csv(root / "real-model/20260101-000001/predictions.csv", n_rows=200)
            _write_csv(root / "analysis/20260101-000001/predictions.csv", n_rows=10000)

            runs = resolver(root)

            self.assertIn("real-model", runs)
            self.assertNotIn("analysis", runs)


if __name__ == "__main__":
    unittest.main()
