"""Build site/data/demo.json + site/assets/img/ from outputs/ predictions."""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Display id -> outputs/ subdirectory name
MAIN_PANEL: dict[str, str] = {
    "llava-onevision-7b":  "llava-onevision-qwen2-7b-ov",
    "qwen2.5-vl-7b":       "qwen2.5-vl-7b-instruct",
    "gemma-3-27b":         "gemma3-27b-it",
    "internvl3-8b":        "internvl3-8b",
    "gemma-3-4b":          "gemma3-4b-it",
}
MAIN_PANEL_LABELS: dict[str, str] = {
    "llava-onevision-7b":  "LLaVA-OneVision 7B (main)",
    "qwen2.5-vl-7b":       "Qwen2.5-VL 7B",
    "gemma-3-27b":         "Gemma-3 27B",
    "internvl3-8b":        "InternVL3 8B",
    "gemma-3-4b":          "Gemma-3 4B",
}
FORBIDDEN_OUTPUT_SUBTREE = "before_C_form"

# Reverse map: outputs/ dir name -> display id
DIR_TO_DISPLAY: dict[str, str] = {v: k for k, v in MAIN_PANEL.items()}

# Inputs subdirectory -> dataset display name
DATASET_LABELS: dict[str, str] = {
    "vqav2_number_val": "VQAv2",
    "chartqa_test": "ChartQA",
    "plotqa_test": "PlotQA",
    "mathvista_testmini": "MathVista",
    "infographicvqa_val": "InfographicsVQA",
    "tallyqa_test": "TallyQA",
}

CONDITION_BASE_TO_CODE: dict[str, str] = {
    "target_only": "b",
    "target_plus_irrelevant_number": "a",
    "target_plus_irrelevant_number_masked": "m",
    "target_plus_irrelevant_neutral": "d",
}


def _condition_code(label: str, anchor_stratum: str) -> str | None:
    """Map a CSV condition string to b/a/m/d, respecting anchor_stratum.

    For stratified labels (e.g. ``target_plus_irrelevant_number_S1``), accept
    only the configured stratum; for non-stratified labels accept directly.
    """
    if label in CONDITION_BASE_TO_CODE:
        return CONDITION_BASE_TO_CODE[label]
    m = re.match(r"^(target_plus_irrelevant_(?:number|number_masked|neutral))_S\d+$", label)
    if not m:
        return None
    base, suffix = m.group(1), label.rsplit("_", 1)[-1]
    if suffix != anchor_stratum:
        return None
    return CONDITION_BASE_TO_CODE.get(base)


def _infer_dataset(input_image_paths: str) -> str:
    """Extract dataset display name from a Python-list-style stringified path."""
    if not isinstance(input_image_paths, str):
        return "unknown"
    for key, label in DATASET_LABELS.items():
        if f"/inputs/{key}/" in input_image_paths or f"\\inputs\\{key}\\" in input_image_paths:
            return label
    return "unknown"


def _to_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _first_path(input_image_paths: str) -> str | None:
    """Extract the first absolute path from a stringified list."""
    if not isinstance(input_image_paths, str):
        return None
    m = re.search(r"'([^']+)'", input_image_paths)
    return m.group(1) if m else None


def _iter_prediction_csvs(outputs_root: Path) -> Iterable[Path]:
    if not outputs_root.exists():
        return
    for path in outputs_root.rglob("predictions.csv"):
        rel = path.relative_to(outputs_root)
        if rel.parts and rel.parts[0] == FORBIDDEN_OUTPUT_SUBTREE:
            continue
        yield path


def _latest_runs(csv_paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    """For each (experiment, model_dir), pick the run with the largest CSV."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in csv_paths:
        parts = path.parts
        try:
            exp_idx = parts.index("outputs") + 1
        except ValueError:
            continue
        if len(parts) < exp_idx + 4:
            continue
        experiment, model_dir = parts[exp_idx], parts[exp_idx + 1]
        if model_dir == "analysis":
            continue
        grouped[(experiment, model_dir)].append(path)

    chosen: dict[tuple[str, str], Path] = {}
    for key, paths in grouped.items():
        # Largest CSV wins — avoids smoke-run pollution (many tiny runs accumulate
        # under outputs/<exp>/<model>/, an alphabetical "latest" rule would shadow
        # the canonical full run).
        chosen[key] = max(paths, key=lambda p: p.stat().st_size)
    return chosen


def load_predictions(
    *, outputs_root: Path, anchor_stratum: str = "S1",
) -> dict[str, dict[str, dict]]:
    """Return {display_model_id: {sample_id: {b, a, m, d, meta}}}.

    ``meta`` carries question, gt, anchor, dataset, and the b-arm input image
    path so the caller can copy assets without re-reading the CSV.
    """
    runs = _latest_runs(_iter_prediction_csvs(outputs_root))
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)

    for (_, model_dir), csv_path in runs.items():
        display_id = DIR_TO_DISPLAY.get(model_dir)
        if display_id is None:
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        for sid, group in df.groupby("sample_instance_id"):
            sample: dict = {"meta": {}}
            for _, row in group.iterrows():
                code = _condition_code(str(row["condition"]), anchor_stratum)
                if code is None:
                    continue
                pred = _to_int(row["prediction"])
                if pred is None:
                    continue
                sample[code] = pred
                if code == "b":
                    sample["meta"] = {
                        "question": str(row.get("question", "")),
                        "gt": _to_int(row.get("ground_truth")),
                        "anchor": _to_int(row.get("anchor_value")),
                        "dataset": _infer_dataset(str(row.get("input_image_paths", ""))),
                        "target_image_path": _first_path(row.get("input_image_paths", "")),
                    }
                elif code == "a" and sample["meta"].get("anchor") is None and "question" in sample["meta"]:
                    sample["meta"]["anchor"] = _to_int(row.get("anchor_value"))
            if "b" not in sample or "question" not in sample["meta"]:
                continue
            by_model[display_id][str(sid)] = sample
    return dict(by_model)


REQUIRED_CONDITIONS = ("b", "a", "m", "d")


def eligible_samples(by_model: dict[str, dict[str, dict]]) -> list[str]:
    """Return sample ids that have every (main-panel model × b/a/m/d)."""
    if not all(mid in by_model for mid in MAIN_PANEL):
        missing = [mid for mid in MAIN_PANEL if mid not in by_model]
        print(f"WARN: missing models in outputs/: {missing}", file=sys.stderr)
        return []
    sample_ids = set(by_model[next(iter(MAIN_PANEL))])
    for mid in MAIN_PANEL:
        sample_ids &= set(by_model[mid])
    eligible = []
    for sid in sorted(sample_ids):
        ok = True
        for mid in MAIN_PANEL:
            sample = by_model[mid][sid]
            if not all(cond in sample for cond in REQUIRED_CONDITIONS):
                ok = False
                break
        if ok:
            eligible.append(sid)
    return eligible


def score_sample(by_model: dict[str, dict[str, dict]], sample_id: str) -> float:
    """Higher score = better demo candidate.

    +2 per model whose b-arm matches GT (criterion 1)
    +3 per model whose a-arm equals the anchor value (criterion 2)
    +2 per model whose m-arm equals GT (criterion 3, digit-pixel control fires)
    +1 if at least 3 models adopt anchor on a (the headline pattern)
    """
    score = 0.0
    pulled = 0
    for mid in MAIN_PANEL:
        s = by_model[mid][sample_id]
        gt = s["meta"]["gt"]
        anchor = s["meta"]["anchor"]
        if gt is not None and s["b"] == gt:
            score += 2
        if anchor is not None and s["a"] == anchor:
            score += 3
            pulled += 1
        if gt is not None and s["m"] == gt:
            score += 2
    if pulled >= 3:
        score += 1
    return score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--inputs-root", type=Path, default=PROJECT_ROOT / "inputs")
    p.add_argument("--site-root", type=Path, default=PROJECT_ROOT / "site")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--anchor-stratum", default="S1",
                   help="Stratum suffix used for the anchor (a) and masked (m) conditions.")
    p.add_argument("--max-image-px", type=int, default=768,
                   help="Long-edge pixel cap for copied images.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"build_demo_data: outputs={args.outputs_root} site={args.site_root}", file=sys.stderr)
    # Pipeline implemented in subsequent tasks.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
