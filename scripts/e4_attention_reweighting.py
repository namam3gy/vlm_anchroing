"""E4 — attention re-weighting mitigation.

Two phases share the same script:
  --phase sweep  : 7 strengths × n=200 stratified samples × 3 conditions
  --phase full   : 1 strength × full VQAv2 number subset × 3 conditions × {baseline, mode}

Resumable: re-running the same command picks up where the previous one stopped
by reading the existing predictions.jsonl and skipping completed
(sample_instance_id, condition, mask_strength) keys.

Usage (Phase 1, single model):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \\
        --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \\
        --phase sweep

Usage (Phase 2, after Phase 1 picks s*):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \\
        --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \\
        --phase full --strength -1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from vlm_anchor.data import (
    assign_irrelevant_images,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.models import InferenceConfig
from vlm_anchor.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_anchor_ablation import (  # noqa: E402
    _get_llm_layers,
    _install_hooks,
    _resolve_anchor_span,
)
from extract_attention_mass import (  # noqa: E402
    EagerAttentionRunner,
    _resolve_image_token_id,
    _select_susceptibility_strata,
    build_eager_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_STRENGTHS: list[float] = [0.0, -0.5, -1.0, -2.0, -3.0, -5.0, -1e4]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--phase", choices=("sweep", "full"), required=True)
    parser.add_argument("--strength", type=float, default=None,
                        help="Required for --phase full; ignored for --phase sweep "
                             "(which iterates the canonical SWEEP_STRENGTHS list).")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--susceptibility-csv",
                        default="docs/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--top-decile-n", type=int, default=100)
    parser.add_argument("--bottom-decile-n", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples (smoke testing). Phase 1 already capped at n=200.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "full" and args.strength is None:
        parser.error("--phase full requires --strength")
    return args


def _output_path(args: argparse.Namespace) -> Path:
    """Canonical output path per (model, phase). Single file accumulates across resumes."""
    sub = "sweep_n200" if args.phase == "sweep" else "full_n17730"
    return PROJECT_ROOT / "outputs" / "e4_mitigation" / args.model / sub / "predictions.jsonl"


def _load_completed_keys(path: Path) -> set[tuple[str, str, float]]:
    """Read existing JSONL, return set of completed (sample_instance_id, condition, strength)
    tuples. Robust to missing file, empty file, and a truncated trailing line."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str, float]] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                completed.add((
                    str(r["sample_instance_id"]),
                    str(r["condition"]),
                    float(r["mask_strength"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _load_samples(args: argparse.Namespace, config: dict) -> list[dict]:
    """Load and enrich samples per phase.

    sweep: same n=200 stratified set as E1b/E1d (top decile susceptible × 100 +
           bottom decile resistant × 100), 1 irrelevant variant per sample.
    full:  full VQAv2 number subset per configs/experiment.yaml; uses the
           configured `irrelevant_sets_per_sample`.
    """
    vqa_cfg = config["vqa_dataset"]
    inputs_cfg = config["inputs"]
    samples = load_number_vqa_samples(
        dataset_path=PROJECT_ROOT / vqa_cfg["local_path"],
        max_samples=None,
        require_single_numeric_gt=vqa_cfg.get("require_single_numeric_gt", True),
        answer_range=vqa_cfg.get("answer_range"),
        samples_per_answer=vqa_cfg.get("samples_per_answer"),
        answer_type_filter=vqa_cfg.get("answer_type_filter"),
    )
    if args.phase == "sweep":
        susc_path = PROJECT_ROOT / args.susceptibility_csv
        target_qids = _select_susceptibility_strata(
            susc_path, args.top_decile_n, args.bottom_decile_n, args.seed
        )
        samples = [s for s in samples if int(s["question_id"]) in target_qids]
        variants = 1
    else:  # full
        variants = inputs_cfg.get("irrelevant_sets_per_sample", 5)

    enriched = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_dir"],
        irrelevant_neutral_dir=PROJECT_ROOT / inputs_cfg["irrelevant_neutral_dir"],
        seed=args.seed,
        variants_per_sample=variants,
    )
    if args.max_samples:
        enriched = enriched[: args.max_samples]
    return enriched


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    enriched = _load_samples(args, config)
    print(f"[setup] phase={args.phase} model={args.model} "
          f"sample_instances={len(enriched)} strength={args.strength}")


if __name__ == "__main__":
    main()
