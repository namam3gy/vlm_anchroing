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


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    print(f"[setup] phase={args.phase} model={args.model} strength={args.strength}")


if __name__ == "__main__":
    main()
