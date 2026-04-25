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


def _resolve_upper_half_layers(n_layers: int) -> list[int]:
    return list(range(n_layers // 2, n_layers))


def _exact_match(parsed, ground_truth) -> int:
    if parsed is None or ground_truth is None:
        return 0
    try:
        return int(int(parsed) == int(str(ground_truth).strip()))
    except (ValueError, TypeError):
        return 0


def _generate_one(runner, sample: dict, image_token_id, layers,
                  layer_indices: list[int], strength: float,
                  max_new_tokens: int) -> dict[str, Any]:
    anchor_span = _resolve_anchor_span(runner, sample, image_token_id)
    install_indices = layer_indices if strength != 0 else []
    handles = _install_hooks(layers, install_indices, anchor_span, strength=strength) \
        if install_indices else []
    try:
        out = runner.generate_number(
            question=sample["question"],
            images=sample["input_images"],
            max_new_tokens=max_new_tokens,
        )
    finally:
        for h in handles:
            h.remove()
    return {
        "anchor_span": list(anchor_span),
        "decoded": out["raw_text"],
        "parsed_number": out["parsed_number"],
    }


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    enriched = _load_samples(args, config)

    out_path = _output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(out_path)
    print(f"[setup] phase={args.phase} model={args.model} "
          f"sample_instances={len(enriched)} resuming_from={len(completed)}")

    sampling = config["sampling"]
    prompt = config["prompt"]
    inference_cfg = InferenceConfig(
        system_prompt=prompt["system"],
        user_template=prompt["user_template"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_new_tokens=sampling["max_new_tokens"],
    )
    print(f"[setup] loading {args.hf_model} (eager attention)")
    runner = build_eager_runner(args.hf_model, inference_config=inference_cfg)
    image_token_id = None
    if isinstance(runner, EagerAttentionRunner):
        image_token_id = _resolve_image_token_id(runner.processor)
    layers = _get_llm_layers(runner.model)
    n_layers = len(layers)
    upper_half = _resolve_upper_half_layers(n_layers)
    print(f"[setup] LLM layers={n_layers}; upper_half={upper_half[0]}..{upper_half[-1]}")

    if args.phase == "sweep":
        strengths = SWEEP_STRENGTHS
    else:
        strengths = [0.0, args.strength]
    print(f"[setup] strengths={strengths}")
    print(f"[setup] writing to {out_path}")

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for sample in enriched:
            for cond in build_conditions(sample):
                for strength in strengths:
                    # Phase 2 optimisation: target_only's anchor span is empty,
                    # so the hook is a guaranteed no-op for any non-zero strength.
                    # Phase 1 already verified em_target_only is invariant across
                    # strengths; skip the redundant generations in Phase 2 to
                    # save ~17 % wall time. Phase 1 still runs the full grid for
                    # the sanity check.
                    if (args.phase == "full"
                            and cond["condition"] == "target_only"
                            and strength != 0.0):
                        continue
                    key = (str(cond["sample_instance_id"]), cond["condition"], float(strength))
                    if key in completed:
                        n_skipped += 1
                        continue
                    try:
                        gen = _generate_one(
                            runner, cond, image_token_id, layers,
                            upper_half, strength, args.max_new_tokens,
                        )
                    except Exception as exc:
                        gen = {"error": str(exc), "decoded": None, "parsed_number": None,
                               "anchor_span": None}
                    record = {
                        "model": args.model,
                        "sample_instance_id": cond["sample_instance_id"],
                        "question_id": cond["question_id"],
                        "image_id": cond["image_id"],
                        "ground_truth": cond["ground_truth"],
                        "condition": cond["condition"],
                        "irrelevant_type": cond["irrelevant_type"],
                        "anchor_value": cond.get("anchor_value_for_metrics"),
                        "mask_strength": float(strength),
                        "n_llm_layers": n_layers,
                        "exact_match": _exact_match(gen.get("parsed_number"),
                                                    cond["ground_truth"]),
                        **gen,
                    }
                    fh.write(json.dumps(record, default=str) + "\n")
                    fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (len(enriched) - n_done) / rate if rate > 0 else 0
                print(f"[progress] {n_done}/{len(enriched)} "
                      f"({rate:.2f}/s, ~{remaining:.0f}s left, skipped={n_skipped})")

    print(f"[done] {n_done} samples processed in {time.time() - t0:.1f}s; "
          f"skipped {n_skipped} pre-completed")


if __name__ == "__main__":
    main()
