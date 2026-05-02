"""Causal test of the per-layer anchor-attention claim.

For each (model, sample, condition), generate twice — once with normal attention
("baseline"), once with attention to the anchor-image-token span zeroed at a target
LLM layer ("ablate_peak"). Compare `direction_follow_rate`, `adoption_rate`, and
`mean_distance_to_anchor` between the two modes. If the per-layer attention signal
identified in E1b is causally responsible for anchor pull, the ablation should
materially reduce direction_follow.

Two sanity controls also supported:
  - ablate_layer0:   ablate at LLM layer 0  (null control — should be near zero effect)
  - ablate_all:      ablate at every LLM layer (upper-bound — full anchor blackout)

Output JSONL schema (one row per sample × condition × mode):
  model, sample_instance_id, question_id, image_id, ground_truth, condition,
  irrelevant_type, anchor_value, mode, ablate_layer, anchor_span,
  decoded, parsed_number

Usage:
    uv run python scripts/causal_anchor_ablation.py \\
        --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \\
        --peak-layer 16 --max-samples 10 --modes baseline,ablate_peak

Model-specific peak layers (from docs/experiments/E1b-per-layer-localisation.md):
  gemma4-e4b: 5 / qwen2.5-vl-7b: 22 / llava-1.5-7b: 16 / internvl3-8b: 14
  convllava-7b: 16 / fastvlm-7b: 22
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from vlm_anchor.data import (
    assign_irrelevant_images,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.models import (
    FASTVLM_IMAGE_TOKEN_INDEX,
    InferenceConfig,
    _to_pil,
)
from vlm_anchor.utils import extract_first_number, set_seed

# Re-use plumbing from the attention extraction script.
from extract_attention_mass import (  # noqa: E402
    EagerAttentionRunner,
    EagerConvLLaVARunner,
    EagerFastVLMRunner,
    _find_image_token_spans,
    _resolve_image_token_id,
    _select_susceptibility_strata,
    _split_onevision_image_run,
    build_eager_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ─── Layer location helper ───────────────────────────────────────────────────────

def _get_llm_layers(model: nn.Module) -> nn.ModuleList:
    """Walk the model hierarchy to find the LLM transformer-layer list.

    Heuristic: pick the longest `nn.ModuleList` whose first child exposes `.self_attn`.
    Vision-tower stacks tend to be shorter (24–26 layers) and (usually) named
    differently — but even if a vision tower also has `.self_attn`, the longest stack
    is the LLM in every model in our panel.
    """
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn"):
                candidates.append((name, module))
    if not candidates:
        raise RuntimeError("could not find LLM transformer layers (no ModuleList with .self_attn)")
    candidates.sort(key=lambda x: -len(x[1]))
    return candidates[0][1]


# ─── Anchor-mask forward pre-hook ────────────────────────────────────────────────

def _make_anchor_mask_hook(anchor_span: tuple[int, int], strength: float = -1e4):
    """Return a forward pre-hook that adds `strength` to `attention_mask` columns
    inside the anchor span, so post-softmax attention to those keys is multiplied
    by exp(strength).

    strength=-1e4 (default) is hard masking — bf16-safe, behaves like -inf without
    NaN propagation in some kernels. This is E1d's behaviour.
    Smaller |strength| values give graded down-weighting (E4 mitigation use).
    Returns None if the anchor span is empty or strength == 0 (caller skips install).
    """
    s, e = anchor_span
    if e <= s:
        return None
    if strength == 0:
        return None

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.dim() == 4:
                    args = list(args)
                    a = a.clone()
                    a[..., s:e] = a[..., s:e] + strength
                    args[i] = a
                    return tuple(args), kwargs
            return None
        mask = mask.clone()
        if mask.dim() == 4:
            mask[..., s:e] = mask[..., s:e] + strength
        elif mask.dim() == 2:
            mask[..., s:e] = mask[..., s:e].masked_fill(mask[..., s:e] >= 0, strength)
        else:
            return None
        kwargs["attention_mask"] = mask
        return args, kwargs

    return hook


def _install_hooks(layers: nn.ModuleList, layer_indices: list[int],
                   anchor_span: tuple[int, int], strength: float = -1e4):
    handles = []
    if anchor_span is None or anchor_span[1] <= anchor_span[0]:
        return handles
    if strength == 0:
        return handles
    hook = _make_anchor_mask_hook(anchor_span, strength=strength)
    if hook is None:
        return handles
    for i in layer_indices:
        if i < 0 or i >= len(layers):
            continue
        h = layers[i].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(h)
    return handles


def _remove_hooks(handles):
    for h in handles:
        h.remove()


# ─── Generation helpers (mode-aware) ─────────────────────────────────────────────

def _resolve_anchor_span(runner, sample: dict, image_token_id: int | None) -> tuple[int, int]:
    """Return (start, end) of the anchor-image span in the sequence the LLM sees.

    For `condition == 'target_only'` (single image) returns (0, 0).
    For multi-image conditions returns the second image span.
    """
    images = sample["input_images"]
    if len(images) < 2:
        return (0, 0)

    if isinstance(runner, EagerAttentionRunner):
        # HF input_ids path — re-prepare inputs and scan for image_token_id
        _, inputs = runner._prepare_inputs(question=sample["question"], images=images)
        spans = _find_image_token_spans(inputs["input_ids"], image_token_id)
        # OneVision concatenates multiple <image> tokens into one big run.
        # Split via image_sizes when it looks like that's happened.
        if (
            len(spans) == 1
            and len(images) > 1
            and type(runner.processor).__name__ == "LlavaOnevisionProcessor"
        ):
            spans = _split_onevision_image_run(
                spans[0], inputs.get("image_sizes"), runner.processor
            )
        return spans[1] if len(spans) >= 2 else (0, 0)

    if isinstance(runner, EagerConvLLaVARunner):
        # Spans known at splice time — replicate the splice arithmetic.
        pil_images = [_to_pil(i) for i in images]
        prompt = runner._build_prompt(sample["question"], num_images=len(pil_images))
        text_chunks = runner._tokenize_with_image_placeholders(prompt, num_images=len(pil_images))
        # Anchor image tokens count = vision tower output length
        # Compute via projector forward — same as runner._encode_images
        with torch.no_grad():
            image_embeds = runner._encode_images(pil_images)
        n_img = image_embeds.shape[1]
        offset = 0
        spans: list[tuple[int, int]] = []
        for idx, chunk in enumerate(text_chunks):
            offset += len(chunk)
            if idx < len(text_chunks) - 1:
                spans.append((offset, offset + n_img))
                offset += n_img
        return spans[1] if len(spans) >= 2 else (0, 0)

    if isinstance(runner, EagerFastVLMRunner):
        # -200 markers expand to N=256 each at forward time — known a priori
        pil_images = [_to_pil(i) for i in images]
        rendered = runner._render_chat(sample["question"], num_images=len(pil_images))
        input_ids = runner._splice_image_tokens(rendered, num_images=len(pil_images))
        n_img_per_image = 256  # FastViT patch count
        offset = 0
        spans: list[tuple[int, int]] = []
        for tok in input_ids[0].tolist():
            if tok == FASTVLM_IMAGE_TOKEN_INDEX:
                spans.append((offset, offset + n_img_per_image))
                offset += n_img_per_image
            else:
                offset += 1
        return spans[1] if len(spans) >= 2 else (0, 0)

    raise RuntimeError(f"unsupported runner type: {type(runner).__name__}")


def _layer_indices_for_mode(mode: str, peak_layer: int, n_layers: int) -> list[int]:
    if mode == "baseline":
        return []
    if mode == "ablate_peak":
        return [peak_layer]
    if mode == "ablate_peak_window":
        # peak ± 2: 5-layer band around the peak (clipped to valid range)
        lo = max(0, peak_layer - 2)
        hi = min(n_layers, peak_layer + 3)
        return list(range(lo, hi))
    if mode == "ablate_lower_half":
        return list(range(0, n_layers // 2))
    if mode == "ablate_upper_half":
        return list(range(n_layers // 2, n_layers))
    if mode == "ablate_layer0":
        return [0]
    if mode == "ablate_all":
        return list(range(n_layers))
    raise ValueError(f"unknown mode: {mode}")


def _generate_one(
    runner,
    sample: dict,
    image_token_id: int | None,
    layers: nn.ModuleList,
    layer_indices: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run a single generation with optional ablation hooks installed for that call only."""
    anchor_span = _resolve_anchor_span(runner, sample, image_token_id)
    handles = _install_hooks(layers, layer_indices, anchor_span) if layer_indices else []
    try:
        out = runner.generate_number(
            question=sample["question"],
            images=sample["input_images"],
            max_new_tokens=max_new_tokens,
        )
    finally:
        _remove_hooks(handles)
    return {
        "anchor_span": anchor_span,
        "decoded": out["raw_text"],
        "parsed_number": out["parsed_number"],
    }


# ─── Main loop ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument("--peak-layer", type=int, required=True,
                        help="LLM layer index to ablate (per docs/experiments/E1b)")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--susceptibility-csv", type=str,
                        default="docs/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--top-decile-n", type=int, default=100)
    parser.add_argument("--bottom-decile-n", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes", type=str,
        default="baseline,ablate_peak,ablate_peak_window,ablate_lower_half,ablate_upper_half,ablate_all",
        help="Comma-separated ablation modes to run."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    susc_path = PROJECT_ROOT / args.susceptibility_csv
    target_qids = _select_susceptibility_strata(
        susc_path, args.top_decile_n, args.bottom_decile_n, args.seed
    )

    vqa_cfg = config["vqa_dataset"]
    samples = load_number_vqa_samples(
        dataset_path=PROJECT_ROOT / vqa_cfg["local_path"],
        max_samples=None,
        require_single_numeric_gt=vqa_cfg.get("require_single_numeric_gt", True),
        answer_range=vqa_cfg.get("answer_range"),
        samples_per_answer=vqa_cfg.get("samples_per_answer"),
        answer_type_filter=vqa_cfg.get("answer_type_filter"),
    )
    samples = [s for s in samples if int(s["question_id"]) in target_qids]
    inputs_cfg = config["inputs"]
    enriched = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_dir"],
        irrelevant_neutral_dir=PROJECT_ROOT / inputs_cfg["irrelevant_neutral_dir"],
        seed=args.seed,
        variants_per_sample=1,
    )
    if args.max_samples:
        enriched = enriched[: args.max_samples]
    print(f"[setup] {len(enriched)} sample-instances")

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
    image_token_id: int | None = None
    if isinstance(runner, EagerAttentionRunner):
        image_token_id = _resolve_image_token_id(runner.processor)
    layers = _get_llm_layers(runner.model)
    n_layers = len(layers)
    print(f"[setup] LLM layers = {n_layers}; peak_layer = {args.peak_layer}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"[setup] modes = {modes}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = PROJECT_ROOT / "outputs" / "causal_ablation" / args.model / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_root / "predictions.jsonl"
    print(f"[setup] writing to {out_jsonl}")

    n_done = 0
    t0 = time.time()
    with out_jsonl.open("w") as fh:
        for sample in enriched:
            for cond in build_conditions(sample):
                for mode in modes:
                    layer_indices = _layer_indices_for_mode(mode, args.peak_layer, n_layers)
                    try:
                        gen = _generate_one(
                            runner, cond, image_token_id, layers, layer_indices, args.max_new_tokens
                        )
                    except Exception as exc:
                        gen = {"error": str(exc)}
                    record = {
                        "model": args.model,
                        "sample_instance_id": cond["sample_instance_id"],
                        "question_id": cond["question_id"],
                        "image_id": cond["image_id"],
                        "ground_truth": cond["ground_truth"],
                        "condition": cond["condition"],
                        "irrelevant_type": cond["irrelevant_type"],
                        "anchor_value": cond.get("anchor_value_for_metrics"),
                        "mode": mode,
                        "ablate_layers": layer_indices,
                        "peak_layer": args.peak_layer,
                        "n_llm_layers": n_layers,
                        **gen,
                    }
                    fh.write(json.dumps(record, default=str) + "\n")
                    fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(enriched) - n_done) / rate if rate > 0 else 0
                print(f"[progress] {n_done}/{len(enriched)} ({rate:.2f}/s, ~{remaining:.0f}s left)")

    print(f"[done] {n_done} samples processed in {time.time() - t0:.1f}s; total records ≈ "
          f"{n_done * 3 * len(modes)}")


if __name__ == "__main__":
    main()
