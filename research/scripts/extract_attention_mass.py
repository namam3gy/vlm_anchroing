"""Extract per-layer attention mass to image-token spans during VLM generation.

For each (model, sample, condition) tuple, runs inference with `output_attentions=True`
and computes — per layer, per generated step — the share of query attention that goes
to each input region:

  - target image tokens (first <image> span)
  - anchor image tokens (second <image> span, only present in target_plus_irrelevant_number)
  - prompt text tokens (everything else in the prompt)
  - already-generated tokens (only non-empty after step 0)

Output: per-record JSONL under outputs/attention_analysis/<model>/<run>/.

Scope (Phase B / E1): start with HFAttentionRunner-compatible models. FastVLM and
ConvLLaVA use inputs_embeds and need a slightly different splice — handled later if E1
on the standard models proves productive.

Usage:
    uv run python research/scripts/extract_attention_mass.py \\
        --model qwen2.5-vl-7b-instruct \\
        --hf-model Qwen/Qwen2.5-VL-7B-Instruct \\
        --max-samples 100
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
import yaml

from vlm_anchor.data import (
    assign_irrelevant_images,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.models import HFAttentionRunner, InferenceConfig, _BaseRunner
from vlm_anchor.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EagerAttentionRunner(HFAttentionRunner):
    """Variant that loads with attn_implementation='eager' so output_attentions returns weights.

    sdpa silently returns no attention tensors; we MUST use eager for E1.
    """

    def __init__(self, model_name: str, inference_config=None, device=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Bypass parent __init__'s sdpa load.
        _BaseRunner.__init__ = _BaseRunner.__init__  # noop, for clarity
        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation="eager",
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        self.model.eval()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Local label for the model (used as output dir name).")
    parser.add_argument("--hf-model", type=str, required=True, help="HuggingFace repo id.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--susceptibility-csv", type=str,
                        default="research/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--top-decile-n", type=int, default=200,
                        help="Sample N from top-decile susceptibility questions.")
    parser.add_argument("--bottom-decile-n", type=int, default=200,
                        help="Sample N from bottom-decile susceptibility questions.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override total sample cap (for quick smoke). Sums top+bottom if not set.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _select_susceptibility_strata(csv_path: Path, top_n: int, bottom_n: int, seed: int) -> set[int]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    top = df.loc[df["susceptibility_stratum"] == "top_decile_susceptible"].sort_values("mean_moved_closer", ascending=False)
    bottom = df.loc[df["susceptibility_stratum"] == "bottom_decile_resistant"].sort_values("mean_moved_closer", ascending=True)
    top_pick = top.head(top_n)["question_id"].tolist()
    # for bottom, choose randomly because many tied at 0
    bottom_pool = bottom["question_id"].tolist()
    bottom_pick = rng.choice(bottom_pool, size=min(bottom_n, len(bottom_pool)), replace=False).tolist()
    return set(int(q) for q in top_pick) | set(int(q) for q in bottom_pick)


def _find_image_token_spans(input_ids: torch.LongTensor, image_token_id: int) -> list[tuple[int, int]]:
    """Find all maximal runs of image_token_id in input_ids[0]. Returns [(start, end), ...] half-open."""
    ids = input_ids[0].tolist()
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, t in enumerate(ids):
        if t == image_token_id:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(ids)))
    return spans


def _resolve_image_token_id(processor: Any) -> int:
    """Find the integer token id used for image patch positions in the processor."""
    # Most HF VLM processors expose this directly
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(processor, attr, None)
        if isinstance(v, int):
            return v
    # Qwen2.5-VL / Qwen3-VL: image_pad token in the tokenizer
    tok = getattr(processor, "tokenizer", processor)
    for candidate in ("<|image_pad|>", "<image>", "<|vision_start|>"):
        try:
            tid = tok.convert_tokens_to_ids(candidate)
            if isinstance(tid, int) and tid >= 0:
                return tid
        except Exception:
            continue
    raise RuntimeError("Could not resolve image token id from processor")


def _compute_region_mass(
    attention_step: tuple[torch.Tensor, ...],
    region_specs: list[tuple[str, int, int]],
    query_idx: int | None = None,
) -> dict[str, list[float]]:
    """For one generation step, compute mean attention mass to each region per layer.

    `attention_step` is a tuple of (num_layers,) tensors, each [batch, heads, q_len, k_len].
    `query_idx` selects which query position to look at (None = last position).
    Returns: {region_name: [layer-0 mean_mass, layer-1 mean_mass, ...]}
    """
    out = {name: [] for name, _, _ in region_specs}
    for layer_attn in attention_step:
        # shape: [1, heads, q_len, k_len]
        q = -1 if query_idx is None else query_idx
        # Average over heads first (cheap), then sum over key positions in each region
        head_avg = layer_attn[0, :, q, :].float().mean(dim=0)  # [k_len]
        for name, start, end in region_specs:
            if end <= start:
                out[name].append(0.0)
                continue
            # Clip to actual k_len
            s, e = start, min(end, head_avg.shape[0])
            out[name].append(float(head_avg[s:e].sum().item()))
    return out


def _process_sample(
    runner: HFAttentionRunner,
    image_token_id: int,
    sample: dict,
) -> dict[str, Any]:
    """Run one (sample, condition) tuple with output_attentions=True and compute region mass."""
    images = sample["input_images"]
    seq_len, inputs = runner._prepare_inputs(question=sample["question"], images=images)

    image_spans = _find_image_token_spans(inputs["input_ids"], image_token_id)
    n_images = len(images)
    if len(image_spans) != n_images:
        # Some processors split image_pad tokens around vision_start/vision_end markers;
        # if our naive scan fails to match expected count, log and continue with what we found.
        pass

    region_specs: list[tuple[str, int, int]] = []
    if image_spans:
        region_specs.append(("image_target", image_spans[0][0], image_spans[0][1]))
        if len(image_spans) >= 2:
            region_specs.append(("image_anchor", image_spans[1][0], image_spans[1][1]))
        else:
            region_specs.append(("image_anchor", 0, 0))
    else:
        region_specs.append(("image_target", 0, 0))
        region_specs.append(("image_anchor", 0, 0))

    # Prompt-text tokens = everything in the prompt that is NOT an image token
    image_positions = set()
    for s, e in image_spans:
        for i in range(s, e):
            image_positions.add(i)
    text_positions = [i for i in range(seq_len) if i not in image_positions]
    if text_positions:
        # Use slice form via min/max — text positions are NOT contiguous (they wrap around image spans).
        # We aggregate by building a boolean mask later. For now, store as a list of contiguous
        # runs and the total count.
        pass
    n_text_tokens = len(text_positions)

    generate_kwargs = runner._build_generate_kwargs(max_new_tokens=8)
    generate_kwargs["output_attentions"] = True

    with torch.no_grad():
        out = runner.model.generate(**inputs, **generate_kwargs)

    attentions = getattr(out, "attentions", None)
    if not attentions:
        return {"error": "no attentions returned"}
    n_layers = len(attentions[0])
    n_steps = len(attentions)

    # Per generated step: collect mass to each region. For text we compute mass = total - image_target - image_anchor - generated.
    per_step_records: list[dict[str, Any]] = []
    for step_idx, step_attn in enumerate(attentions):
        # last query position attends to: prompt + generated[0..step_idx-1]
        # For step_idx > 0, we want the LAST generated token's attention pattern.
        step_specs = list(region_specs)
        if step_idx > 0:
            step_specs.append(("generated", seq_len, seq_len + step_idx))
        else:
            step_specs.append(("generated", 0, 0))

        masses = _compute_region_mass(step_attn, step_specs, query_idx=None)

        # text mass = 1 - sum(image_target + image_anchor + generated) per layer (sum should be 1.0)
        text_mass = []
        for layer_idx in range(n_layers):
            sum_other = (
                masses["image_target"][layer_idx]
                + masses["image_anchor"][layer_idx]
                + masses["generated"][layer_idx]
            )
            text_mass.append(max(0.0, 1.0 - sum_other))

        per_step_records.append({
            "step": step_idx,
            "n_layers": n_layers,
            "image_target": masses["image_target"],
            "image_anchor": masses["image_anchor"],
            "generated": masses["generated"],
            "text": text_mass,
        })

    # Decode the prediction so we can join with the existing experiment data
    generated = out.sequences[:, seq_len:]
    tokenizer = getattr(runner.processor, "tokenizer", runner.processor)
    decoded = runner.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # Per-step tokens — needed by downstream analysis to find the "answer-digit step"
    # (not step 0, which is usually the opening JSON brace `{`).
    gen_ids = generated[0].tolist()
    per_step_tokens = []
    for step_idx in range(min(n_steps, len(gen_ids))):
        tid = int(gen_ids[step_idx])
        per_step_tokens.append({
            "step": step_idx,
            "token_id": tid,
            "token_text": tokenizer.decode([tid], skip_special_tokens=False),
        })

    return {
        "n_images": n_images,
        "image_spans": image_spans,
        "n_text_tokens": n_text_tokens,
        "seq_len": seq_len,
        "n_steps": n_steps,
        "n_layers": n_layers,
        "decoded": decoded,
        "per_step": per_step_records,
        "per_step_tokens": per_step_tokens,
    }


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text())

    # Load susceptibility-stratified sample set
    susceptibility_csv = PROJECT_ROOT / args.susceptibility_csv
    target_question_ids = _select_susceptibility_strata(
        susceptibility_csv, args.top_decile_n, args.bottom_decile_n, args.seed
    )
    print(f"[setup] {len(target_question_ids)} target questions from susceptibility strata")

    # Load full dataset, then filter to target questions
    vqa_cfg = config["vqa_dataset"]
    samples = load_number_vqa_samples(
        dataset_path=PROJECT_ROOT / vqa_cfg["local_path"],
        max_samples=None,
        require_single_numeric_gt=vqa_cfg.get("require_single_numeric_gt", True),
        answer_range=vqa_cfg.get("answer_range"),
        samples_per_answer=vqa_cfg.get("samples_per_answer"),
        answer_type_filter=vqa_cfg.get("answer_type_filter"),
    )
    samples = [s for s in samples if int(s["question_id"]) in target_question_ids]
    print(f"[setup] {len(samples)} samples after filtering to target questions")

    inputs_cfg = config["inputs"]
    enriched = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_dir"],
        irrelevant_neutral_dir=PROJECT_ROOT / inputs_cfg["irrelevant_neutral_dir"],
        seed=args.seed,
        variants_per_sample=1,  # one variant is enough for attention analysis
    )
    if args.max_samples:
        enriched = enriched[:args.max_samples]
    print(f"[setup] {len(enriched)} sample-instances (1 irrelevant variant each)")

    # Build the runner
    sampling = config["sampling"]
    prompt = config["prompt"]
    inference_cfg = InferenceConfig(
        system_prompt=prompt["system"],
        user_template=prompt["user_template"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_new_tokens=sampling["max_new_tokens"],
    )
    print(f"[setup] loading runner for {args.hf_model} with eager attention")
    runner = EagerAttentionRunner(args.hf_model, inference_config=inference_cfg)
    image_token_id = _resolve_image_token_id(runner.processor)
    print(f"[setup] image_token_id = {image_token_id}")

    # Output dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = PROJECT_ROOT / "outputs" / "attention_analysis" / args.model / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_root / "per_step_attention.jsonl"
    print(f"[setup] writing to {out_jsonl}")

    n_done = 0
    t0 = time.time()
    with out_jsonl.open("w") as fh:
        for sample in enriched:
            for cond in build_conditions(sample):
                try:
                    result = _process_sample(runner, image_token_id, cond)
                except Exception as exc:
                    result = {"error": str(exc)}
                record = {
                    "model": args.model,
                    "sample_instance_id": cond["sample_instance_id"],
                    "question_id": cond["question_id"],
                    "image_id": cond["image_id"],
                    "question": cond["question"],
                    "ground_truth": cond["ground_truth"],
                    "condition": cond["condition"],
                    "irrelevant_type": cond["irrelevant_type"],
                    "anchor_value": cond.get("anchor_value_for_metrics"),
                    **result,
                }
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(enriched) - n_done) / rate if rate > 0 else 0
                print(f"[progress] {n_done}/{len(enriched)} samples ({rate:.2f}/s, ~{remaining:.0f}s left)")

    print(f"[done] {n_done} samples processed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
