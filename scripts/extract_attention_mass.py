"""Extract per-layer attention mass to image-token spans during VLM generation.

For each (model, sample, condition) tuple, runs inference with `output_attentions=True`
and computes — per layer, per generated step — the share of query attention that goes
to each input region:

  - target image tokens (first <image> span)
  - anchor image tokens (second <image> span, only present in target_plus_irrelevant_number)
  - prompt text tokens (everything else in the prompt)
  - already-generated tokens (only non-empty after step 0)

When ``--bbox-file`` is supplied (E1-patch), two more regions are emitted alongside:

  - ``image_anchor_digit``      attention to the digit-pixel patches inside the anchor image
  - ``image_anchor_background`` attention to the rest of the anchor image (= anchor − digit)

The bbox-file is a JSON dict ``{anchor_value: {bbox_xyxy, image_size, ...}}`` produced
by ``scripts/compute_anchor_digit_bboxes.py`` (which diffs the original anchor image
against its masked counterpart). The mapping from pixel bbox to patch tokens uses
*normalized* coordinates and the image-span's row-major patch grid (perfect square only;
multi-tile / multi-scale spans are skipped).

Output: per-record JSONL under outputs/attention_analysis/<model>/<run>/.

Dispatches on HF model id:
  - HFAttentionRunner-compatible models (Gemma-SigLIP, Qwen-VL, LLaVA-1.5, InternVL3,
    LLaVA-Interleave): scan input_ids for the processor's image-token id.
  - ConvLLaVA: inputs_embeds path — image spans tracked during splice.
  - FastVLM: input_ids with -200 markers expanded internally — spans back-computed from
    the first attention tensor's shape.

Usage:
    uv run python scripts/extract_attention_mass.py \\
        --model qwen2.5-vl-7b-instruct \\
        --hf-model Qwen/Qwen2.5-VL-7B-Instruct \\
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import math
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
from vlm_anchor.models import (
    FASTVLM_IMAGE_TOKEN_INDEX,
    ConvLLaVARunner,
    FastVLMRunner,
    HFAttentionRunner,
    InferenceConfig,
    _BaseRunner,
    _to_pil,
)
from vlm_anchor.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set by main() from --max-new-tokens. Module-global so _process_sample_* helpers use it
# without threading an extra arg through the dispatcher.
MAX_NEW_TOKENS: int = 8

# E1-patch: digit bbox lookup loaded from --bbox-file. None when feature is off.
_BBOXES: dict[str, dict] | None = None


def _load_bboxes(path: Path | None) -> dict[str, dict] | None:
    """Load bbox JSON if path given. Keys are anchor-value strings; values
    have at least ``bbox_xyxy`` and ``image_size``."""
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _bbox_for_anchor(anchor_value: Any) -> dict | None:
    """Look up bbox info for an anchor value; returns None if unknown."""
    if _BBOXES is None or anchor_value is None:
        return None
    key = str(anchor_value)
    return _BBOXES.get(key)


def _compute_anchor_bbox_mass(
    attention_step: tuple[torch.Tensor, ...],
    image_span: tuple[int, int],
    bbox_info: dict,
) -> list[float] | None:
    """Per-layer attention sum over the digit-bbox token positions inside
    ``image_span``. Uses normalized bbox coords + perfect-square row-major
    grid. Returns None if the span isn't a perfect square (multi-tile /
    -200-marker expansion etc.)."""
    span_start, span_end = image_span
    n_tokens = span_end - span_start
    grid = int(math.isqrt(n_tokens))
    if grid * grid != n_tokens or grid <= 0:
        return None

    W, H = bbox_info["image_size"]
    x0, y0, x1, y1 = bbox_info["bbox_xyxy"]
    px0 = max(0, min(grid - 1, int(x0 * grid / W)))
    py0 = max(0, min(grid - 1, int(y0 * grid / H)))
    px1 = max(px0 + 1, min(grid, math.ceil(x1 * grid / W)))
    py1 = max(py0 + 1, min(grid, math.ceil(y1 * grid / H)))

    out: list[float] = []
    for layer_attn in attention_step:
        head_avg = layer_attn[0, :, -1, :].float().mean(dim=0)  # [k_len]
        total = 0.0
        for py in range(py0, py1):
            row_start = span_start + py * grid + px0
            row_end = span_start + py * grid + px1
            row_end = min(row_end, head_avg.shape[0])
            row_start = min(row_start, row_end)
            if row_end > row_start:
                total += float(head_avg[row_start:row_end].sum().item())
        out.append(total)
    return out


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


class EagerConvLLaVARunner(ConvLLaVARunner):
    """ConvLLaVA variant that loads the LLaMA backbone with attn_implementation='eager'.

    ConvLLaVA builds inputs_embeds directly (no image-token id in input_ids), so image
    spans are tracked at splice time rather than scanned.
    """

    def __init__(self, model_name: str, inference_config=None, device=None):
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            CLIPImageProcessor,
            ConvNextModel,
            LlamaForCausalLM,
        )
        import torch.nn as nn

        # Re-implement parent __init__ with eager-attn on the LLaMA backbone.
        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()

        ckpt_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        vt_name = getattr(ckpt_cfg, "mm_vision_tower", None)
        if vt_name is None:
            raise ValueError(f"{model_name} config missing mm_vision_tower")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        self.model.eval()

        self.vision_tower = ConvNextModel.from_pretrained(vt_name, dtype=dtype).to(self.device)
        self.vision_tower.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(vt_name)

        hidden = ckpt_cfg.hidden_size
        mm_hidden = ckpt_cfg.mm_hidden_size
        projector = nn.Sequential(
            nn.Linear(mm_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        ).to(device=self.device, dtype=dtype)
        self._load_vision_and_projector(projector)
        self.mm_projector = projector
        self.mm_projector.eval()


class EagerFastVLMRunner(FastVLMRunner):
    """FastVLM variant that asks for eager attention on the wrapped Qwen LLM.

    FastVLM's custom LlavaQwen2ForCausalLM expands -200 markers into visual features
    during forward. To see attention weights we must (a) load with eager, and (b)
    back-compute the expanded image spans from the first attention tensor's shape.
    """

    def __init__(self, model_name: str, inference_config=None, device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.cfg = inference_config
        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation="eager",
        )
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        self.model.eval()
        self.image_processor = self.model.get_vision_tower().image_processor


def build_eager_runner(
    hf_model: str,
    inference_config: InferenceConfig,
    device: str | None = None,
) -> _BaseRunner:
    """Dispatch same as models.build_runner, but loading each backbone with eager attention."""
    lower = hf_model.lower()
    if "fastvlm" in lower or "llava-qwen" in lower:
        return EagerFastVLMRunner(hf_model, inference_config=inference_config, device=device)
    if "convllava" in lower:
        return EagerConvLLaVARunner(hf_model, inference_config=inference_config, device=device)
    return EagerAttentionRunner(hf_model, inference_config=inference_config, device=device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Local label for the model (used as output dir name).")
    parser.add_argument("--hf-model", type=str, required=True, help="HuggingFace repo id.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--susceptibility-csv", type=str,
                        default="docs/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--top-decile-n", type=int, default=200,
                        help="Sample N from top-decile susceptibility questions.")
    parser.add_argument("--bottom-decile-n", type=int, default=200,
                        help="Sample N from bottom-decile susceptibility questions.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override total sample cap (for quick smoke). Sums top+bottom if not set.")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Generation length cap. Default 8 matches the 4-model E1 runs. "
                             "Bump for FastVLM (emits prose before the digit under JSON prompt).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox-file", type=Path, default=None,
                        help="JSON of digit-pixel bboxes per anchor value (E1-patch). "
                             "Produced by scripts/compute_anchor_digit_bboxes.py. "
                             "When supplied, per-step record gains image_anchor_digit + "
                             "image_anchor_background fields. Models whose anchor span isn't "
                             "a perfect-square grid (e.g. multi-tile InternVL3, FastVLM -200) "
                             "transparently skip the new fields.")
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


def _build_per_step_records(
    attentions: tuple,
    seq_len: int,
    image_spans: list[tuple[int, int]],
    bbox_info: dict | None = None,
) -> list[dict[str, Any]]:
    """Assemble per-step, per-layer attention-mass records from generate output.

    Shared across all three extraction paths (HF / ConvLLaVA / FastVLM) because the
    structure of `attentions` (a tuple over steps, each a tuple over layers of
    [batch, heads, q_len, k_len] tensors) is identical once spans are known.

    If ``bbox_info`` is provided and the anchor span is a perfect-square grid,
    additionally emits ``image_anchor_digit`` and ``image_anchor_background`` per
    layer per step (E1-patch).
    """
    n_layers = len(attentions[0])
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

    bbox_eligible = (
        bbox_info is not None
        and image_spans is not None
        and len(image_spans) >= 2
    )

    per_step_records: list[dict[str, Any]] = []
    for step_idx, step_attn in enumerate(attentions):
        step_specs = list(region_specs)
        if step_idx > 0:
            step_specs.append(("generated", seq_len, seq_len + step_idx))
        else:
            step_specs.append(("generated", 0, 0))
        masses = _compute_region_mass(step_attn, step_specs, query_idx=None)
        text_mass = []
        for layer_idx in range(n_layers):
            sum_other = (
                masses["image_target"][layer_idx]
                + masses["image_anchor"][layer_idx]
                + masses["generated"][layer_idx]
            )
            text_mass.append(max(0.0, 1.0 - sum_other))

        record = {
            "step": step_idx,
            "n_layers": n_layers,
            "image_target": masses["image_target"],
            "image_anchor": masses["image_anchor"],
            "generated": masses["generated"],
            "text": text_mass,
        }

        if bbox_eligible:
            digit_mass = _compute_anchor_bbox_mass(
                step_attn, image_spans[1], bbox_info
            )
            if digit_mass is not None:
                bg_mass = [
                    max(0.0, anchor - digit)
                    for anchor, digit in zip(masses["image_anchor"], digit_mass)
                ]
                record["image_anchor_digit"] = digit_mass
                record["image_anchor_background"] = bg_mass

        per_step_records.append(record)
    return per_step_records


def _per_step_tokens_from_sequences(
    generated_ids: torch.LongTensor,
    n_steps: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    gen_ids = generated_ids[0].tolist()
    out = []
    for step_idx in range(min(n_steps, len(gen_ids))):
        tid = int(gen_ids[step_idx])
        out.append({
            "step": step_idx,
            "token_id": tid,
            "token_text": tokenizer.decode([tid], skip_special_tokens=False),
        })
    return out


def _process_sample_hf(
    runner: EagerAttentionRunner,
    image_token_id: int,
    sample: dict,
) -> dict[str, Any]:
    """HFAttentionRunner path: scan input_ids for image-token id."""
    images = sample["input_images"]
    seq_len, inputs = runner._prepare_inputs(question=sample["question"], images=images)

    image_spans = _find_image_token_spans(inputs["input_ids"], image_token_id)
    n_images = len(images)

    generate_kwargs = runner._build_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)
    generate_kwargs["output_attentions"] = True

    with torch.no_grad():
        out = runner.model.generate(**inputs, **generate_kwargs)

    attentions = getattr(out, "attentions", None)
    if not attentions:
        return {"error": "no attentions returned"}

    bbox_info = _bbox_for_anchor(sample.get("anchor_value"))
    per_step_records = _build_per_step_records(attentions, seq_len, image_spans, bbox_info=bbox_info)
    generated = out.sequences[:, seq_len:]
    tokenizer = getattr(runner.processor, "tokenizer", runner.processor)
    decoded = runner.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    per_step_tokens = _per_step_tokens_from_sequences(generated, len(attentions), tokenizer)

    image_positions = {i for s, e in image_spans for i in range(s, e)}
    n_text_tokens = seq_len - len(image_positions)
    return {
        "n_images": n_images,
        "image_spans": image_spans,
        "n_text_tokens": n_text_tokens,
        "seq_len": seq_len,
        "n_steps": len(attentions),
        "n_layers": len(attentions[0]),
        "decoded": decoded,
        "per_step": per_step_records,
        "per_step_tokens": per_step_tokens,
    }


def _process_sample_convllava(
    runner: "EagerConvLLaVARunner",
    sample: dict,
) -> dict[str, Any]:
    """ConvLLaVA path: build inputs_embeds manually, track spans at splice time."""
    pil_images = [_to_pil(i) for i in sample["input_images"]]
    question = sample["question"]

    prompt = runner._build_prompt(question, num_images=len(pil_images))
    text_chunks = runner._tokenize_with_image_placeholders(prompt, num_images=len(pil_images))
    image_embeds = runner._encode_images(pil_images) if pil_images else None  # (N, T, hidden)

    embed_layer = runner.model.get_input_embeddings()
    parts: list[torch.Tensor] = []
    image_spans: list[tuple[int, int]] = []
    offset = 0
    for idx, chunk in enumerate(text_chunks):
        chunk_ids = torch.tensor(chunk, device=runner.device, dtype=torch.long)
        parts.append(embed_layer(chunk_ids))
        offset += len(chunk)
        if idx < len(text_chunks) - 1 and image_embeds is not None:
            img_emb = image_embeds[idx]
            parts.append(img_emb)
            image_spans.append((offset, offset + img_emb.shape[0]))
            offset += img_emb.shape[0]

    inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0).to(runner.model.dtype)
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=runner.device)

    generate_kwargs = runner._build_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)
    generate_kwargs["output_attentions"] = True

    with torch.no_grad():
        out = runner.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    attentions = getattr(out, "attentions", None)
    if not attentions:
        return {"error": "no attentions returned (ConvLLaVA)"}

    bbox_info = _bbox_for_anchor(sample.get("anchor_value"))
    per_step_records = _build_per_step_records(attentions, seq_len, image_spans, bbox_info=bbox_info)
    generated = out.sequences  # inputs_embeds path -> only new tokens
    decoded = runner.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    per_step_tokens = _per_step_tokens_from_sequences(generated, len(attentions), runner.tokenizer)

    n_text_tokens = seq_len - sum(e - s for s, e in image_spans)
    return {
        "n_images": len(pil_images),
        "image_spans": image_spans,
        "n_text_tokens": n_text_tokens,
        "seq_len": seq_len,
        "n_steps": len(attentions),
        "n_layers": len(attentions[0]),
        "decoded": decoded,
        "per_step": per_step_records,
        "per_step_tokens": per_step_tokens,
    }


def _process_sample_fastvlm(
    runner: "EagerFastVLMRunner",
    sample: dict,
) -> dict[str, Any]:
    """FastVLM path: -200 markers expand inside model; back-compute spans from attention shape."""
    pil_images = [_to_pil(i) for i in sample["input_images"]]
    question = sample["question"]

    rendered = runner._render_chat(question, num_images=len(pil_images))
    input_ids = runner._splice_image_tokens(rendered, num_images=len(pil_images)).to(
        runner.model.device
    )
    attention_mask = torch.ones_like(input_ids, device=runner.model.device)

    pixel_values = runner.image_processor(images=pil_images, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(runner.model.device, dtype=runner.model.dtype)

    generate_kwargs = runner._build_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)
    generate_kwargs["output_attentions"] = True

    with torch.no_grad():
        out = runner.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            **generate_kwargs,
        )

    attentions = getattr(out, "attentions", None)
    if not attentions:
        return {"error": "no attentions returned (FastVLM)"}

    # Back-compute the per-image expanded patch count from step-0 attention k_len.
    expanded_seq_len = int(attentions[0][0].shape[-1])
    input_ids_list = input_ids[0].tolist()
    num_images = len(pil_images)
    text_ids_len = len(input_ids_list) - num_images  # -200 markers get replaced, not kept
    if num_images > 0:
        expansion_total = expanded_seq_len - text_ids_len
        if expansion_total % num_images != 0:
            return {
                "error": (
                    f"FastVLM expansion not evenly divisible: expanded={expanded_seq_len}, "
                    f"text_ids={text_ids_len}, num_images={num_images}"
                )
            }
        n_img_per_image = expansion_total // num_images
    else:
        n_img_per_image = 0

    image_spans: list[tuple[int, int]] = []
    offset = 0
    for tok in input_ids_list:
        if tok == FASTVLM_IMAGE_TOKEN_INDEX:
            image_spans.append((offset, offset + n_img_per_image))
            offset += n_img_per_image
        else:
            offset += 1
    if offset != expanded_seq_len:
        return {
            "error": (
                f"FastVLM span reconstruction mismatch: offset={offset} vs expanded={expanded_seq_len}"
            )
        }

    bbox_info = _bbox_for_anchor(sample.get("anchor_value"))
    per_step_records = _build_per_step_records(attentions, expanded_seq_len, image_spans, bbox_info=bbox_info)
    generated = out.sequences  # only new tokens per FastVLMRunner convention
    decoded = runner.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    per_step_tokens = _per_step_tokens_from_sequences(generated, len(attentions), runner.tokenizer)

    n_text_tokens = expanded_seq_len - sum(e - s for s, e in image_spans)
    return {
        "n_images": num_images,
        "image_spans": image_spans,
        "n_text_tokens": n_text_tokens,
        "seq_len": expanded_seq_len,
        "n_steps": len(attentions),
        "n_layers": len(attentions[0]),
        "decoded": decoded,
        "per_step": per_step_records,
        "per_step_tokens": per_step_tokens,
        "n_img_per_image": n_img_per_image,
    }


def _process_sample(
    runner: _BaseRunner,
    image_token_id: int | None,
    sample: dict,
) -> dict[str, Any]:
    """Dispatch on runner type."""
    if isinstance(runner, EagerConvLLaVARunner):
        return _process_sample_convllava(runner, sample)
    if isinstance(runner, EagerFastVLMRunner):
        return _process_sample_fastvlm(runner, sample)
    if image_token_id is None:
        raise RuntimeError("HF path requires image_token_id")
    return _process_sample_hf(runner, image_token_id, sample)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    global MAX_NEW_TOKENS, _BBOXES
    MAX_NEW_TOKENS = args.max_new_tokens
    print(f"[setup] max_new_tokens = {MAX_NEW_TOKENS}")
    if args.bbox_file is not None:
        _BBOXES = _load_bboxes(PROJECT_ROOT / args.bbox_file if not args.bbox_file.is_absolute() else args.bbox_file)
        print(f"[setup] bbox file loaded — n={len(_BBOXES) if _BBOXES else 0}")
    else:
        _BBOXES = None

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
    runner = build_eager_runner(args.hf_model, inference_config=inference_cfg)
    image_token_id: int | None = None
    if isinstance(runner, EagerAttentionRunner):
        image_token_id = _resolve_image_token_id(runner.processor)
        print(f"[setup] image_token_id = {image_token_id}")
    else:
        path_kind = "ConvLLaVA (inputs_embeds)" if isinstance(runner, EagerConvLLaVARunner) else "FastVLM (-200 + expand)"
        print(f"[setup] non-HF path: {path_kind}")

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
