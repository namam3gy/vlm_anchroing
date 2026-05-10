"""P0-1 Phase C: gamma-beta residual-stream bridge inference.

For each (model, condition, sample_instance) in gamma-beta MathVista S1 stimuli,
run generation with per-generated-token residual capture at configured layers,
project onto the calibrated V_K, and emit per-trace amplitude statistics to
JSONL.

Pipeline:
  1. Load V_K[L=33] subspace (calibrated by Phase B
     `e6_steering_vector.py --phase calibrate-subspace`).
  2. Iterate over MathVista S1 stratified samples (a-S1 anchor + d neutral arms).
  3. For each (model x condition x sample_instance) trace, register a forward
     hook on `model.model.language_model.layers[L]` for each L in
     {29, 30, 33, 34} capturing per-generated-token residual.
  4. Project decode-step residuals onto V_K, compute amplitude per token
     (`||V_K^T h_t||_2`).
  5. Aggregate per-trace mean/max + raw per-token sequence at primary layer.

Notes:
  - `attn_implementation="sdpa"` — memory:feedback_sdpa_mask_hook_bug only
    affects pre-hooks that modify kwargs["attention_mask"]; our forward_hook
    on layer *output* is unaffected. Eager OOMs at ~125 GB on multi-image
    MathVista contexts (Q*K^T tensor); SDPA chunks for memory efficiency.
    SDPA bypasses pre-hook attention_mask kwargs; while this script uses only
    forward hooks (safe under SDPA), staying on eager keeps callbacks
    deterministic across the gamma-beta + steering branches.
  - `trust_remote_code=True` is required: Qwen3-VL ships custom config code
    that AutoModelForImageTextToText only loads with the flag set.
  - The script does NOT run on GPU during initial implementation; smoke
    verification (n=5) is gated on Phase B GPU availability.

Usage:
    uv run python scripts/run_gamma_beta_bridge.py \\
        --config configs/p0_1_gamma_beta_bridge.yaml \\
        --models qwen3-vl-8b-instruct \\
        --max-samples 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_anchor.data import (  # noqa: E402
    ANCHOR_DISTANCE_STRATA,
    assign_irrelevant_images,
    assign_stratified_anchors,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.utils import resolve_path, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True,
                   help="YAML config (e.g. configs/p0_1_gamma_beta_bridge.yaml).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override model list from config (subset of names).")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Override condition allowlist from config.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on sample_instances for smoke runs.")
    p.add_argument("--output-tag", default=None,
                   help="Override output directory timestamp tag.")
    return p.parse_args()


def load_subspace_per_layer(
    subspace_path: Path,
    layers: list[int],
    K: int,
) -> dict[int, torch.Tensor]:
    """Returns {layer: V_K shape (d_model, K)}.

    The on-disk subspace tensor has shape (n_layers, K_full, d_model) where
    K_full=16. We slice the top-K rows (Phase B convention: rows are sorted
    by singular value descending) and transpose to (d_model, K) so the
    projection h @ V_K returns shape (..., K).
    """
    sp = torch.load(subspace_path, map_location="cpu", weights_only=False)
    if not isinstance(sp, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor on disk at {subspace_path}, got {type(sp)}"
        )
    if sp.dim() != 3:
        raise ValueError(
            f"Expected 3-D subspace tensor (n_layers, K_full, d_model); got shape {tuple(sp.shape)}"
        )
    n_layers, K_full, d_model = sp.shape
    if K > K_full:
        raise ValueError(
            f"Requested K={K} exceeds K_full={K_full} stored on disk"
        )
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        if not 0 <= L < n_layers:
            raise IndexError(
                f"Layer {L} out of range for subspace with n_layers={n_layers}"
            )
        # sp[L, :K, :]  shape (K, d_model)  -> transpose -> (d_model, K)
        out[L] = sp[L, :K, :].T.contiguous().float()
    return out


def install_per_token_hook(
    layers,
    layer_indices: list[int],
):
    """Register forward hooks on each requested decoder layer.

    Returns (handles, captured) where captured[L] is a list of per-step
    detached CPU tensors. After generation, parts[0] is the prefill
    (shape (1, prompt_len, d_model)) and parts[1:] are decode steps
    (each shape (1, 1, d_model)).
    """
    captured: dict[int, list[torch.Tensor]] = {L: [] for L in layer_indices}

    def make_hook(L: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[L].append(hidden.detach().to("cpu", dtype=torch.float32))
        return hook

    handles = [layers[L].register_forward_hook(make_hook(L)) for L in layer_indices]
    return handles, captured


def collect_decode_residuals(
    captured: dict[int, list[torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """From per-step captures, extract decode-only residuals.

    Convention: captured[L][0] is the prefill (shape (1, prompt_len, d_model));
    captured[L][1:] are decode-step outputs (shape (1, 1, d_model) each). The
    decode output at step i is the residual that produced generated token #i.
    The prefill is dropped — its last position participates in token #1
    only after the (already-captured) decode_1 forward.

    Returns {L: (n_gen, d_model)} stacked tensors. If only the prefill was
    captured (max_new_tokens=0 or generation aborted), returns shape (0, d_model).
    """
    out: dict[int, torch.Tensor] = {}
    for L, parts in captured.items():
        if not parts:
            out[L] = torch.zeros(0, 0)
            continue
        d_model = parts[0].shape[-1]
        if len(parts) < 2:
            out[L] = torch.zeros(0, d_model)
            continue
        # parts[1:] are decode steps (1, 1, d_model)
        decodes = [p[0, -1, :] for p in parts[1:]]
        out[L] = torch.stack(decodes, dim=0)  # (n_gen, d_model)
    return out


def project_amplitude(residuals: torch.Tensor, V_K: torch.Tensor) -> torch.Tensor:
    """Compute per-token projection amplitude onto V_K.

    Args:
        residuals: shape (n_gen, d_model).
        V_K: shape (d_model, K), columns assumed orthonormal (Phase B SVD).

    Returns:
        amplitudes shape (n_gen,) where amplitudes[i] = || V_K^T residuals[i] ||_2.
    """
    proj = residuals @ V_K  # (n_gen, K)
    return proj.norm(dim=1)


def project_coefficients(residuals: torch.Tensor, V_K: torch.Tensor) -> torch.Tensor:
    """Compute per-token raw projection coefficients onto V_K basis.

    Args:
        residuals: shape (n_gen, d_model).
        V_K: shape (d_model, K), columns assumed orthonormal.

    Returns:
        coefficients shape (n_gen, K) where coefs[i, k] = V_K[:, k]^T residuals[i].
        Post-hoc: amplitude(K_subset) = ||coefs[:, K_subset]||_2 along last axis.
    """
    return residuals @ V_K  # (n_gen, K)


def _build_messages(system_prompt: str, user_text: str, image_paths: list) -> list[dict]:
    """Build chat-template messages with embedded image placeholders.

    HFAttentionRunner convention: each image gets a content item
    `{"type": "image"}` (no inline path); the processor consumes the actual
    PIL images via the `images=` kwarg in its __call__.
    """
    image_content = [{"type": "image"} for _ in image_paths]
    content = [*image_content, {"type": "text", "text": user_text}]
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    return messages


def _to_pil(path):
    from PIL import Image
    if hasattr(path, "convert"):
        return path.convert("RGB")
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    cfg_path = resolve_path(args.config, base_dir=Path.cwd())
    if not cfg_path.exists():
        cfg_path = resolve_path(args.config, base_dir=PROJECT_ROOT)
    cfg = yaml.safe_load(cfg_path.read_text())
    set_seed(cfg.get("seed", 42))

    capture_cfg = cfg["capture"]
    layers_to_capture = list(capture_cfg["layers"])
    primary_layer = int(capture_cfg["primary_layer"])
    if primary_layer not in layers_to_capture:
        raise ValueError(
            f"primary_layer={primary_layer} not in capture.layers={layers_to_capture}"
        )
    K = int(capture_cfg["K"])
    subspace_path = resolve_path(capture_cfg["subspace_path"], base_dir=PROJECT_ROOT)

    if not subspace_path.exists():
        print(
            f"[fatal] subspace not found at {subspace_path}\n"
            "        Phase B (calibrate-subspace) must complete first."
        )
        sys.exit(2)

    V_K_per_layer = load_subspace_per_layer(subspace_path, layers_to_capture, K)
    print(
        f"[load] V_K loaded for layers {sorted(V_K_per_layer.keys())} K={K} "
        f"d_model={next(iter(V_K_per_layer.values())).shape[0]}"
    )

    # Sample loading + anchor assignment (mirrors run_experiment.py path)
    ds_cfg = cfg["vqa_dataset"]
    samples = load_number_vqa_samples(
        dataset_path=resolve_path(ds_cfg["local_path"], base_dir=PROJECT_ROOT),
        max_samples=ds_cfg.get("max_samples"),
        require_single_numeric_gt=ds_cfg.get("require_single_numeric_gt", True),
        answer_range=ds_cfg.get("answer_range"),
        samples_per_answer=ds_cfg.get("samples_per_answer"),
        answer_type_filter=ds_cfg.get("answer_type_filter"),
    )

    inputs_cfg = cfg["inputs"]
    anchor_sampling = inputs_cfg.get("anchor_sampling", "uniform")
    if anchor_sampling == "stratified":
        extras = inputs_cfg.get("stratified_extras", [])
        masked_dir_cfg = (
            inputs_cfg.get("irrelevant_number_masked_dir") if "masked" in extras else None
        )
        neutral_dir_cfg = (
            inputs_cfg.get("irrelevant_neutral_dir") if "neutral" in extras else None
        )
        anchor_distance_scheme = inputs_cfg.get("anchor_distance_scheme", "absolute")
        custom_strata_cfg = inputs_cfg.get("anchor_distance_strata")
        if custom_strata_cfg is not None:
            strata = [tuple(pair) for pair in custom_strata_cfg]
        else:
            strata = ANCHOR_DISTANCE_STRATA
        samples = assign_stratified_anchors(
            samples,
            irrelevant_number_dir=resolve_path(
                inputs_cfg["irrelevant_number_dir"], base_dir=PROJECT_ROOT
            ),
            seed=cfg.get("seed", 42),
            strata=strata,
            irrelevant_number_masked_dir=(
                resolve_path(masked_dir_cfg, base_dir=PROJECT_ROOT)
                if masked_dir_cfg
                else None
            ),
            irrelevant_neutral_dir=(
                resolve_path(neutral_dir_cfg, base_dir=PROJECT_ROOT)
                if neutral_dir_cfg
                else None
            ),
            scheme=anchor_distance_scheme,
        )
    else:
        samples = assign_irrelevant_images(
            samples,
            irrelevant_number_dir=resolve_path(
                inputs_cfg["irrelevant_number_dir"], base_dir=PROJECT_ROOT
            ),
            irrelevant_neutral_dir=resolve_path(
                inputs_cfg["irrelevant_neutral_dir"], base_dir=PROJECT_ROOT
            ),
            seed=cfg.get("seed", 42),
            variants_per_sample=int(inputs_cfg.get("irrelevant_sets_per_sample", 1)),
        )

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"[data] {len(samples)} sample instances loaded")

    cond_allow = (
        args.conditions
        or cfg.get("conditions_to_run")
        or ["target_plus_irrelevant_number_S1", "target_plus_irrelevant_neutral"]
    )
    cond_allow_set = set(cond_allow)
    print(f"[cond] allowlist: {cond_allow}")

    model_list = cfg["models"]
    if args.models:
        wanted = set(args.models)
        model_list = [m for m in model_list if m["name"] in wanted]
    if not model_list:
        raise RuntimeError(f"No models matched filter {args.models}")

    timestamp = args.output_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    # `output_root` in the config is already experiment-specific
    # (e.g. outputs/gamma_beta_bridge), so we land traces directly under
    # `<output_root>/<model>/<timestamp>/` rather than nesting another
    # config-stem directory the way run_experiment.py does.
    experiment_root = resolve_path(cfg["output_root"], base_dir=PROJECT_ROOT)

    # Defer torch/transformers heavy imports until after sample list is built —
    # if anything is wrong with config or paths, fail before booting CUDA.
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sampling_cfg = cfg["sampling"]
    max_new_tokens = int(sampling_cfg["max_new_tokens"])
    temperature = float(sampling_cfg.get("temperature", 0.0))
    do_sample = temperature > 0.0
    sys_prompt = cfg["prompt"]["system"]
    user_template = cfg["prompt"]["user_template"]

    for model_cfg in model_list:
        name = model_cfg["name"]
        hf_id = model_cfg["hf_model"]
        out_dir = experiment_root / name / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "amplitude_per_trace.jsonl"
        manifest_path = out_dir / "run_manifest.json"
        print(f"\n[model] {name} ({hf_id}) -> {out_path}")

        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        # SDPA chosen over eager: our hook is register_forward_hook on the LM
        # decoder layer's *output* (residual stream after the layer), not a
        # register_forward_pre_hook on self_attn modifying kwargs["attention_mask"]
        # — the latter is what memory:feedback_sdpa_mask_hook_bug warns against.
        # Eager OOMs at ~125 GB for multi-image MathVista contexts (the explicit
        # Q*K^T attention scores tensor); SDPA chunks attention computation
        # memory-efficiently and produces identical layer output for our purposes.
        model = AutoModelForImageTextToText.from_pretrained(
            hf_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        # Locate the LM decoder layer stack (Qwen3-VL: model.model.language_model.layers).
        lm = getattr(model, "model", model)
        lm = getattr(lm, "language_model", lm)
        layers = lm.layers
        n_layers = len(layers)
        max_required = max(layers_to_capture)
        if n_layers <= max_required:
            raise RuntimeError(
                f"Model has only {n_layers} LM layers; need >= {max_required + 1}"
            )
        d_model_check = next(iter(V_K_per_layer.values())).shape[0]
        print(
            f"[model] LM layer stack length={n_layers}; "
            f"capturing layers {layers_to_capture}; primary={primary_layer}; d_model={d_model_check}"
        )

        manifest = {
            "model": name,
            "hf_model": hf_id,
            "config": str(cfg_path),
            "subspace_path": str(subspace_path),
            "layers_to_capture": layers_to_capture,
            "primary_layer": primary_layer,
            "K": K,
            "conditions_to_run": list(cond_allow),
            "n_sample_instances": len(samples),
            "max_new_tokens": max_new_tokens,
            "timestamp": timestamp,
            "attn_implementation": "sdpa",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        n_done = 0
        t_start = time.time()
        total_traces = len(samples) * len(cond_allow_set)

        with out_path.open("w") as fout:
            for sample in samples:
                for cond in build_conditions(sample):
                    cond_label = cond["condition"]
                    if cond_label not in cond_allow_set:
                        continue
                    sid = cond["sample_instance_id"]

                    image_paths = cond["input_images"]
                    pil_images = [_to_pil(p) for p in image_paths]
                    user_text = user_template.replace("{question}", cond["question"])
                    messages = _build_messages(sys_prompt, user_text, image_paths)
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    proc_inputs = processor(
                        images=pil_images, text=text, return_tensors="pt"
                    )
                    proc_inputs = {
                        k: (v.to(model.device) if hasattr(v, "to") else v)
                        for k, v in proc_inputs.items()
                    }
                    prompt_seq_len = int(proc_inputs["input_ids"].shape[-1])

                    handles, captured = install_per_token_hook(layers, layers_to_capture)
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **proc_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature if do_sample else None,
                                use_cache=True,
                            )
                    finally:
                        for h in handles:
                            h.remove()

                    seq_out = out[0] if out.dim() == 2 else out
                    n_gen = int(seq_out.shape[-1] - prompt_seq_len)
                    raw_text = processor.batch_decode(
                        out[:, prompt_seq_len:], skip_special_tokens=True
                    )[0].strip()

                    per_layer_residuals = collect_decode_residuals(captured)

                    record: dict = {
                        "sample_instance_id": sid,
                        "question_id": cond.get("question_id"),
                        "image_id": cond.get("image_id"),
                        "condition": cond_label,
                        "anchor_value": cond.get("anchor_value_for_metrics"),
                        "anchor_stratum_id": cond.get("anchor_stratum_id"),
                        "ground_truth": cond.get("ground_truth"),
                        "model": name,
                        "n_generated_tokens": n_gen,
                        "raw_text": raw_text,
                        "amplitude_per_layer": {},
                        # Raw per-token coefficients in V_K basis at every captured layer.
                        # Format: {str(L): [[c0..c{K-1}] for token in 0..n_gen-1]}.
                        # Post-hoc: amplitude(K') = sqrt(sum_{k<K'} coef[k]^2) for any K' <= K.
                        # This enables L-sweep + K-sweep without re-running inference.
                        "coefficients_per_layer": {},
                    }
                    for L in layers_to_capture:
                        residuals = per_layer_residuals.get(L)
                        V_K_L = V_K_per_layer[L]
                        if residuals is None or residuals.numel() == 0:
                            record["amplitude_per_layer"][str(L)] = {
                                "mean": None,
                                "max": None,
                                "n": 0,
                            }
                            record["coefficients_per_layer"][str(L)] = []
                            continue
                        amps = project_amplitude(residuals, V_K_L)
                        coefs = project_coefficients(residuals, V_K_L)  # (n_gen, K)
                        entry: dict = {
                            "mean": float(amps.mean().item()),
                            "max": float(amps.max().item()),
                            "n": int(amps.shape[0]),
                        }
                        if L == primary_layer:
                            entry["per_token"] = [float(x) for x in amps.tolist()]
                        record["amplitude_per_layer"][str(L)] = entry
                        # Store raw K=full coefficients for post-hoc L-sweep + K-sweep.
                        record["coefficients_per_layer"][str(L)] = [
                            [round(float(c), 4) for c in row]
                            for row in coefs.tolist()
                        ]

                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    n_done += 1
                    if n_done % 25 == 0:
                        dt = time.time() - t_start
                        rate = n_done / dt if dt > 0 else 0.0
                        eta_s = (total_traces - n_done) / rate if rate > 0 else 0.0
                        print(
                            f"  [{n_done}/{total_traces}] "
                            f"{dt:.1f}s elapsed, {rate:.2f} it/s, ETA {eta_s/60:.1f}min"
                        )

        print(f"[done] {name}: {n_done} traces -> {out_path}")
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[done] all models complete")


if __name__ == "__main__":
    main()
