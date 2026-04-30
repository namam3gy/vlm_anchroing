"""E6 Method 4a — CogBias-style activation steering with decode-step correction.

Inspired by: CogBias (arXiv:2604.01366, 2026) — 26-32% reduction on 8 cognitive
biases including anchoring, via dynamic-alpha activation steering.

Key difference from Method 0 (single-direction ActAdd):
  Method 0: correction only at prefill last token (seq_len > 1 guard)
  Method 4a: correction at BOTH prefill last token AND each decode step
  → "dynamic alpha scheduling": the correction follows the generation, not
    just the context encoding.

Sweep grid: L × alpha_prefill × alpha_decode
  - L ∈ {20, 25, 28, 30, 31}  (5 target layers)
  - alpha_prefill ∈ {0.5, 1.0, 2.0}  (prefill-only strength)
  - alpha_decode ∈ {0.0, 0.5, 1.0, 2.0}  (decode-step strength; 0 = Method 0)
  → 5 × 3 × 4 = 60 steered cells + 1 baseline = 61 cells total

Direction: v_general = mean(v_wrong across VQA, TallyQA, ChartQA).
Label-free at inference — same direction applied to any input.

Output:
  outputs/e6_steering/<model>/sweep_cogbias_<dataset>_pooled/predictions.jsonl

Usage:
    # Smoke check on TallyQA
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_cogbias.py \\
        --phase smoke-cogbias \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/experiment_e5e_tallyqa_full/.../predictions.jsonl \\
        --dataset-tag tally

    # Sweep TallyQA n=100
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_cogbias.py \\
        --phase sweep-cogbias \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/experiment_e5e_tallyqa_full/.../predictions.jsonl \\
        --dataset-tag tally --max-sweep-sids 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from vlm_anchor.models import InferenceConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_anchor_ablation import _get_llm_layers  # noqa: E402
from extract_attention_mass import build_eager_runner  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",
                    choices=("smoke-cogbias", "sweep-cogbias"),
                    required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--config", default="configs/experiment_e5c_vqa.yaml")
    ap.add_argument("--predictions-path", default=None)
    ap.add_argument("--dataset-tag", default=None)
    ap.add_argument("--max-sweep-sids", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--calib-tags", default="vqa,tally,chartqa")
    ap.add_argument("--target-cells", default=None,
                    help="Comma-sep cell labels to run.")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _calib_dir(model: str, tag: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / f"calibration_{tag}"


def _sweep_out_path(model: str, dataset_tag: str, out_tag: str | None) -> Path:
    suffix = f"_{out_tag}" if out_tag else ""
    return (PROJECT_ROOT / "outputs" / "e6_steering" / model
            / f"sweep_cogbias_{dataset_tag}{suffix}_pooled" / "predictions.jsonl")


def _build_runner(args) -> Any:
    import yaml
    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)
    sampling = config["sampling"]
    prompt_cfg = config["prompt"]
    inference_cfg = InferenceConfig(
        system_prompt=prompt_cfg["system"],
        user_template=prompt_cfg["user_template"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[setup] loading {args.hf_model}")
    return build_eager_runner(args.hf_model, inference_config=inference_cfg)


def _open_images(paths: list[str]):
    from PIL import Image as _Image
    return [_Image.open(p).convert("RGB") for p in (paths or [])]


def _load_predictions(pred_path: Path) -> dict[str, dict[str, dict]]:
    by_sid: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            by_sid[str(r["sample_instance_id"])][r["condition"]] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return by_sid


def _wrong_sids(by_sid: dict) -> set[str]:
    return {sid for sid, d in by_sid.items()
            if d.get("target_only", {}).get("exact_match") == 0}


def _infer_tag(pred_path: Path) -> str:
    try:
        return pred_path.parents[2].name
    except IndexError:
        return "alt"


# ---------------------------------------------------------------------------
# v_general: mean v_wrong across calibration datasets
# ---------------------------------------------------------------------------

def _load_v_general(model: str, calib_tags: list[str]) -> torch.Tensor:
    """Average v_wrong across datasets → (n_layers, d_model)."""
    vs = []
    for tag in calib_tags:
        p = _calib_dir(model, tag) / "v.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run calibrate-subspace first.")
        v = torch.load(p, map_location="cpu", weights_only=True)  # [2, n_layers, d]
        vs.append(v[0].float())  # v_wrong index 0
    return torch.stack(vs).mean(0)  # [n_layers, d]


# ---------------------------------------------------------------------------
# CogBias hook: fires at both prefill and decode steps
# ---------------------------------------------------------------------------

def _make_cogbias_hook(v_layer: torch.Tensor,
                        alpha_prefill: float,
                        alpha_decode: float):
    """Post-hook: subtract direction at BOTH prefill (last token) AND decode steps.

    - Prefill (seq_len > 1): h[:, -1, :] -= alpha_prefill * v
    - Decode (seq_len == 1): h[:, 0, :] -= alpha_decode * v

    Setting alpha_decode=0 degrades to standard ActAdd (Method 0).
    """
    if alpha_prefill == 0.0 and alpha_decode == 0.0:
        return None

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        v = v_layer.to(device=hidden.device, dtype=hidden.dtype)
        seq_len = hidden.shape[1]
        if seq_len > 1:
            if alpha_prefill != 0.0:
                hidden[:, -1, :] = hidden[:, -1, :] - alpha_prefill * v
        else:
            if alpha_decode != 0.0:
                hidden[:, 0, :] = hidden[:, 0, :] - alpha_decode * v
        return output

    return hook


def _install_cogbias_hook(layers, layer_idx: int, v: torch.Tensor,
                           alpha_prefill: float, alpha_decode: float):
    if alpha_prefill == 0.0 and alpha_decode == 0.0:
        return []
    hook = _make_cogbias_hook(v[layer_idx], alpha_prefill, alpha_decode)
    if hook is None:
        return []
    return [layers[layer_idx].register_forward_hook(hook)]


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

def _cogbias_cells() -> list[dict]:
    L_values = [20, 25, 28, 30, 31]
    alphas_prefill = [0.5, 1.0, 2.0]
    alphas_decode = [0.0, 0.5, 1.0, 2.0]  # 0.0 = pure prefill (Method 0 equivalent)
    cells: list[dict] = [{"L": -1, "alpha_prefill": 0.0,
                           "alpha_decode": 0.0, "label": "baseline"}]
    for L in L_values:
        for ap in alphas_prefill:
            for ad in alphas_decode:
                cells.append({
                    "L": L,
                    "alpha_prefill": ap,
                    "alpha_decode": ad,
                    "label": f"L{L:02d}_ap{ap}_ad{ad}",
                })
    return cells


# ---------------------------------------------------------------------------
# Phase: smoke-cogbias
# ---------------------------------------------------------------------------

def _phase_smoke_cogbias(args) -> None:
    if not args.predictions_path:
        raise ValueError("--phase smoke-cogbias requires --predictions-path")
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path

    dataset_tag = args.dataset_tag or _infer_tag(pred_path)
    by_sid = _load_predictions(pred_path)
    wrong = _wrong_sids(by_sid)
    smoke_sids = [s for s in wrong
                  if "target_plus_irrelevant_number_S1" in by_sid[s]][:5]
    if not smoke_sids:
        raise RuntimeError("No wrong-base a-arm samples found.")

    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    v_general = _load_v_general(args.model, calib_tags)

    runner = _build_runner(args)
    layers = _get_llm_layers(runner.model)

    L = 28
    alpha_prefill = 1.0
    alpha_decode = 1.0
    print(f"[smoke-cogbias] {dataset_tag}: L={L} "
          f"alpha_prefill={alpha_prefill} alpha_decode={alpha_decode}")
    n_changed = 0
    for sid in smoke_sids:
        a_rec = by_sid[sid].get("target_plus_irrelevant_number_S1")
        if not a_rec:
            continue
        imgs = _open_images(a_rec.get("input_image_paths") or [])

        out_base = runner.generate_number(question=a_rec["question"],
                                          images=imgs,
                                          max_new_tokens=args.max_new_tokens)
        handles = _install_cogbias_hook(layers, L, v_general,
                                         alpha_prefill, alpha_decode)
        try:
            out_steered = runner.generate_number(question=a_rec["question"],
                                                  images=imgs,
                                                  max_new_tokens=args.max_new_tokens)
        finally:
            for h in handles:
                h.remove()

        changed = out_base.get("parsed_number") != out_steered.get("parsed_number")
        if changed:
            n_changed += 1
        print(f"  sid={sid}: base={out_base.get('parsed_number')} "
              f"steered={out_steered.get('parsed_number')} changed={changed}")

    result = "PASS" if n_changed > 0 else "FAIL"
    print(f"[smoke-cogbias] {result}: {n_changed}/{len(smoke_sids)} outputs changed")


# ---------------------------------------------------------------------------
# Phase: sweep-cogbias
# ---------------------------------------------------------------------------

def _load_completed_cogbias_keys(path: Path) -> set:
    if not path.exists():
        return set()
    completed: set = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add((
                    str(r["sample_instance_id"]),
                    str(r["condition"]),
                    str(r["cell_label"]),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _phase_sweep_cogbias(args) -> None:
    if not args.predictions_path:
        raise ValueError("--phase sweep-cogbias requires --predictions-path")
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    dataset_tag = args.dataset_tag or _infer_tag(pred_path)

    by_sid = _load_predictions(pred_path)
    wrong = _wrong_sids(by_sid)
    eligible = [
        s for s in wrong
        if "target_only" in by_sid[s]
        and "target_plus_irrelevant_number_S1" in by_sid[s]
    ]
    if args.max_sweep_sids:
        eligible = eligible[:args.max_sweep_sids]
    print(f"[sweep-cogbias] {dataset_tag}: {len(eligible)} sids")

    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    v_general = _load_v_general(args.model, calib_tags)

    cells = _cogbias_cells()
    if args.target_cells:
        keep = {c.strip() for c in args.target_cells.split(",")}
        cells = [c for c in cells if c["label"] in keep]
    print(f"[sweep-cogbias] {len(cells)} cells ({len(cells)-1} steered + 1 baseline)")

    out_path = _sweep_out_path(args.model, dataset_tag, args.out_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_cogbias_keys(out_path)
    print(f"[sweep-cogbias] {len(completed)} records already done")

    runner = _build_runner(args)
    layers = _get_llm_layers(runner.model)

    sweep_conditions = [
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
    ]

    n_written = 0
    t0 = time.time()
    with out_path.open("a") as fout:
        for cell_idx, cell in enumerate(cells):
            L = cell["L"]
            alpha_p = cell["alpha_prefill"]
            alpha_d = cell["alpha_decode"]
            label = cell["label"]
            is_baseline = (label == "baseline")

            if not is_baseline:
                handles = _install_cogbias_hook(layers, L, v_general,
                                                  alpha_p, alpha_d)
            else:
                handles = []

            try:
                for sid in eligible:
                    for cond in sweep_conditions:
                        if cond not in by_sid[sid]:
                            continue
                        key = (sid, cond, label)
                        if key in completed:
                            continue
                        rec = by_sid[sid][cond]
                        try:
                            imgs = _open_images(rec.get("input_image_paths") or [])
                            out = runner.generate_number(
                                question=rec["question"],
                                images=imgs,
                                max_new_tokens=args.max_new_tokens,
                            )
                        except Exception as exc:
                            print(f"  [error] sid={sid} cond={cond}: {exc}")
                            continue

                        row = {
                            "sample_instance_id": sid,
                            "condition": cond,
                            "cell_label": label,
                            "cell_L": L,
                            "cell_alpha_prefill": alpha_p,
                            "cell_alpha_decode": alpha_d,
                            "parsed_number": out.get("parsed_number"),
                            "raw_output": out.get("raw_output"),
                            "anchor_value": rec.get("anchor_value"),
                            "ground_truth": rec.get("ground_truth"),
                        }
                        fout.write(json.dumps(row) + "\n")
                        n_written += 1
            finally:
                for h in handles:
                    h.remove()

            elapsed = time.time() - t0
            print(f"[sweep-cogbias] cell {cell_idx+1}/{len(cells)}: {label}  "
                  f"written={n_written}  elapsed={elapsed:.1f}s")

    print(f"[done] sweep-cogbias: {n_written} new records in {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.phase == "smoke-cogbias":
        _phase_smoke_cogbias(args)
    elif args.phase == "sweep-cogbias":
        _phase_sweep_cogbias(args)


if __name__ == "__main__":
    main()
