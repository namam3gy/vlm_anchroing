"""E6 — anchor-agnostic steering-vector mitigation.

Phase 0 (calibrate): for each calibration sid (matched to E5c S1 anchor + masked
pair), run two forward passes (a-S1 and m-S1), capture the residual stream at
the last input token across all LLM decoder layers, and compute two `v` vectors:

  v_wrong[L] = mean over wrong-base sids of  ( h_anchor[L] − h_masked[L] )
  v_all  [L] = mean over all calibration sids of  ( h_anchor[L] − h_masked[L] )

Wrong-base sids are read from an existing E5c VQAv2 run's predictions.jsonl
(`condition == 'target_only' AND exact_match == 0`). Calibration pairs are
generated from `configs/experiment_e5c_vqa.yaml` with the same seed, which
deterministically reproduces the (sid → irrelevant_image) assignment of the
E5c run; the model only sees images already on disk.

Output:
  outputs/e6_steering/<model>/calibration/v.pt              # (2, n_layers, d_model), [v_wrong, v_all]
  outputs/e6_steering/<model>/calibration/v_meta.json
  outputs/e6_steering/<model>/calibration/norms_per_layer.csv

Usage (full PoC, ~15-20 min H200 on llava-next-interleaved-7b):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_steering_vector.py \\
        --phase calibrate \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --e5c-run-dir outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331

Smoke (2 sids, ~30 s wall):
    [add --max-samples 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from vlm_anchor.data import (
    assign_stratified_anchors,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.models import InferenceConfig
from vlm_anchor.utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_anchor_ablation import _get_llm_layers  # noqa: E402
from extract_attention_mass import build_eager_runner  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CALIB_CONDITIONS = {
    "target_plus_irrelevant_number_S1",
    "target_plus_irrelevant_number_masked_S1",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("calibrate",), required=True)
    ap.add_argument("--model", required=True,
                    help="Project model id (e.g. llava-next-interleaved-7b)")
    ap.add_argument("--hf-model", required=True,
                    help="HuggingFace model id passed to AutoProcessor / AutoModelForImageTextToText")
    ap.add_argument("--e5c-run-dir", required=True,
                    help="Path to outputs/experiment_e5c_vqa/<model>/<timestamp>/. "
                         "predictions.jsonl is read to identify wrong-base sids.")
    ap.add_argument("--config", default="configs/experiment_e5c_vqa.yaml",
                    help="Must match the config used by the source E5c run "
                         "(same seed + same vqa_dataset slice → same sid → image mapping)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap calibration sids (smoke testing).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _read_wrong_base_sids(e5c_run_dir: Path) -> set[str]:
    """Read predictions.jsonl. Return sids where target_only exact_match == 0.

    Note: `exact_match` is `0/1` int (not bool), and `parsed_number` may be
    `None` in older runs — see E6 design doc status block.
    """
    p = e5c_run_dir / "predictions.jsonl"
    sids: set[str] = set()
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") == "target_only" and r.get("exact_match") == 0:
                sids.add(str(r["sample_instance_id"]))
    return sids


def _load_calibration_samples(args: argparse.Namespace, config: dict) -> list[dict]:
    """Load + stratify samples to match E5c (same seed + config → same sids/images)."""
    vqa_cfg = config["vqa_dataset"]
    inputs_cfg = config["inputs"]
    samples = load_number_vqa_samples(
        dataset_path=PROJECT_ROOT / vqa_cfg["local_path"],
        max_samples=vqa_cfg.get("max_samples"),
        require_single_numeric_gt=vqa_cfg.get("require_single_numeric_gt", True),
        answer_range=vqa_cfg.get("answer_range"),
        samples_per_answer=vqa_cfg.get("samples_per_answer"),
    )
    enriched = assign_stratified_anchors(
        samples,
        irrelevant_number_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_dir"],
        irrelevant_number_masked_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_masked_dir"],
        irrelevant_neutral_dir=PROJECT_ROOT / inputs_cfg.get("irrelevant_neutral_dir"),
        seed=args.seed,
    )
    if args.max_samples:
        enriched = enriched[: args.max_samples]
    return enriched


@torch.no_grad()
def _capture_last_token_residuals(runner: Any, cond: dict) -> torch.Tensor:
    """Run a single prefill forward pass on (sample, condition); return residuals
    at the LAST input token across ALL decoder layers, shape (n_layers, d_model).

    Uses `output_hidden_states=True` rather than forward hooks. HF VLM forward
    propagates this flag to the LLM, and `out.hidden_states` is a tuple of
    `n_layers + 1` tensors of shape `(1, seq_len_after_image_splice, d_model)`.
    `[0]` is the post-embedding pre-layer-0 residual; `[1:]` are the outputs
    of decoder layers 0..n_layers-1.
    """
    _seq_len, inputs = runner._prepare_inputs(
        question=cond["question"], images=cond["input_images"]
    )
    out = runner.model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states
    # Stack outputs of each decoder layer at the last input position;
    # cast to fp32 on CPU so means accumulate cleanly across calls.
    layer_residuals = torch.stack(
        [h[0, -1, :].detach().to(torch.float32).cpu() for h in hidden[1:]]
    )
    return layer_residuals  # (n_layers, d_model)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())

    e5c_run = (PROJECT_ROOT / args.e5c_run_dir).resolve()
    wrong_sids = _read_wrong_base_sids(e5c_run)
    print(f"[setup] wrong-base sids from {e5c_run.name}: {len(wrong_sids)}")

    enriched = _load_calibration_samples(args, config)
    print(f"[setup] enriched samples (calibration sids): {len(enriched)}")

    sampling = config["sampling"]
    prompt_cfg = config["prompt"]
    inference_cfg = InferenceConfig(
        system_prompt=prompt_cfg["system"],
        user_template=prompt_cfg["user_template"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_new_tokens=sampling["max_new_tokens"],
    )
    print(f"[setup] loading {args.hf_model}")
    runner = build_eager_runner(args.hf_model, inference_config=inference_cfg)
    layers = _get_llm_layers(runner.model)
    n_layers = len(layers)
    print(f"[setup] LLM layers detected: {n_layers}")

    out_dir = PROJECT_ROOT / "outputs" / "e6_steering" / args.model / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] writing to {out_dir}")

    diffs_wrong: list[torch.Tensor] = []
    diffs_all: list[torch.Tensor] = []
    n_pairs = 0
    n_skipped = 0
    t0 = time.time()

    for i, sample in enumerate(enriched):
        sid = str(sample["sample_instance_id"])
        a_res, m_res = None, None
        for cond in build_conditions(sample):
            if cond["condition"] not in CALIB_CONDITIONS:
                continue
            try:
                res = _capture_last_token_residuals(runner, cond)
            except Exception as exc:
                print(f"  [error] sid={sid} cond={cond['condition']}: {exc}")
                continue
            if cond["condition"].endswith("masked_S1"):
                m_res = res
            else:
                a_res = res
        if a_res is None or m_res is None:
            n_skipped += 1
            continue
        diff = a_res - m_res
        diffs_all.append(diff)
        if sid in wrong_sids:
            diffs_wrong.append(diff)
        n_pairs += 1
        if n_pairs % 20 == 0 or n_pairs == 1:
            elapsed = time.time() - t0
            eta = elapsed * (len(enriched) - i - 1) / max(n_pairs, 1)
            print(f"  [progress] {n_pairs}/{len(enriched)} pairs "
                  f"(wrong-base subset: {len(diffs_wrong)}); "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    if not diffs_all:
        raise RuntimeError("no calibration pairs collected — check E5c dir / sids / conditions")
    if not diffs_wrong:
        print("[warn] no wrong-base pairs found; v_wrong falls back to zeros")

    v_all = torch.stack(diffs_all).mean(dim=0)
    v_wrong = (torch.stack(diffs_wrong).mean(dim=0)
               if diffs_wrong else torch.zeros_like(v_all))

    v = torch.stack([v_wrong, v_all], dim=0)  # (2, n_layers, d_model)
    torch.save(v, out_dir / "v.pt")

    sidecar = {
        "model": args.model,
        "hf_model": args.hf_model,
        "n_wrong": len(diffs_wrong),
        "n_all": len(diffs_all),
        "n_skipped": n_skipped,
        "n_layers": n_layers,
        "d_model": int(v.shape[-1]),
        "source_e5c_run": str(args.e5c_run_dir),
        "config": args.config,
        "seed": args.seed,
        "v_index_0": "v_wrong",
        "v_index_1": "v_all",
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "v_meta.json").write_text(json.dumps(sidecar, indent=2))

    with (out_dir / "norms_per_layer.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "norm_v_wrong", "norm_v_all", "cos_wrong_all"])
        for L in range(n_layers):
            nw = float(v_wrong[L].norm().item())
            na = float(v_all[L].norm().item())
            cos = float(
                torch.nn.functional.cosine_similarity(
                    v_wrong[L].unsqueeze(0), v_all[L].unsqueeze(0), dim=1
                ).item()
            )
            w.writerow([L, f"{nw:.4f}", f"{na:.4f}", f"{cos:.4f}"])

    elapsed = time.time() - t0
    print(f"[done] n_wrong={len(diffs_wrong)} n_all={len(diffs_all)} "
          f"n_layers={n_layers} d_model={v.shape[-1]} wall={elapsed:.1f}s")
    print(f"[done] saved {out_dir / 'v.pt'}")


if __name__ == "__main__":
    main()
