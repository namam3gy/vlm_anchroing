"""Minimal repro: load OneVision, run 5 forward passes on PlotQA target_only
images, report peak GPU memory per iteration.

Goal: isolate whether the memory growth is in the forward pass itself,
the script's loop, or HF transformer caching.
"""
from __future__ import annotations
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlm_anchor.models import InferenceConfig, build_runner

PREDS = ("outputs/experiment_e5b_5strat_plotqa_onevision/"
         "llava-onevision-qwen2-7b-ov/20260504-075037/predictions.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gpu_mb(stage: str):
    """Report GPU memory in MiB."""
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    print(f"  [{stage:>14s}]  allocated={a:7.1f} MiB  reserved={r:7.1f} MiB")


@torch.inference_mode()
def capture(runner, question, images):
    _seq, inputs = runner._prepare_inputs(question=question, images=images)
    seq_len = inputs["input_ids"].shape[-1]
    print(f"  seq_len after image splice: {seq_len}")
    gpu_mb("after-inputs")
    out = runner.model(**inputs, output_hidden_states=True, use_cache=False)
    gpu_mb("after-forward")
    hidden = out.hidden_states
    result = torch.stack(
        [h[0, -1, :].detach().to(torch.float32).cpu() for h in hidden[1:]]
    )
    del out, hidden, inputs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gpu_mb("after-cleanup")
    return result


def main():
    inference_cfg = InferenceConfig(
        system_prompt='You are a visual question answering system.\n'
                      'Return valid JSON only in the form {"result": <number>}.',
        user_template='Answer the question using the provided image(s).\n'
                      'Return JSON only in the form {"result": <number>}.\n'
                      'Question: {question}',
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=8,
    )

    gpu_mb("startup")
    runner = build_runner("llava-hf/llava-onevision-qwen2-7b-ov-hf",
                          inference_config=inference_cfg,
                          attn_implementation="sdpa")
    gpu_mb("after-load")

    # Sample 5 wrong-base target_only records from PlotQA predictions
    samples = []
    with open(PROJECT_ROOT / PREDS) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == "target_only" and r.get("exact_match") == 0:
                samples.append(r)
                if len(samples) >= 5:
                    break

    for i, r in enumerate(samples, 1):
        print(f"\niter {i}: sid={r['sample_instance_id']}")
        images = [Image.open(p).convert("RGB") for p in (r["input_image_paths"] or [])]
        for img in images:
            print(f"  image size: {img.size}")
        result = capture(runner, r["question"], images)
        print(f"  captured residuals: shape={tuple(result.shape)}")


if __name__ == "__main__":
    main()
