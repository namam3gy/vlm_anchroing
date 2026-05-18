"""Diagnostic C: logit margin baseline vs mitigation on Qwen2.5-VL.

For a small subset of ChartQA wrong-base a-arm sids where Stage-4 showed
baseline == mitigation predictions, compare top-K next-token logits at
each generation step with and without the L=26 K=8 α=1.0 subspace hook.

Output: outputs/e6_steering/qwen2.5-vl-7b-instruct/_diagnostic_logits/
  per_step.jsonl  (one record per (sid, step), top-5 baseline + top-5 mit)
  summary.json    (aggregate stats)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_anchor.hooks import make_subspace_projection_hook  # noqa: E402
from vlm_anchor.models import HFAttentionRunner, InferenceConfig  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-predictions", required=True,
                    help="E5e/E7 predictions.jsonl with input_image_paths + question.")
    ap.add_argument("--subspace-path", required=True,
                    help=".pt file of shape (n_layers, K, d_model).")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-sids", type=int, default=20)
    ap.add_argument("--condition", default="target_plus_irrelevant_number_S1")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--top-k-alt", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="outputs/e6_steering/qwen2.5-vl-7b-instruct/_diagnostic_logits")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    return ap.parse_args()


def _load_records(path: Path, condition: str, max_sids: int) -> list[dict]:
    out = []
    seen = set()
    for line in path.open():
        r = json.loads(line)
        if r.get("condition") != condition:
            continue
        sid = r.get("sample_instance_id")
        if sid in seen:
            continue
        seen.add(sid)
        out.append(r)
        if len(out) >= max_sids:
            break
    return out


def _get_llm_layers(model) -> list:
    """Walk model to find the decoder layer list. Robust to nested wrappers."""
    for path in ("model.language_model.layers", "model.layers",
                 "language_model.model.layers", "model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) >= 1:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate LLM layer list on model")


@torch.no_grad()
def _gen_with_scores(runner: HFAttentionRunner, question: str, image_paths: list[str],
                     max_new_tokens: int) -> dict:
    """Run generate; return list of per-step logits tensors + sequences."""
    images = [Image.open(p) for p in image_paths]
    inputs_cpu = runner.prepare_inputs_cpu(question=question, images=images)
    inputs = {k: (v.to(runner.model.device) if hasattr(v, "to") else v)
              for k, v in inputs_cpu.items()}
    seq_len = inputs["input_ids"].shape[-1]
    out = runner.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    return {
        "scores": out.scores,  # tuple of (1, vocab) per step
        "generated_ids": out.sequences[0, seq_len:].tolist(),
    }


def _topk_record(scores_step: torch.Tensor, tokenizer, top_k: int) -> dict:
    """Extract top-K alternatives + chosen-token info from one step's logit vector."""
    logits = scores_step[0].float()  # (vocab,)
    probs = torch.softmax(logits, dim=-1)
    chosen = int(logits.argmax().item())
    top_logits, top_ids = logits.topk(top_k)
    return {
        "chosen_id": chosen,
        "chosen_text": tokenizer.decode([chosen], skip_special_tokens=False),
        "chosen_logit": float(logits[chosen].item()),
        "chosen_prob": float(probs[chosen].item()),
        "top_k": [
            {
                "id": int(tid.item()),
                "text": tokenizer.decode([int(tid.item())], skip_special_tokens=False),
                "logit": float(tl.item()),
                "prob": float(probs[int(tid.item())].item()),
            }
            for tl, tid in zip(top_logits, top_ids)
        ],
    }


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading {args.hf_model}")
    cfg = InferenceConfig(
        system_prompt=(
            "You are a visual question answering system.\n"
            "Return valid JSON only in the form {\"result\": <number>}.\n"
        ),
        user_template=(
            "Answer the question using the provided image(s).\n"
            "Return JSON only in the form {\"result\": <number>}.\n"
            "Question: {question}\n"
        ),
        temperature=0.0, top_p=1.0, max_new_tokens=16,
    )
    runner = HFAttentionRunner(args.hf_model, cfg,
                                device=args.device, attn_implementation="sdpa")
    tokenizer = getattr(runner.processor, "tokenizer", runner.processor)

    print(f"[setup] loading subspace {args.subspace_path}")
    subspace = torch.load(args.subspace_path, map_location="cpu", weights_only=True).float()
    print(f"[setup] subspace shape: {tuple(subspace.shape)}; using L={args.layer} K={args.K}")
    V_K = subspace[args.layer, :args.K, :]  # (K, d)

    print(f"[setup] reading sids from {args.baseline_predictions}")
    records = _load_records(Path(args.baseline_predictions), args.condition, args.max_sids)
    print(f"[setup] {len(records)} sids to diagnose")

    llm_layers = _get_llm_layers(runner.model)
    print(f"[setup] LLM has {len(llm_layers)} layers; hook target = layer {args.layer}")

    out_path = out_dir / "per_step.jsonl"
    summary = {
        "n_sids": len(records),
        "condition": args.condition,
        "layer": args.layer, "K": args.K, "alpha": args.alpha,
        "argmax_unchanged": 0,
        "argmax_changed": 0,
        "step1_chosen_logit_delta_sum": 0.0,
        "step1_top1_top2_margin_base_sum": 0.0,
        "step1_top1_top2_margin_mit_sum": 0.0,
    }

    with out_path.open("w") as fout:
        for i, r in enumerate(records):
            sid = r["sample_instance_id"]
            q = r["question"]
            paths = r["input_image_paths"]
            print(f"  [{i+1}/{len(records)}] sid={sid} q={q[:50]!r}")

            # Baseline (no hook)
            base = _gen_with_scores(runner, q, paths, args.max_new_tokens)
            # Mit (with subspace hook)
            hook = make_subspace_projection_hook(V_K, args.alpha)
            handle = llm_layers[args.layer].register_forward_hook(hook)
            try:
                mit = _gen_with_scores(runner, q, paths, args.max_new_tokens)
            finally:
                handle.remove()

            # Per-step records
            for step in range(min(len(base["scores"]), len(mit["scores"]))):
                base_step = _topk_record(base["scores"][step], tokenizer, args.top_k_alt)
                mit_step = _topk_record(mit["scores"][step], tokenizer, args.top_k_alt)
                # logit delta on baseline-chosen token
                base_chosen_id = base_step["chosen_id"]
                mit_logit_for_base_chosen = float(
                    mit["scores"][step][0, base_chosen_id].item()
                )
                base_top2 = base_step["top_k"][1]["logit"] if len(base_step["top_k"]) > 1 else None
                mit_top2 = mit_step["top_k"][1]["logit"] if len(mit_step["top_k"]) > 1 else None
                rec = {
                    "sid": sid,
                    "step": step,
                    "argmax_changed": base_step["chosen_id"] != mit_step["chosen_id"],
                    "base_chosen": {"id": base_step["chosen_id"], "text": base_step["chosen_text"],
                                     "logit": base_step["chosen_logit"], "prob": base_step["chosen_prob"]},
                    "mit_chosen": {"id": mit_step["chosen_id"], "text": mit_step["chosen_text"],
                                    "logit": mit_step["chosen_logit"], "prob": mit_step["chosen_prob"]},
                    "mit_logit_for_base_chosen": mit_logit_for_base_chosen,
                    "logit_delta_on_base_chosen": mit_logit_for_base_chosen - base_step["chosen_logit"],
                    "base_top1_top2_margin": (base_step["chosen_logit"] - base_top2) if base_top2 is not None else None,
                    "mit_top1_top2_margin": (mit_step["chosen_logit"] - mit_top2) if mit_top2 is not None else None,
                    "base_top5": base_step["top_k"],
                    "mit_top5": mit_step["top_k"],
                }
                fout.write(json.dumps(rec) + "\n")

                if step == 0:
                    # Aggregate summary on first generated token (most informative)
                    if rec["argmax_changed"]:
                        summary["argmax_changed"] += 1
                    else:
                        summary["argmax_unchanged"] += 1
                    summary["step1_chosen_logit_delta_sum"] += rec["logit_delta_on_base_chosen"]
                    if rec["base_top1_top2_margin"] is not None:
                        summary["step1_top1_top2_margin_base_sum"] += rec["base_top1_top2_margin"]
                    if rec["mit_top1_top2_margin"] is not None:
                        summary["step1_top1_top2_margin_mit_sum"] += rec["mit_top1_top2_margin"]

    n_step1 = summary["argmax_changed"] + summary["argmax_unchanged"]
    summary["mean_step1_chosen_logit_delta"] = summary["step1_chosen_logit_delta_sum"] / max(n_step1, 1)
    summary["mean_step1_top1_top2_margin_base"] = summary["step1_top1_top2_margin_base_sum"] / max(n_step1, 1)
    summary["mean_step1_top1_top2_margin_mit"] = summary["step1_top1_top2_margin_mit_sum"] / max(n_step1, 1)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=== Step 1 (first generated token) summary ===")
    print(f"  n_sids = {n_step1}")
    print(f"  argmax changed (mit ≠ base): {summary['argmax_changed']}/{n_step1} = {100*summary['argmax_changed']/max(n_step1,1):.1f}%")
    print(f"  mean logit Δ on base-chosen token: {summary['mean_step1_chosen_logit_delta']:+.4f}")
    print(f"  mean top1-top2 margin (base):  {summary['mean_step1_top1_top2_margin_base']:.4f}")
    print(f"  mean top1-top2 margin (mit):   {summary['mean_step1_top1_top2_margin_mit']:.4f}")
    print(f"\n[write] {out_path}")
    print(f"[write] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
