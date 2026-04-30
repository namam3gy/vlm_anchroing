"""E6 Method 3 — MIA-DPO LoRA fine-tuning for anchor-bias reduction.

Preference pairs:
  prompt  : [target_image, anchor_image] + question  (a-arm condition)
  chosen  : str(ground_truth)               (correct answer)
  rejected: str(anchor_value)               (the distractor number in the image)

Rationale: trains the model to prefer the correct count/value over the
distractor number shown in the irrelevant image, specifically in the
multi-image context. Follows MIA-DPO (arXiv:2410.17637) — preference
learning for multi-image VLMs.

Pipeline phases
---------------
  build-pairs  : Read E5* predictions.jsonl files, extract (prompt, chosen,
                 rejected) triples, save to outputs/e6_dpo/<model>/pairs.jsonl.

  train-dpo    : LoRA DPO fine-tuning with TRL DPOTrainer. Saves adapter to
                 outputs/e6_dpo/<model>/adapter/.

  smoke-adapter: Quick 5-sample eval with adapter loaded to verify EM changes.

  sweep-adapter: Full (n=100 per dataset) sweep with adapter loaded.
                 Outputs predictions.jsonl in the standard format.

Usage:
    # Build pairs
    uv run python scripts/e6_dpo_lora.py \\
        --phase build-pairs --model llava-next-interleaved-7b

    # Train (GPU required, ~4-8 h)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_dpo_lora.py \\
        --phase train-dpo --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf

    # Sweep TallyQA n=100 with adapter
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_dpo_lora.py \\
        --phase sweep-adapter \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/.../predictions.jsonl \\
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",
                    choices=("build-pairs", "train-dpo", "smoke-adapter", "sweep-adapter"),
                    required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--config", default="configs/experiment_e5c_vqa.yaml")
    ap.add_argument("--predictions-path", default=None)
    ap.add_argument("--dataset-tag", default=None)
    ap.add_argument("--max-sweep-sids", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=5000,
                    help="Cap on DPO training pairs (subsampled from pooled datasets).")
    ap.add_argument("--lora-rank", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-train-epochs", type=int, default=1)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--adapter-dir", default=None,
                    help="Path to LoRA adapter (default: outputs/e6_dpo/<model>/adapter)")
    ap.add_argument("--calib-tags", default="vqa,tally,chartqa")
    ap.add_argument("--rejected-mode", default="anchor",
                    choices=("anchor", "case_by_case"),
                    help="anchor: rejected=anchor_value (v1, current). "
                         "case_by_case: rejected=anchor if pred==anchor; "
                         "rejected=pred_a if df_moved=True AND pred!=anchor; "
                         "skip if pred==gt or random-wrong (low signal).")
    ap.add_argument("--out-tag", default=None,
                    help="Suffix for the pairs.jsonl + adapter dir "
                         "(e.g. 'v2' → pairs_v2.jsonl).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _dpo_dir(model: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_dpo" / model


def _adapter_dir(model: str, override: str | None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return _dpo_dir(model) / "adapter"


def _sweep_out_path(model: str, dataset_tag: str, out_tag: str | None = None) -> Path:
    suffix = f"_{out_tag}" if out_tag else ""
    return (PROJECT_ROOT / "outputs" / "e6_steering" / model
            / f"sweep_dpo_{dataset_tag}{suffix}_pooled" / "predictions.jsonl")


# ---------------------------------------------------------------------------
# Phase: build-pairs
# ---------------------------------------------------------------------------

_PRED_PATHS = {
    "tally": "outputs/experiment_e5e_tallyqa_full/llava-next-interleaved-7b/20260427-171240/predictions.jsonl",
    "chartqa": "outputs/experiment_e5e_chartqa_full/llava-next-interleaved-7b/20260427-171240/predictions.jsonl",
    "vqa": "outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331/predictions.jsonl",
}


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


def _phase_build_pairs(args) -> None:
    """Build preference pairs from E5* predictions and save to pairs.jsonl."""
    import random
    random.seed(args.seed)

    out_dir = _dpo_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    pairs_path = out_dir / f"pairs{suffix}.jsonl"

    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    all_pairs: list[dict] = []
    n_skipped_pred_eq_gt = 0
    n_skipped_no_signal = 0
    n_anchor_rejected = 0
    n_pred_rejected = 0

    for tag in calib_tags:
        pred_rel = _PRED_PATHS.get(tag)
        if not pred_rel:
            print(f"  [warn] no pred path for tag={tag!r}; skip")
            continue
        pred_path = PROJECT_ROOT / pred_rel
        if not pred_path.exists():
            print(f"  [warn] {pred_path} not found; skip")
            continue

        by_sid = _load_predictions(pred_path)
        n_tag = 0
        for sid, conds in by_sid.items():
            a = conds.get("target_plus_irrelevant_number_S1")
            b = conds.get("target_only")
            if not a:
                continue
            anchor = a.get("anchor_value")
            gt = a.get("ground_truth")
            if anchor is None or gt is None:
                continue
            if str(anchor) == str(gt):
                continue  # trivial pair — anchor matches gt
            paths = a.get("input_image_paths") or []
            if len(paths) < 2:
                continue
            question = a.get("question", "")
            if not question:
                continue

            if args.rejected_mode == "anchor":
                rejected = str(anchor)
            else:  # case_by_case
                pa = a.get("parsed_number")
                pb = (b.get("parsed_number") if b else None)
                # Need anchor / pred_a as numerics
                try:
                    a_num = float(anchor)
                    pa_num = float(pa) if pa is not None else None
                    pb_num = float(pb) if pb is not None else None
                    gt_num = float(gt)
                except (TypeError, ValueError):
                    continue
                if pa_num is None:
                    continue
                # Skip already-correct samples (no learning signal)
                if pa_num == gt_num:
                    n_skipped_pred_eq_gt += 1
                    continue
                # Direct anchor adoption
                if pa_num == a_num:
                    rejected = str(int(a_num)) if a_num.is_integer() else str(a_num)
                    n_anchor_rejected += 1
                # df_moved (pred drifted toward anchor) → rejected = pred_a
                elif (pb_num is not None
                      and (pa_num - pb_num) * (a_num - pb_num) > 0
                      and pa_num != pb_num):
                    rejected = str(int(pa_num)) if pa_num.is_integer() else str(pa_num)
                    n_pred_rejected += 1
                else:
                    n_skipped_no_signal += 1
                    continue

            all_pairs.append({
                "dataset": tag,
                "sample_instance_id": sid,
                "question": question,
                "image_paths": paths[:2],
                "chosen": str(gt),
                "rejected": rejected,
                "ground_truth": str(gt),
                "anchor_value": str(anchor),
            })
            n_tag += 1
        print(f"  {tag}: {n_tag} pairs extracted from {len(by_sid)} sids")

    if args.rejected_mode == "case_by_case":
        print(f"\n  [case_by_case stats]")
        print(f"    pred==gt skipped (already correct):  {n_skipped_pred_eq_gt}")
        print(f"    rejected=anchor (direct adoption):   {n_anchor_rejected}")
        print(f"    rejected=pred_a (df_moved):          {n_pred_rejected}")
        print(f"    skipped (no signal/random wrong):    {n_skipped_no_signal}")

    if not all_pairs:
        raise RuntimeError("No DPO pairs found. Check predictions paths.")

    # Subsample if over cap
    if len(all_pairs) > args.max_pairs:
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:args.max_pairs]
        print(f"[build-pairs] subsampled to {len(all_pairs)} pairs (cap={args.max_pairs})")
    else:
        random.shuffle(all_pairs)

    with pairs_path.open("w") as fout:
        for p in all_pairs:
            fout.write(json.dumps(p) + "\n")

    print(f"[build-pairs] saved {len(all_pairs)} pairs → {pairs_path}")


# ---------------------------------------------------------------------------
# Phase: train-dpo
# ---------------------------------------------------------------------------

def _phase_train_dpo(args) -> None:
    from PIL import Image as _Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, TaskType
    from trl import DPOConfig, DPOTrainer
    from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
    from datasets import Dataset

    if not args.hf_model:
        raise ValueError("--hf-model required for train-dpo")

    pairs_path = _dpo_dir(args.model) / "pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs.jsonl not found at {pairs_path}. Run build-pairs first.")

    pairs: list[dict] = []
    with pairs_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[train-dpo] loaded {len(pairs)} pairs from {pairs_path}")

    # Load prompt template from config
    import yaml
    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)
    system_prompt = config["prompt"]["system"].strip()
    user_template = config["prompt"]["user_template"].strip()

    def _make_user_text(question: str) -> str:
        return user_template.replace("{question}", question)

    def _build_example(pair: dict) -> dict | None:
        try:
            imgs = [_Image.open(p).convert("RGB") for p in pair["image_paths"]]
        except Exception as exc:
            print(f"  [skip] {pair['sample_instance_id']}: image error {exc}")
            return None

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _make_user_text(pair["question"])},
        ]
        chosen = [{"role": "assistant", "content": '{"result": ' + pair["chosen"] + '}'}]
        rejected = [{"role": "assistant", "content": '{"result": ' + pair["rejected"] + '}'}]
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": imgs}

    print("[train-dpo] building dataset examples ...")
    examples = []
    for p in pairs:
        ex = _build_example(p)
        if ex is not None:
            examples.append(ex)
    print(f"[train-dpo] {len(examples)} valid examples built")

    dataset = Dataset.from_list(examples)

    # Load model
    print(f"[train-dpo] loading model {args.hf_model}")
    processor = AutoProcessor.from_pretrained(args.hf_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    out_dir = _adapter_dir(args.model, args.adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForVisionPreference(processor)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # use implicit reference (KL from LoRA base)
        args=train_args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collator,
        peft_config=lora_cfg,
    )

    print(f"[train-dpo] starting training: {len(dataset)} samples, "
          f"lr={args.lr}, beta={args.beta}, rank={args.lora_rank}, "
          f"epochs={args.num_train_epochs}")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train-dpo] training done in {elapsed:.0f}s")

    trainer.save_model(str(out_dir))
    print(f"[train-dpo] adapter saved to {out_dir}")


# ---------------------------------------------------------------------------
# Phase: sweep-adapter
# ---------------------------------------------------------------------------

def _infer_tag(pred_path: Path) -> str:
    try:
        return pred_path.parents[2].name
    except IndexError:
        return "alt"


def _open_images(paths: list[str]):
    from PIL import Image as _Image
    return [_Image.open(p).convert("RGB") for p in (paths or [])]


def _wrong_sids(by_sid: dict) -> set[str]:
    return {sid for sid, d in by_sid.items()
            if d.get("target_only", {}).get("exact_match") == 0}


def _phase_sweep_adapter(args) -> None:
    """Sweep with the LoRA adapter loaded; writes predictions in standard format."""
    if not args.predictions_path:
        raise ValueError("--phase sweep-adapter requires --predictions-path")
    if not args.hf_model:
        raise ValueError("--hf-model required for sweep-adapter")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    import yaml

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
    print(f"[sweep-adapter] {dataset_tag}: {len(eligible)} sids")

    adapter_path = _adapter_dir(args.model, args.adapter_dir)
    print(f"[sweep-adapter] loading model + adapter from {adapter_path}")
    processor = AutoProcessor.from_pretrained(args.hf_model)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.hf_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)
    system_prompt = config["prompt"]["system"].strip()
    user_template = config["prompt"]["user_template"].strip()

    from vlm_anchor.utils import extract_first_number

    def _generate(question: str, images: list) -> dict:
        user_text = user_template.replace("{question}", question)
        content: list = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if images:
            inputs = processor(text=prompt_text, images=images,
                               return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
        out_ids = ids[0, inputs["input_ids"].shape[1]:]
        raw = processor.decode(out_ids, skip_special_tokens=True)
        pn = extract_first_number(raw)
        return {"raw_output": raw, "parsed_number": pn}

    out_path = _sweep_out_path(args.model, dataset_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_conditions = [
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
    ]

    n_written = 0
    t0 = time.time()
    with out_path.open("w") as fout:
        for si, sid in enumerate(eligible):
            for cond in sweep_conditions:
                if cond not in by_sid[sid]:
                    continue
                rec = by_sid[sid][cond]
                try:
                    imgs = _open_images(rec.get("input_image_paths") or [])
                    out = _generate(rec["question"], imgs)
                except Exception as exc:
                    print(f"  [error] sid={sid} cond={cond}: {exc}")
                    continue
                row = {
                    "sample_instance_id": sid,
                    "condition": cond,
                    "cell_label": "dpo_adapter",
                    "cell_L": -1,
                    "cell_alpha": 1.0,
                    "parsed_number": out.get("parsed_number"),
                    "raw_output": out.get("raw_output"),
                    "anchor_value": rec.get("anchor_value"),
                    "ground_truth": rec.get("ground_truth"),
                }
                fout.write(json.dumps(row) + "\n")
                n_written += 1
            if si % 10 == 0:
                print(f"  [progress] {si+1}/{len(eligible)} sids "
                      f"elapsed={time.time()-t0:.1f}s")

    print(f"[done] sweep-adapter: {n_written} records in {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.phase == "build-pairs":
        _phase_build_pairs(args)
    elif args.phase == "train-dpo":
        _phase_train_dpo(args)
    elif args.phase == "smoke-adapter":
        print("[smoke-adapter] not yet implemented; run sweep-adapter with --max-sweep-sids 5")
    elif args.phase == "sweep-adapter":
        _phase_sweep_adapter(args)


if __name__ == "__main__":
    main()
