"""One-sample smoke test that loads a model and runs all 3 conditions for 1 sample.

Usage:
    uv run python scripts/smoke_models.py --model llava-1.5-7b
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from vlm_anchor.data import assign_irrelevant_images, build_conditions, load_number_vqa_samples
from vlm_anchor.models import InferenceConfig, build_runner
from vlm_anchor.utils import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--model", type=str, required=True, help="Model name as listed in the config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(project_root / args.config)

    model_cfg = next((m for m in cfg["models"] if m["name"] == args.model), None)
    if model_cfg is None:
        raise SystemExit(f"Model {args.model!r} not found in config")

    ds_cfg = cfg["vqa_dataset"]
    samples = load_number_vqa_samples(
        dataset_path=resolve_path(ds_cfg["local_path"], base_dir=project_root),
        max_samples=1,
        require_single_numeric_gt=ds_cfg.get("require_single_numeric_gt", True),
        answer_range=ds_cfg.get("answer_range"),
        samples_per_answer=ds_cfg.get("samples_per_answer"),
    )
    samples = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
        irrelevant_neutral_dir=resolve_path(cfg["inputs"]["irrelevant_neutral_dir"], base_dir=project_root),
        seed=cfg["seed"],
        variants_per_sample=1,
    )

    inf = InferenceConfig(
        system_prompt=cfg["prompt"]["system"],
        user_template=cfg["prompt"]["user_template"],
        temperature=float(cfg["sampling"]["temperature"]),
        top_p=float(cfg["sampling"]["top_p"]),
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )

    print(f">>> loading {model_cfg['hf_model']}", flush=True)
    runner = build_runner(model_cfg["hf_model"], inference_config=inf)

    results: list[dict] = []
    for sample in samples:
        for cond in build_conditions(sample):
            out = runner.generate_number(
                cond["question"], cond["input_images"], max_new_tokens=inf.max_new_tokens
            )
            row = {
                "condition": cond["condition"],
                "ground_truth": cond["ground_truth"],
                "anchor": cond["anchor_value_for_metrics"],
                "raw": out["raw_text"],
                "parsed": out["parsed_number"],
                "answer_token_text": out.get("answer_token_text"),
                "answer_token_probability": out.get("answer_token_probability"),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    del runner
    gc.collect()
    torch.cuda.empty_cache()
    print(">>> done", flush=True)


if __name__ == "__main__":
    main()
