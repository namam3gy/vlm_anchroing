from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import torch.utils.data

from vlm_anchor.data import (
    ANCHOR_DISTANCE_STRATA,
    assign_irrelevant_images,
    assign_stratified_anchors,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.metrics import evaluate_sample, summarize_experiment
from vlm_anchor.models import InferenceConfig, build_runner
from vlm_anchor.utils import dump_csv, dump_json, dump_jsonl, ensure_dir, load_yaml, resolve_path, set_seed
from vlm_anchor.visualization import save_experiment_analysis_figures


class _PrefetchPrepDataset(torch.utils.data.Dataset):
    """DataLoader-fed CPU preprocessing. Runs build_conditions +
    runner.prepare_inputs_cpu in worker processes so model.generate() in the
    main thread overlaps with image processing / tokenization on the next
    sample. Only safe with HFAttentionRunner (other runners route through
    a sequential fallback)."""

    def __init__(self, samples: list, runner) -> None:
        self.samples = samples
        self.runner = runner

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        prepared: list = []
        for cond in build_conditions(sample):
            inputs_cpu = self.runner.prepare_inputs_cpu(
                cond["question"], cond["input_images"]
            )
            prepared.append((cond, inputs_cpu))
        return prepared


def _identity_collate(batch):
    # batch_size=1; unwrap so main loop sees the per-sample list directly.
    return batch[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names from config.")
    # Sharding (multi-GPU fan-out). When --num-shards > 1, samples are sliced
    # round-robin so each shard processes samples[shard_idx::num_shards].
    # Greedy decoding (temperature=0) makes the union byte-identical to a
    # single-process run; merging is concat + re-summarize.
    parser.add_argument("--shard-idx", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    # When set, write predictions/summary directly into this directory and
    # skip the default <experiment_root>/<model>/<timestamp>/ layout. Used by
    # the sharded driver to land each shard in its own subdir while sharing
    # one timestamp at the parent level.
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, base_dir=Path.cwd())
    if not config_path.exists():
        config_path = resolve_path(args.config, base_dir=project_root)
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])

    output_root = (
        resolve_path(args.output_root, base_dir=Path.cwd())
        if args.output_root
        else resolve_path(cfg["output_root"], base_dir=project_root)
    )
    experiment_root = ensure_dir(output_root / config_path.stem)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    sharded = args.num_shards is not None and args.num_shards > 1
    if sharded:
        if args.shard_idx is None or not (0 <= args.shard_idx < args.num_shards):
            raise ValueError("--shard-idx must be in [0, num_shards) when --num-shards > 1")
        if args.output_dir is None:
            raise ValueError("--output-dir is required when sharding (driver provides shared parent path)")

    ds_cfg = cfg["vqa_dataset"]
    max_samples = args.max_samples if args.max_samples is not None else ds_cfg.get("max_samples")
    samples = load_number_vqa_samples(
        dataset_path=resolve_path(ds_cfg["local_path"], base_dir=project_root),
        max_samples=max_samples,
        require_single_numeric_gt=ds_cfg.get("require_single_numeric_gt", True),
        answer_range=ds_cfg.get("answer_range"),
        samples_per_answer=ds_cfg.get("samples_per_answer"),
        answer_type_filter=ds_cfg.get("answer_type_filter"),
    )
    # Modes: "stratified" (E5b) or anything else (legacy uniform sampling).
    anchor_sampling = cfg["inputs"].get("anchor_sampling", "uniform")
    if anchor_sampling == "stratified":
        # E5c gates extra arms (masked anchors, neutral) behind an explicit
        # cfg flag list; absent → behaves identically to E5b (6 conditions).
        extras = cfg["inputs"].get("stratified_extras", [])
        masked_dir_cfg = cfg["inputs"].get("irrelevant_number_masked_dir") if "masked" in extras else None
        neutral_dir_cfg = cfg["inputs"].get("irrelevant_neutral_dir") if "neutral" in extras else None
        anchor_distance_scheme = cfg["inputs"].get("anchor_distance_scheme", "absolute")
        # Optional explicit absolute strata override: a list of [lo, hi] pairs in
        # YAML. When set, replaces the default ANCHOR_DISTANCE_STRATA — used for
        # single-stratum runs (e.g. E5e TallyQA with [[0, 5]]).
        custom_strata_cfg = cfg["inputs"].get("anchor_distance_strata")
        if custom_strata_cfg is not None:
            strata = [tuple(pair) for pair in custom_strata_cfg]
        else:
            strata = ANCHOR_DISTANCE_STRATA
        samples = assign_stratified_anchors(
            samples,
            irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
            seed=cfg["seed"],
            strata=strata,
            irrelevant_number_masked_dir=(
                resolve_path(masked_dir_cfg, base_dir=project_root) if masked_dir_cfg else None
            ),
            irrelevant_neutral_dir=(
                resolve_path(neutral_dir_cfg, base_dir=project_root) if neutral_dir_cfg else None
            ),
            scheme=anchor_distance_scheme,
        )
    else:
        samples = assign_irrelevant_images(
            samples,
            irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
            irrelevant_neutral_dir=resolve_path(cfg["inputs"]["irrelevant_neutral_dir"], base_dir=project_root),
            seed=cfg["seed"],
            variants_per_sample=int(cfg["inputs"].get("irrelevant_sets_per_sample", 1)),
        )

    # Apply shard slicing AFTER all sample-list construction so every shard
    # observes the identical canonical sample order before slicing.
    if sharded:
        samples = samples[args.shard_idx :: args.num_shards]
        print(
            f"[shard {args.shard_idx}/{args.num_shards}] processing {len(samples)} samples"
        )

    inf_cfg = InferenceConfig(
        system_prompt=cfg["prompt"]["system"],
        user_template=cfg["prompt"]["user_template"],
        temperature=float(cfg["sampling"]["temperature"]),
        top_p=float(cfg["sampling"]["top_p"]),
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )

    selected_models = cfg["models"]
    if args.models:
        wanted = set(args.models)
        selected_models = [m for m in selected_models if m["name"] in wanted]

    if args.output_dir is not None and len(selected_models) != 1:
        raise ValueError(
            "--output-dir requires exactly one selected model (use --models <name>)"
        )

    all_records: list[dict] = []

    import gc
    import torch

    for model_idx, model_cfg in enumerate(selected_models):
        model_name = model_cfg["name"]
        print(f"\n=== Running {model_name} ===")
        if args.output_dir is not None:
            model_out_dir = ensure_dir(resolve_path(args.output_dir, base_dir=Path.cwd()))
        else:
            model_out_dir = ensure_dir(experiment_root / model_name / timestamp)
        hf_model = model_cfg.get("hf_model")
        if not hf_model:
            raise ValueError(f"Model {model_name} is missing `hf_model`, which is required for HF-only execution.")
        runner = build_runner(hf_model, inference_config=inf_cfg)

        records: list[dict] = []

        # DataLoader prefetch path: only when the runner exposes the split API
        # (HFAttentionRunner). Workers pre-build (cond, CPU inputs) tuples while
        # the main thread runs model.generate, overlapping CPU preprocessing
        # with GPU compute. Other runners (FastVLM, ConvLLaVA) fall back to the
        # sequential path because they bundle prep + generate in one call.
        #
        # OFF by default during initial bring-up; set VLM_ENABLE_PREFETCH=1 to
        # turn on. Once validated on a smoke run, can be flipped to default-on.
        import os as _os
        _enable = _os.environ.get("VLM_ENABLE_PREFETCH", "").strip() == "1"
        use_prefetch = (
            _enable
            and hasattr(runner, "prepare_inputs_cpu")
            and hasattr(runner, "generate_from_cpu_inputs")
        )
        if use_prefetch:
            loader = torch.utils.data.DataLoader(
                _PrefetchPrepDataset(samples, runner),
                batch_size=1,
                num_workers=2,
                prefetch_factor=2,
                collate_fn=_identity_collate,
                persistent_workers=False,
            )
            sample_iter = ((cond_inputs, None) for cond_inputs in loader)
        else:
            sample_iter = ((None, sample) for sample in samples)

        for prepared, raw_sample in tqdm(sample_iter, total=len(samples), desc=model_name):
            base_prediction: str | None = None  # reset per sample-instance; target_only fills this in
            cond_iter = (
                prepared if use_prefetch else ((cond, None) for cond in build_conditions(raw_sample))
            )
            for cond, inputs_cpu in cond_iter:
                if use_prefetch:
                    result = runner.generate_from_cpu_inputs(
                        inputs_cpu,
                        max_new_tokens=cfg["sampling"]["max_new_tokens"],
                    )
                else:
                    result = runner.generate_number(
                        cond["question"],
                        cond["input_images"],
                        max_new_tokens=cfg["sampling"]["max_new_tokens"],
                    )
                sample_eval = evaluate_sample(
                    prediction=result["parsed_number"],
                    gt_answer=cond["ground_truth"],
                    all_answers=cond["answers"],
                    anchor_value=cond["anchor_value_for_metrics"],
                    base_prediction=base_prediction,
                )
                if cond["condition"] == "target_only":
                    base_prediction = result["parsed_number"]
                row = {
                    "model": model_name,
                    "sample_instance_id": cond.get("sample_instance_id"),
                    "sample_instance_index": cond.get("sample_instance_index"),
                    "question_id": cond["question_id"],
                    "image_id": cond["image_id"],
                    "question": cond["question"],
                    "question_type": cond["question_type"],
                    "condition": cond["condition"],
                    "irrelevant_type": cond.get("irrelevant_type"),
                    "irrelevant_image": str(cond["irrelevant_image"]) if cond.get("irrelevant_image") else None,
                    "ground_truth": cond["ground_truth"],
                    "answers": cond["answers"],
                    "backend": result["backend"],
                    "raw_prediction": result["raw_text"],
                    "prediction": sample_eval.normalized_prediction,
                    "answer_token_id": result.get("answer_token_id"),
                    "answer_token_text": result.get("answer_token_text"),
                    "answer_token_logit": result.get("answer_token_logit"),
                    "answer_token_probability": result.get("answer_token_probability"),
                    "token_info": result.get("token_info", []),
                    "thinking_marker_present": result.get("thinking_marker_present"),
                    "n_generated_tokens": result.get("n_generated_tokens"),
                    "anchor_value": cond["anchor_value_for_metrics"],
                    "anchor_stratum_id": cond.get("anchor_stratum_id"),
                    "anchor_stratum_range": cond.get("anchor_stratum_range"),
                    "standard_vqa_accuracy": sample_eval.standard_vqa_accuracy,
                    "exact_match": sample_eval.exact_match,
                    "anchor_adopted": sample_eval.anchor_adopted,
                    "anchor_direction_followed": sample_eval.anchor_direction_followed,
                    "anchor_direction_followed_moved": sample_eval.anchor_direction_followed_moved,
                    "pred_b_equal_anchor": sample_eval.pred_b_equal_anchor,
                    "pred_diff_from_base": sample_eval.pred_diff_from_base,
                    "numeric_distance_to_anchor": sample_eval.numeric_distance_to_anchor,
                    "input_image_paths": [str(x) if isinstance(x, (str, Path)) else "<dataset_image>" for x in cond["input_images"]],
                }
                records.append(row)

        summary = summarize_experiment(records)
        dump_jsonl(records, model_out_dir / "predictions.jsonl")
        dump_csv(records, model_out_dir / "predictions.csv")
        dump_json(summary, model_out_dir / "summary.json")
        all_records.extend(records)
        print(summary)

        # Release this model's GPU memory before loading the next checkpoint —
        # 8B BF16 ≈ 18 GB + KV cache; back-to-back loads can OOM otherwise.
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Skip figure rendering for sharded runs — the driver merges predictions
    # and regenerates figures from the unified record set. Also skip when an
    # explicit --output-dir is provided (caller is orchestrating layout).
    if all_records and not sharded and args.output_dir is None:
        save_experiment_analysis_figures(all_records, experiment_root / "analysis" / timestamp)


if __name__ == "__main__":
    main()
