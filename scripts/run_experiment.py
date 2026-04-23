from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_anchor.data import assign_irrelevant_images, build_conditions, load_number_vqa_samples
from vlm_anchor.metrics import evaluate_sample, summarize_experiment
from vlm_anchor.models import AttentionVisualizationConfig, InferenceConfig, build_model_runner
from vlm_anchor.utils import dump_csv, dump_json, dump_jsonl, ensure_dir, load_yaml, resolve_path, set_seed
from vlm_anchor.visualization import save_attention_panel, save_experiment_analysis_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names from config.")
    return parser.parse_args()


def _safe_path_part(value: str | int | None) -> str:
    text = str(value) if value is not None else "none"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return text.strip("-") or "none"


def _build_attention_key(cond: dict) -> str:
    irrelevant_name = Path(str(cond["irrelevant_image"])).stem if cond.get("irrelevant_image") else "none"
    return "_".join(
        [
            f"img{_safe_path_part(cond['image_id'])}",
            f"q{_safe_path_part(cond['question_id'])}",
            f"set{int(cond.get('sample_instance_index', 0)):02d}",
            _safe_path_part(cond["irrelevant_type"]),
            _safe_path_part(cond["condition"]),
            _safe_path_part(irrelevant_name),
        ]
    )


def _build_attention_output_path(model_out_dir: Path, cond: dict) -> Path:
    return model_out_dir / "attention_maps" / cond["condition"] / f"{_build_attention_key(cond)}.png"



def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
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
    ensure_dir(output_root)

    ds_cfg = cfg["vqa_dataset"]
    max_samples = args.max_samples if args.max_samples is not None else ds_cfg.get("max_samples")
    samples = load_number_vqa_samples(
        dataset_path=resolve_path(ds_cfg["local_path"], base_dir=project_root),
        max_samples=max_samples,
        require_single_numeric_gt=ds_cfg.get("require_single_numeric_gt", True),
        answer_range=ds_cfg.get("answer_range"),
        samples_per_answer=ds_cfg.get("samples_per_answer"),
    )
    samples = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
        irrelevant_neutral_dir=resolve_path(cfg["inputs"]["irrelevant_neutral_dir"], base_dir=project_root),
        seed=cfg["seed"],
        variants_per_sample=int(cfg["inputs"].get("irrelevant_sets_per_sample", 1)),
    )

    inf_cfg = InferenceConfig(
        system_prompt=cfg["prompt"]["system"],
        user_template=cfg["prompt"]["user_template"],
        temperature=float(cfg["sampling"]["temperature"]),
        top_p=float(cfg["sampling"]["top_p"]),
        num_ctx=int(cfg["sampling"]["num_ctx"]),
        max_new_tokens=int(cfg["sampling"]["max_new_tokens"]),
    )

    selected_models = cfg["models"]
    if args.models:
        wanted = set(args.models)
        selected_models = [m for m in selected_models if m["name"] in wanted]

    all_records: list[dict] = []
    vis_cfg = cfg.get("visualization", {})
    vis_enabled = args.visualize or vis_cfg.get("enabled", False)
    attention_vis_cfg = AttentionVisualizationConfig.from_dict(vis_cfg)

    for model_cfg in selected_models:
        model_name = model_cfg["name"]
        print(f"\n=== Running {model_name} ===")
        model_out_dir = ensure_dir(output_root / model_name)
        hf_model = model_cfg.get("hf_model")
        if not hf_model:
            raise ValueError(f"Model {model_name} is missing `hf_model`, which is required for HF-only execution.")
        try:
            runner = build_model_runner(
                hf_model,
                inference_config=inf_cfg,
                attention_visualization_config=attention_vis_cfg,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not initialize HF runner for {model_name}: {exc}") from exc

        records: list[dict] = []
        visualized = 0
        max_cases_cfg = vis_cfg.get("max_cases_per_model", 8)
        max_cases = None if max_cases_cfg in (None, 0) else int(max_cases_cfg)
        warned_attention_unavailable = False

        for sample in tqdm(samples, desc=model_name):
            for cond in build_conditions(sample):
                should_vis = (
                    vis_enabled
                    and getattr(runner, "supports_attention", False)
                    and cond["irrelevant_type"] != "none"
                    and (max_cases is None or visualized < max_cases)
                )
                if vis_enabled and not getattr(runner, "supports_attention", False) and not warned_attention_unavailable:
                    print(f"[WARN] Attention visualization is unavailable for {model_name}; continuing without heatmaps.")
                    warned_attention_unavailable = True
                if should_vis:
                    result = runner.generate_with_attention(
                        cond["question"],
                        cond["input_images"],
                        max_new_tokens=cfg["sampling"]["max_new_tokens"],
                        target_answer_text=cond["ground_truth"],
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
                )
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
                    "anchor_value": cond["anchor_value_for_metrics"],
                    "standard_vqa_accuracy": sample_eval.standard_vqa_accuracy,
                    "exact_match": sample_eval.exact_match,
                    "anchor_adopted": sample_eval.anchor_adopted,
                    "anchor_direction_followed": sample_eval.anchor_direction_followed,
                    "numeric_distance_to_anchor": sample_eval.numeric_distance_to_anchor,
                    "input_image_paths": [str(x) if isinstance(x, (str, Path)) else "<dataset_image>" for x in cond["input_images"]],
                }
                records.append(row)

                if should_vis:
                    try:
                        attention_key = _build_attention_key(cond)
                        save_attention_panel(
                            sample_id=attention_key,
                            question=cond["question"],
                            images=cond["input_images"],
                            heatmaps=result["image_heatmaps"],
                            prediction=result["parsed_number"],
                            output_path=_build_attention_output_path(model_out_dir, cond),
                            attention_tokens=result.get("attention_tokens"),
                        )
                        visualized += 1
                    except Exception as exc:
                        print(
                            f"[WARN] Visualization failed for {model_name} / {cond['question_id']} / "
                            f"{cond['condition']} / {cond.get('irrelevant_image')}: {exc}"
                        )

        summary = summarize_experiment(records)
        dump_jsonl(records, model_out_dir / "predictions.jsonl")
        dump_csv(records, model_out_dir / "predictions.csv")
        dump_json(summary, model_out_dir / "summary.json")
        all_records.extend(records)
        print(summary)

    if all_records:
        save_experiment_analysis_figures(all_records, output_root / "analysis")


if __name__ == "__main__":
    main()
