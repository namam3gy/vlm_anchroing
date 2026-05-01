"""E6 — anchor-agnostic steering-vector mitigation.

Three sub-commands share this script.

  --phase calibrate : Phase 0. Run forward passes on (a-S1, m-S1) calibration
                      pairs, capture residuals at the last input token across
                      all LLM decoder layers, emit v_wrong + v_all.

  --phase smoke     : Phase 0.5. 10-pair wiring smoke for the steering hook.
                      Compares baseline vs. (L=N//2, α=2, v=v_wrong) on
                      wrong-base anchor-arm forward passes. Proves the hook
                      attaches to the right thing (output digit changes;
                      fluency does not explode).

  --phase sweep     : Phase 1. (L × α × v-var) sweep on n=200 stratified
                      samples × 4 conditions (b / a-S1 / m-S1 / d). Sweep
                      grid: 7 L × 3 α × 2 v-var = 42 cells + baseline.
                      Uses susceptibility_strata.csv top-decile +
                      bottom-decile per E1d/E4 convention.

Wrong-base sids are read from an existing E5c VQAv2 run's predictions.jsonl
(`condition == 'target_only' AND exact_match == 0`). Calibration pairs are
generated from `configs/experiment_e5c_vqa.yaml` with the same seed, which
deterministically reproduces the (sid → irrelevant_image) assignment of the
E5c run; the model only sees images already on disk.

Output (Phase 0):
  outputs/e6_steering/<model>/calibration/v.pt              # (2, n_layers, d_model), [v_wrong, v_all]
  outputs/e6_steering/<model>/calibration/v_meta.json
  outputs/e6_steering/<model>/calibration/norms_per_layer.csv

Output (Phase 0.5):
  outputs/e6_steering/<model>/smoke/smoke_results.json

Output (Phase 1):
  outputs/e6_steering/<model>/sweep_n200/predictions.jsonl  (resumable)

Usage:
    # Phase 0 (full calibration, ~4 min H200)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \\
        --phase calibrate --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --e5c-run-dir outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331

    # Phase 0.5 (smoke, ~1 min)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \\
        --phase smoke --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --e5c-run-dir outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331

    # Phase 1 (sweep, ~5-7 h, resumable)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \\
        --phase sweep --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --e5c-run-dir outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
from extract_attention_mass import (  # noqa: E402
    _select_susceptibility_strata,
    build_eager_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CALIB_CONDITIONS = {
    "target_plus_irrelevant_number_S1",
    "target_plus_irrelevant_number_masked_S1",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",
                    choices=("calibrate", "smoke", "sweep", "tiebreaker",
                             "calibrate-subspace", "smoke-subspace", "sweep-subspace"),
                    required=True)
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
                    help="Cap calibration sids (smoke testing). For smoke phase, "
                         "default = 10 wrong-base sids; for sweep phase, default "
                         "= 200 stratified sids.")
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="generate_number budget. Match standard inference (8).")
    ap.add_argument("--susceptibility-csv",
                    default="docs/insights/_data/susceptibility_strata.csv",
                    help="Phase 1 stratified set source.")
    ap.add_argument("--top-decile-n", type=int, default=100)
    ap.add_argument("--bottom-decile-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predictions-path",
                    help="Phase tiebreaker only: path to a non-VQAv2 E5* "
                         "predictions.jsonl (TallyQA / ChartQA / MathVista) "
                         "from which (sid, condition) records are read; "
                         "input_image_paths field reconstructs the (target, "
                         "irrelevant) image pair without re-loading the dataset.")
    ap.add_argument("--cells", default="30:1.0:0,30:4.0:0",
                    help="Phase tiebreaker only: comma-separated L:alpha:v_var_idx "
                         "tuples specifying steered cells to compare against "
                         "baseline. Default: chosen L30/α=1/v_wrong + alternate "
                         "L30/α=4/v_wrong.")
    ap.add_argument("--dataset-tag", default=None,
                    help="Phase tiebreaker / calibrate-from-predictions: short "
                         "label for the output dir (e.g. 'tally', 'chartqa'). "
                         "Defaults to the parent experiment dir name parsed "
                         "from --predictions-path.")
    ap.add_argument("--calibration-tag", default=None,
                    help="Phase tiebreaker only: load v.pt from "
                         "calibration_<tag>/ instead of the default "
                         "calibration/ (VQAv2). Used for cross-direction "
                         "transfer experiments.")
    ap.add_argument("--max-calibrate-pairs", type=int, default=400,
                    help="calibrate-subspace: cap on D matrix rows per dataset.")
    ap.add_argument("--subspace-path", default=None,
                    help="smoke-subspace / sweep-subspace: path to precomputed "
                         "subspace .pt (n_layers, K_max, d_model).")
    ap.add_argument("--sweep-layers", default=None,
                    help="Comma-sep layer indices for sweep-subspace "
                         "(default [16,22,28,30,31]).")
    ap.add_argument("--sweep-ks", default=None,
                    help="Comma-sep K values (default [2,4,8,16]).")
    ap.add_argument("--sweep-alphas", default=None,
                    help="Comma-sep alpha values (default [0.5,1.0,2.0,4.0]).")
    ap.add_argument("--subspace-scope", default="pooled",
                    help="Label for the subspace scope; written into records and "
                         "output directory name (e.g. 'pooled', 'vqa').")
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


def _make_residual_offset_hook(v_layer: torch.Tensor, alpha: float):
    """Forward post-hook on an LLM decoder layer.

    On the prefill forward (seq_len > 1), subtract `alpha * v_layer` from the
    residual at the last input position. On decode steps (seq_len == 1) this
    is a no-op — the offset's downstream effect propagates through the KV
    cache from the prefill (canonical ActAdd; Turner et al. 2023).

    `v_layer` is one row of the saved (n_layers, d_model) tensor (i.e.,
    `v[v_var_idx, L]`). It is moved to the layer's device/dtype lazily on
    first call to avoid eager copies.
    """
    if alpha == 0:
        return None

    def hook(module, args, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output  # unexpected shape; skip safely
        seq_len = hidden.shape[1]
        if seq_len <= 1:
            return output  # decode step
        # Cast `v` lazily; first call sees the layer's actual dtype/device.
        v_cast = v_layer.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] - alpha * v_cast
        return output

    return hook


def _install_offset_hook(layers, layer_idx: int, v: torch.Tensor, alpha: float):
    """Install the offset hook on layers[layer_idx]; return a list of handles."""
    if alpha == 0:
        return []
    hook = _make_residual_offset_hook(v[layer_idx], alpha)
    if hook is None:
        return []
    return [layers[layer_idx].register_forward_hook(hook)]


def _sweep_cells(n_layers: int) -> list[dict]:
    """Phase 1 sweep grid. Returns list of cell dicts.

    Each cell: {layer, alpha, v_var_idx, label}. Includes a baseline cell
    (alpha=0) at the head; v_var_idx for the baseline is irrelevant and set
    to -1.
    """
    L_values = sorted(set([
        2,
        n_layers // 4,
        n_layers // 2 - 2,
        n_layers // 2,
        n_layers // 2 + 2,
        3 * n_layers // 4,
        n_layers - 2,
    ]))
    alphas = [1.0, 2.0, 4.0]
    v_var_names = ["v_wrong", "v_all"]  # idx 0, 1

    cells: list[dict] = [
        {"layer": -1, "alpha": 0.0, "v_var_idx": -1, "label": "baseline"}
    ]
    for L in L_values:
        for a in alphas:
            for vv_idx, vv_name in enumerate(v_var_names):
                cells.append({
                    "layer": L,
                    "alpha": a,
                    "v_var_idx": vv_idx,
                    "label": f"L{L:02d}_a{a}_{vv_name}",
                })
    return cells


def _load_v(model: str, tag: str | None = None) -> tuple[torch.Tensor, dict]:
    """Load the calibration tensor + sidecar from Phase 0 output.

    `tag=None` reads `calibration/v.pt` (default VQAv2 calibration);
    `tag='tally'` reads `calibration_tally/v.pt`, etc. — used for
    cross-direction transfer experiments.
    """
    sub = "calibration" if not tag else f"calibration_{tag}"
    cal_dir = PROJECT_ROOT / "outputs" / "e6_steering" / model / sub
    v = torch.load(cal_dir / "v.pt", weights_only=True)  # (2, n_layers, d_model)
    meta = json.loads((cal_dir / "v_meta.json").read_text())
    return v, meta


def _select_phase1_sids(args: argparse.Namespace, all_sids: list[str]) -> set[str]:
    """Pick the n=200 stratified set used by E1d / E4 (top-decile susceptible
    + bottom-decile resistant). Falls back to deterministic random subset
    if the susceptibility CSV is missing or the question_ids don't intersect.
    """
    susc_path = PROJECT_ROOT / args.susceptibility_csv
    n_target = args.top_decile_n + args.bottom_decile_n
    if susc_path.exists():
        target_qids = _select_susceptibility_strata(
            susc_path, args.top_decile_n, args.bottom_decile_n, args.seed
        )
        # sample_instance_id = "{question_id}_{image_id}_set{idx}"; split on "_".
        kept = {
            sid for sid in all_sids
            if int(sid.split("_")[0]) in target_qids
        }
        if len(kept) >= 0.5 * n_target:
            print(f"[setup] Phase 1 sids from susceptibility CSV: "
                  f"{len(kept)}/{n_target}")
            return kept
        print(f"[warn] susceptibility CSV intersects only {len(kept)} sids "
              f"of {n_target}; falling back to random {n_target}")

    import random as _random
    rng = _random.Random(args.seed)
    pool = list(all_sids)
    rng.shuffle(pool)
    kept = set(pool[:n_target])
    print(f"[setup] Phase 1 sids from random seed-{args.seed} subset: {len(kept)}")
    return kept


@torch.no_grad()
def _generate_with_cell(runner: Any, cond: dict, cell: dict,
                        layers, v: torch.Tensor, max_new_tokens: int) -> dict:
    """Run generate_number on (sample, condition) with the cell's hook
    installed; return the standard runner output dict."""
    handles: list = []
    if cell["alpha"] != 0:
        handles = _install_offset_hook(
            layers, cell["layer"], v[cell["v_var_idx"]], cell["alpha"]
        )
    try:
        out = runner.generate_number(
            question=cond["question"],
            images=cond["input_images"],
            max_new_tokens=max_new_tokens,
        )
    finally:
        for h in handles:
            h.remove()
    return out


def _exact_match(parsed, ground_truth) -> int:
    if parsed is None or ground_truth is None:
        return 0
    try:
        return int(int(parsed) == int(str(ground_truth).strip()))
    except (ValueError, TypeError):
        return 0


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


def _build_runner_and_layers(args, config):
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
    print(f"[setup] LLM layers detected: {len(layers)}")
    return runner, layers


def _phase_calibrate(args, config) -> None:
    if args.predictions_path:
        _phase_calibrate_from_predictions(args, config)
        return

    e5c_run = (PROJECT_ROOT / args.e5c_run_dir).resolve()
    wrong_sids = _read_wrong_base_sids(e5c_run)
    print(f"[setup] wrong-base sids from {e5c_run.name}: {len(wrong_sids)}")

    enriched = _load_calibration_samples(args, config)
    print(f"[setup] enriched samples (calibration sids): {len(enriched)}")

    runner, layers = _build_runner_and_layers(args, config)
    n_layers = len(layers)

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


def _phase_calibrate_from_predictions(args, config) -> None:
    """Reverse-direction calibration: extract `v_wrong`/`v_all` from any E5*
    dataset (TallyQA, ChartQA, ...) by reading its predictions.jsonl, filtering
    to wrong-base S1 pairs, reconstructing images from `input_image_paths`,
    and capturing residuals at the last input position.

    Produces `outputs/e6_steering/<model>/calibration_<dataset_tag>/v.pt`,
    which `_load_v(model, tag=...)` callers can use for cross-direction
    transfer experiments.
    """
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    print(f"[setup] reverse-calibration source: {pred_path}")

    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        try:
            dataset_tag = pred_path.parents[2].name
        except IndexError:
            dataset_tag = "alt"
    print(f"[setup] dataset_tag = {dataset_tag}")

    # Group records by sid → cond
    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    target_conds_calib = (
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
    )
    eligible_sids: list[str] = []
    for sid, d in by_sid_cond.items():
        if not all(c in d for c in target_conds_calib):
            continue
        eligible_sids.append(sid)
    print(f"[setup] sids with (b, a-S1, m-S1): {len(eligible_sids)}")
    wrong_sids = {sid for sid in eligible_sids
                  if by_sid_cond[sid]["target_only"].get("exact_match") == 0}
    print(f"[setup] wrong-base subset: {len(wrong_sids)}")
    if args.max_samples:
        eligible_sids = eligible_sids[: args.max_samples]
        print(f"[setup] capped to {len(eligible_sids)} sids")

    runner, layers = _build_runner_and_layers(args, config)
    n_layers = len(layers)

    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / f"calibration_{dataset_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] writing to {out_dir}")

    from PIL import Image as _Image

    diffs_wrong: list[torch.Tensor] = []
    diffs_all: list[torch.Tensor] = []
    n_pairs, n_skipped = 0, 0
    t0 = time.time()

    for i, sid in enumerate(eligible_sids):
        a_rec = by_sid_cond[sid].get("target_plus_irrelevant_number_S1")
        m_rec = by_sid_cond[sid].get("target_plus_irrelevant_number_masked_S1")
        if a_rec is None or m_rec is None:
            n_skipped += 1
            continue

        a_res, m_res = None, None
        for rec, slot in ((a_rec, "a"), (m_rec, "m")):
            try:
                paths = rec.get("input_image_paths") or []
                images = [_Image.open(p).convert("RGB") for p in paths]
                cond_dict = {
                    "question": rec["question"],
                    "input_images": images,
                }
                res = _capture_last_token_residuals(runner, cond_dict)
                if slot == "a":
                    a_res = res
                else:
                    m_res = res
            except Exception as exc:
                print(f"  [error] sid={sid} slot={slot}: {exc}")

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
            eta = elapsed * (len(eligible_sids) - i - 1) / max(n_pairs, 1)
            print(f"  [progress] {n_pairs}/{len(eligible_sids)} pairs "
                  f"(wrong-base subset: {len(diffs_wrong)}); "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    if not diffs_all:
        raise RuntimeError("no calibration pairs collected — check predictions path")
    if not diffs_wrong:
        print("[warn] no wrong-base pairs found; v_wrong falls back to zeros")

    v_all = torch.stack(diffs_all).mean(dim=0)
    v_wrong = (torch.stack(diffs_wrong).mean(dim=0)
               if diffs_wrong else torch.zeros_like(v_all))

    v = torch.stack([v_wrong, v_all], dim=0)
    torch.save(v, out_dir / "v.pt")

    sidecar = {
        "model": args.model,
        "hf_model": args.hf_model,
        "dataset_tag": dataset_tag,
        "n_wrong": len(diffs_wrong),
        "n_all": len(diffs_all),
        "n_skipped": n_skipped,
        "n_layers": n_layers,
        "d_model": int(v.shape[-1]),
        "source_predictions": str(args.predictions_path),
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
    print(f"[done] reverse-calibrate: n_wrong={len(diffs_wrong)} "
          f"n_all={len(diffs_all)} n_layers={n_layers} d={v.shape[-1]} "
          f"wall={elapsed:.1f}s; saved {out_dir/'v.pt'}")


def _phase_smoke(args, config) -> None:
    """Phase 0.5 — wiring smoke for the steering hook."""
    v, meta = _load_v(args.model)
    n_layers = meta["n_layers"]
    print(f"[setup] loaded v.pt: shape={tuple(v.shape)} n_wrong={meta['n_wrong']}")

    e5c_run = (PROJECT_ROOT / args.e5c_run_dir).resolve()
    wrong_sids = _read_wrong_base_sids(e5c_run)
    enriched = _load_calibration_samples(args, config)
    enriched_wrong = [s for s in enriched if str(s["sample_instance_id"]) in wrong_sids]
    n_smoke = args.max_samples or 10
    enriched_wrong = enriched_wrong[:n_smoke]
    print(f"[setup] smoke set: first {len(enriched_wrong)} wrong-base sids")

    runner, layers = _build_runner_and_layers(args, config)

    cell = {
        "layer": n_layers // 2,
        "alpha": 2.0,
        "v_var_idx": 0,  # v_wrong
        "label": f"L{n_layers // 2:02d}_a2.0_v_wrong",
    }
    print(f"[setup] smoke cell: {cell}")

    out_dir = PROJECT_ROOT / "outputs" / "e6_steering" / args.model / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_changed = 0
    n_anchor_adopted_baseline = 0
    n_anchor_adopted_steered = 0
    t0 = time.time()
    for sample in enriched_wrong:
        sid = str(sample["sample_instance_id"])
        for cond in build_conditions(sample):
            if cond["condition"] != "target_plus_irrelevant_number_S1":
                continue
            base_out = _generate_with_cell(
                runner, cond,
                {"layer": -1, "alpha": 0.0, "v_var_idx": -1, "label": "baseline"},
                layers, v, args.max_new_tokens,
            )
            steered_out = _generate_with_cell(
                runner, cond, cell, layers, v, args.max_new_tokens,
            )
            base_pn = base_out["parsed_number"]
            steered_pn = steered_out["parsed_number"]
            anchor_v = cond.get("anchor_value_for_metrics") or cond.get("anchor_value")
            try:
                anchor_int = int(anchor_v) if anchor_v is not None else None
            except (ValueError, TypeError):
                anchor_int = None
            try:
                base_int = int(base_pn) if base_pn is not None else None
            except (ValueError, TypeError):
                base_int = None
            try:
                steered_int = int(steered_pn) if steered_pn is not None else None
            except (ValueError, TypeError):
                steered_int = None

            adopted_base = (base_int is not None and anchor_int is not None
                            and base_int == anchor_int)
            adopted_steer = (steered_int is not None and anchor_int is not None
                             and steered_int == anchor_int)
            n_anchor_adopted_baseline += int(adopted_base)
            n_anchor_adopted_steered += int(adopted_steer)
            changed = (base_pn != steered_pn) or (base_out["raw_text"] != steered_out["raw_text"])
            n_changed += int(changed)

            results.append({
                "sample_instance_id": sid,
                "condition": cond["condition"],
                "anchor_value": anchor_int,
                "ground_truth": cond["ground_truth"],
                "baseline_pn": base_pn,
                "steered_pn": steered_pn,
                "baseline_text": base_out["raw_text"],
                "steered_text": steered_out["raw_text"],
                "changed": changed,
            })
            break  # only S1 anchor — one cond per sid

    summary = {
        "model": args.model,
        "cell": cell,
        "n_pairs": len(results),
        "n_changed": n_changed,
        "n_anchor_adopted_baseline": n_anchor_adopted_baseline,
        "n_anchor_adopted_steered": n_anchor_adopted_steered,
        "wall_seconds": time.time() - t0,
        "details": results,
    }
    (out_dir / "smoke_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[smoke] {n_changed}/{len(results)} predictions changed under steering "
          f"at L={cell['layer']} α={cell['alpha']} v=v_wrong")
    print(f"[smoke] anchor adoption: baseline {n_anchor_adopted_baseline} → "
          f"steered {n_anchor_adopted_steered}  (lower under steering = good)")
    print(f"[smoke] wall: {summary['wall_seconds']:.1f}s; saved {out_dir/'smoke_results.json'}")
    if n_changed == 0:
        raise RuntimeError(
            "smoke FAIL: 0 predictions changed under steering — hook not "
            "wiring correctly. Check residual position / layer indexing / "
            "device-dtype casting."
        )


def _sweep_output_path(model: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / "sweep_n200" / "predictions.jsonl"


def _load_completed_keys(path: Path) -> set[tuple[str, str, int, float, int]]:
    """(sid, condition, L, alpha, v_var_idx) tuples already written."""
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
                    int(r["cell_layer"]),
                    float(r["cell_alpha"]),
                    int(r["cell_v_var_idx"]),
                ))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


SWEEP_CONDITIONS = (
    "target_only",
    "target_plus_irrelevant_number_S1",
    "target_plus_irrelevant_number_masked_S1",
    "target_plus_irrelevant_neutral",
)


# ─── Method 1: subspace projection (CIPHER / VCE / RepE style) ─────────────


def _make_subspace_projection_hook(V_K: torch.Tensor, alpha: float):
    """At prefill last token: h ← h − α · V_K^T · V_K · h.

    V_K is (K, d_model); rows are top-K right singular vectors of the pooled
    D matrix. Decode steps (seq_len == 1) are skipped so the effect propagates
    through the KV cache (same convention as the ActAdd offset hook).
    """
    if alpha == 0:
        return None

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        if hidden.shape[1] <= 1:
            return output  # decode step — no-op
        V_cast = V_K.to(device=hidden.device, dtype=hidden.dtype)  # (K, d)
        last = hidden[:, -1, :]  # (batch, d)
        proj = (last @ V_cast.T) @ V_cast  # (batch, d)
        hidden[:, -1, :] = last - alpha * proj
        return output

    return hook


def _install_projection_hook(layers, layer_idx: int, V_K: torch.Tensor, alpha: float):
    """Install subspace projection hook on layers[layer_idx]; return handle list."""
    if alpha == 0:
        return []
    hook = _make_subspace_projection_hook(V_K, alpha)
    if hook is None:
        return []
    return [layers[layer_idx].register_forward_hook(hook)]


def _subspace_sweep_cells(layers_arg: str | None = None,
                           ks_arg: str | None = None,
                           alphas_arg: str | None = None) -> list[dict]:
    """Method 1 grid (default 5×4×4 = 80 steered + 1 baseline = 81 total)."""
    if layers_arg:
        L_values = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    else:
        L_values = [16, 22, 28, 30, 31]
    if ks_arg:
        K_values = [int(x.strip()) for x in ks_arg.split(",") if x.strip()]
    else:
        K_values = [2, 4, 8, 16]
    if alphas_arg:
        alphas = [float(x.strip()) for x in alphas_arg.split(",") if x.strip()]
    else:
        alphas = [0.5, 1.0, 2.0, 4.0]
    cells: list[dict] = [{"layer": -1, "alpha": 0.0, "K": 0, "label": "baseline"}]
    for L in L_values:
        for K in K_values:
            for a in alphas:
                cells.append({"layer": L, "alpha": a, "K": K,
                               "label": f"L{L:02d}_K{K:02d}_a{a}"})
    return cells


def _load_subspace(subspace_path: "str | Path") -> torch.Tensor:
    """Load precomputed subspace tensor. Shape: (n_layers, K_max, d_model)."""
    p = Path(subspace_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return torch.load(p, weights_only=True)


def _load_completed_keys_subspace(path: Path) -> set:
    """(sid, condition, layer, K, alpha) tuples already in sweep-subspace output."""
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
                    int(r["cell_layer"]),
                    int(r["subspace_K"]),
                    float(r["cell_alpha"]),
                ))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def _phase_sweep(args, config) -> None:
    v, meta = _load_v(args.model)
    n_layers = meta["n_layers"]
    cells = _sweep_cells(n_layers)
    print(f"[setup] sweep grid: {len(cells)} cells (1 baseline + {len(cells) - 1} steered)")

    e5c_run = (PROJECT_ROOT / args.e5c_run_dir).resolve()
    wrong_sids = _read_wrong_base_sids(e5c_run)
    enriched = _load_calibration_samples(args, config)
    all_sids = [str(s["sample_instance_id"]) for s in enriched]
    chosen_sids = _select_phase1_sids(args, all_sids)
    enriched = [s for s in enriched if str(s["sample_instance_id"]) in chosen_sids]
    if args.max_samples:
        enriched = enriched[: args.max_samples]
    print(f"[setup] Phase 1 calibration sample-instances: {len(enriched)}")

    runner, layers = _build_runner_and_layers(args, config)
    out_path = _sweep_output_path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(out_path)
    print(f"[setup] writing to {out_path}; resuming_from={len(completed)}")

    total_target = len(enriched) * len(SWEEP_CONDITIONS) * len(cells)
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        # Outer loop on cells: install hook once per cell, run all (sid, cond)
        # under it, remove.
        for cell in cells:
            handles = []
            if cell["alpha"] != 0:
                handles = _install_offset_hook(
                    layers, cell["layer"], v[cell["v_var_idx"]], cell["alpha"]
                )
            try:
                for sample in enriched:
                    sid = str(sample["sample_instance_id"])
                    for cond in build_conditions(sample):
                        if cond["condition"] not in SWEEP_CONDITIONS:
                            continue
                        key = (sid, cond["condition"], int(cell["layer"]),
                               float(cell["alpha"]), int(cell["v_var_idx"]))
                        if key in completed:
                            n_skipped += 1
                            continue
                        try:
                            out = runner.generate_number(
                                question=cond["question"],
                                images=cond["input_images"],
                                max_new_tokens=args.max_new_tokens,
                            )
                            err = None
                        except Exception as exc:
                            out = {"raw_text": None, "parsed_number": None}
                            err = str(exc)

                        anchor_v = cond.get("anchor_value_for_metrics") or cond.get("anchor_value")
                        try:
                            anchor_int = int(anchor_v) if anchor_v is not None else None
                        except (ValueError, TypeError):
                            anchor_int = None
                        record = {
                            "model": args.model,
                            "sample_instance_id": sid,
                            "question_id": cond["question_id"],
                            "condition": cond["condition"],
                            "ground_truth": cond["ground_truth"],
                            "anchor_value": anchor_int,
                            "is_wrong_base": sid in wrong_sids,
                            "cell_label": cell["label"],
                            "cell_layer": cell["layer"],
                            "cell_alpha": cell["alpha"],
                            "cell_v_var_idx": cell["v_var_idx"],
                            "raw_text": out.get("raw_text"),
                            "parsed_number": out.get("parsed_number"),
                            "exact_match": _exact_match(out.get("parsed_number"),
                                                        cond["ground_truth"]),
                            "error": err,
                        }
                        fh.write(json.dumps(record) + "\n")
                        fh.flush()
                        n_done += 1
                        if n_done % 200 == 0 or n_done == 1:
                            elapsed = time.time() - t0
                            done_total = n_done + n_skipped
                            rate = n_done / max(elapsed, 1)
                            remaining = total_target - done_total
                            eta = remaining / max(rate, 1e-6)
                            print(f"  [progress] cell={cell['label']:30s} "
                                  f"done={n_done} skipped={n_skipped} "
                                  f"rate={rate:.1f}/s eta={eta/60:.1f}min")
            finally:
                for h in handles:
                    h.remove()

    elapsed = time.time() - t0
    print(f"[done] sweep written: {n_done} new + {n_skipped} resumed = "
          f"{n_done + n_skipped}/{total_target} target")
    print(f"[done] wall={elapsed:.1f}s ({elapsed/60:.1f}min); saved {out_path}")


def _parse_cell_spec(spec: str) -> dict:
    """Parse 'L:alpha:v_var_idx' → cell dict."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"cell spec must be 'L:alpha:v_var_idx', got {spec!r}")
    L = int(parts[0])
    alpha = float(parts[1])
    vv = int(parts[2])
    v_var_name = "v_wrong" if vv == 0 else ("v_all" if vv == 1 else f"v{vv}")
    return {
        "layer": L, "alpha": alpha, "v_var_idx": vv,
        "label": f"L{L:02d}_a{alpha}_{v_var_name}",
    }


def _phase_tiebreaker(args, config) -> None:
    """Cross-dataset tiebreaker. Read existing E5* predictions, reconstruct
    (target, irrelevant) image pairs from input_image_paths, run baseline +
    candidate cells, summarize df / em per cell."""
    if not args.predictions_path:
        raise ValueError("--phase tiebreaker requires --predictions-path")

    v, meta = _load_v(args.model, tag=args.calibration_tag)
    n_layers = meta["n_layers"]
    print(f"[setup] loaded v from calibration"
          f"{'_' + args.calibration_tag if args.calibration_tag else ''}/ "
          f"(n_wrong={meta.get('n_wrong')}, source="
          f"{meta.get('dataset_tag') or 'vqa'})")

    # Cells: baseline + parsed candidates
    cells = [{"layer": -1, "alpha": 0.0, "v_var_idx": -1, "label": "baseline"}]
    for spec in args.cells.split(","):
        cells.append(_parse_cell_spec(spec.strip()))
    print(f"[setup] cells: {[c['label'] for c in cells]}")

    pred_path = Path(args.predictions_path).resolve()
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    print(f"[setup] reading {pred_path}")

    # Group: sid → cond → record
    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    target_conds = (
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
        "target_plus_irrelevant_neutral",
    )

    # Filter to wrong-base sids that have all 4 target conditions
    eligible_sids: list[str] = []
    for sid, d in by_sid_cond.items():
        if not all(c in d for c in target_conds):
            continue
        if d["target_only"].get("exact_match") != 0:
            continue  # not wrong-base
        eligible_sids.append(sid)
    print(f"[setup] eligible (wrong-base, all 4 conds): {len(eligible_sids)}")
    if args.max_samples:
        eligible_sids = eligible_sids[: args.max_samples]
        print(f"[setup] capped to {len(eligible_sids)} sids")

    runner, layers = _build_runner_and_layers(args, config)

    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        # outputs/<exp>/<model>/<run>/predictions.jsonl  →  <exp>
        try:
            dataset_tag = pred_path.parents[2].name
        except IndexError:
            dataset_tag = "tiebreaker"
    sub_name = f"tiebreaker_{dataset_tag}"
    if args.calibration_tag:
        sub_name = f"{sub_name}__from_{args.calibration_tag}"
    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / sub_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    completed = _load_completed_keys(out_path)
    print(f"[setup] writing to {out_path}; resuming_from={len(completed)}")

    from PIL import Image

    total_target = len(eligible_sids) * len(target_conds) * len(cells)
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for cell in cells:
            handles = []
            if cell["alpha"] != 0:
                handles = _install_offset_hook(
                    layers, cell["layer"], v[cell["v_var_idx"]], cell["alpha"]
                )
            try:
                for sid in eligible_sids:
                    for cond_name in target_conds:
                        r0 = by_sid_cond[sid].get(cond_name)
                        if r0 is None:
                            continue
                        key = (sid, cond_name, int(cell["layer"]),
                               float(cell["alpha"]), int(cell["v_var_idx"]))
                        if key in completed:
                            n_skipped += 1
                            continue

                        # Reconstruct images from recorded paths
                        try:
                            img_paths = r0.get("input_image_paths") or []
                            images = [Image.open(p).convert("RGB") for p in img_paths]
                            out = runner.generate_number(
                                question=r0["question"],
                                images=images,
                                max_new_tokens=args.max_new_tokens,
                            )
                            err = None
                        except Exception as exc:
                            out = {"raw_text": None, "parsed_number": None}
                            err = str(exc)

                        anchor_int = None
                        if r0.get("anchor_value") is not None:
                            try:
                                anchor_int = int(str(r0["anchor_value"]).strip())
                            except (ValueError, TypeError):
                                pass
                        rec = {
                            "model": args.model,
                            "dataset_tag": dataset_tag,
                            "sample_instance_id": sid,
                            "question_id": r0.get("question_id"),
                            "condition": cond_name,
                            "ground_truth": r0.get("ground_truth"),
                            "anchor_value": anchor_int,
                            "is_wrong_base": True,
                            "cell_label": cell["label"],
                            "cell_layer": cell["layer"],
                            "cell_alpha": cell["alpha"],
                            "cell_v_var_idx": cell["v_var_idx"],
                            "raw_text": out.get("raw_text"),
                            "parsed_number": out.get("parsed_number"),
                            "exact_match": _exact_match(
                                out.get("parsed_number"), r0.get("ground_truth")
                            ),
                            "error": err,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_done += 1
                        if n_done % 200 == 0 or n_done == 1:
                            elapsed = time.time() - t0
                            rate = n_done / max(elapsed, 1)
                            remaining = total_target - (n_done + n_skipped)
                            eta = remaining / max(rate, 1e-6)
                            print(f"  [progress] cell={cell['label']:25s} "
                                  f"done={n_done} skipped={n_skipped} "
                                  f"rate={rate:.1f}/s eta={eta/60:.1f}min")
            finally:
                for h in handles:
                    h.remove()

    elapsed = time.time() - t0
    print(f"[done] tiebreaker: {n_done} new + {n_skipped} resumed = "
          f"{n_done + n_skipped}/{total_target} target  wall={elapsed/60:.1f}min")
    print(f"[done] saved {out_path}")


def _phase_calibrate_subspace(args, config) -> None:
    """Phase calibrate-subspace: save per-pair D matrices (n_pairs, n_layers, d_model)
    alongside v.pt for subsequent SVD in e6_compute_subspace.py."""
    if args.predictions_path:
        _calibrate_subspace_from_predictions(args, config)
    else:
        _calibrate_subspace_from_e5c(args, config)


def _calibrate_subspace_from_e5c(args, config) -> None:
    """Collect per-pair diff tensors from VQAv2 (E5c run) for SVD."""
    e5c_run = (PROJECT_ROOT / args.e5c_run_dir).resolve()
    wrong_sids = _read_wrong_base_sids(e5c_run)
    print(f"[setup] wrong-base sids from {e5c_run.name}: {len(wrong_sids)}")
    enriched = _load_calibration_samples(args, config)
    max_pairs = args.max_calibrate_pairs

    # Prioritize wrong-base samples for maximum D_wrong coverage
    enriched = (
        [s for s in enriched if str(s["sample_instance_id"]) in wrong_sids]
        + [s for s in enriched if str(s["sample_instance_id"]) not in wrong_sids]
    )

    runner, layers = _build_runner_and_layers(args, config)
    n_layers = len(layers)
    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / "calibration_vqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] writing to {out_dir}  (max_pairs={max_pairs})")

    diffs_wrong: list[torch.Tensor] = []
    diffs_all: list[torch.Tensor] = []
    n_skipped = 0
    t0 = time.time()
    for i, sample in enumerate(enriched):
        if len(diffs_all) >= max_pairs:
            break
        sid = str(sample["sample_instance_id"])
        a_res, m_res = None, None
        for cond in build_conditions(sample):
            if cond["condition"] not in CALIB_CONDITIONS:
                continue
            try:
                res = _capture_last_token_residuals(runner, cond)
            except Exception as exc:
                print(f"  [error] sid={sid}: {exc}")
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
        n_pairs_so_far = len(diffs_all)
        if n_pairs_so_far % 20 == 0 or n_pairs_so_far == 1:
            elapsed = time.time() - t0
            print(f"  [progress] {n_pairs_so_far}/{max_pairs} pairs "
                  f"(wrong-base: {len(diffs_wrong)}); elapsed={elapsed:.1f}s")

    _save_D_and_v(out_dir, diffs_wrong, diffs_all, n_layers, n_skipped, args,
                  "vqa", time.time() - t0)


def _calibrate_subspace_from_predictions(args, config) -> None:
    """Collect per-pair diff tensors from any E5e dataset predictions.jsonl."""
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        try:
            dataset_tag = pred_path.parents[2].name
        except IndexError:
            dataset_tag = "alt"
    max_pairs = args.max_calibrate_pairs
    print(f"[setup] {dataset_tag}: reading {pred_path}  (max_pairs={max_pairs})")

    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    eligible_sids = [
        sid for sid, d in by_sid_cond.items()
        if all(c in d for c in (
            "target_only",
            "target_plus_irrelevant_number_S1",
            "target_plus_irrelevant_number_masked_S1",
        ))
    ]
    wrong_sids = {sid for sid in eligible_sids
                  if by_sid_cond[sid]["target_only"].get("exact_match") == 0}
    print(f"[setup] {len(eligible_sids)} eligible sids, {len(wrong_sids)} wrong-base")

    # Prioritize wrong-base sids so D_wrong gets max coverage before hitting max_pairs
    ordered_sids = ([s for s in eligible_sids if s in wrong_sids]
                    + [s for s in eligible_sids if s not in wrong_sids])

    runner, layers = _build_runner_and_layers(args, config)
    n_layers = len(layers)
    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / f"calibration_{dataset_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] writing to {out_dir}")

    from PIL import Image as _Image
    diffs_wrong: list[torch.Tensor] = []
    diffs_all: list[torch.Tensor] = []
    n_skipped = 0
    t0 = time.time()

    for sid in ordered_sids:
        if len(diffs_all) >= max_pairs:
            break
        a_rec = by_sid_cond[sid].get("target_plus_irrelevant_number_S1")
        m_rec = by_sid_cond[sid].get("target_plus_irrelevant_number_masked_S1")
        if a_rec is None or m_rec is None:
            n_skipped += 1
            continue
        a_res, m_res = None, None
        for rec, slot in ((a_rec, "a"), (m_rec, "m")):
            try:
                images = [_Image.open(p).convert("RGB")
                          for p in (rec.get("input_image_paths") or [])]
                res = _capture_last_token_residuals(
                    runner, {"question": rec["question"], "input_images": images})
                if slot == "a":
                    a_res = res
                else:
                    m_res = res
            except Exception as exc:
                print(f"  [error] sid={sid} slot={slot}: {exc}")
        if a_res is None or m_res is None:
            n_skipped += 1
            continue
        diff = a_res - m_res
        diffs_all.append(diff)
        if sid in wrong_sids:
            diffs_wrong.append(diff)
        n_pairs_so_far = len(diffs_all)
        if n_pairs_so_far % 20 == 0 or n_pairs_so_far == 1:
            elapsed = time.time() - t0
            print(f"  [progress] {n_pairs_so_far}/{max_pairs} pairs "
                  f"(wrong-base: {len(diffs_wrong)}); elapsed={elapsed:.1f}s")

    _save_D_and_v(out_dir, diffs_wrong, diffs_all, n_layers, n_skipped, args,
                  dataset_tag, time.time() - t0)


def _save_D_and_v(out_dir: Path, diffs_wrong: list, diffs_all: list,
                  n_layers: int, n_skipped: int, args,
                  dataset_tag: str, wall_seconds: float) -> None:
    """Save D_wrong.pt, D_all.pt, v.pt, and v_meta.json."""
    if not diffs_all:
        raise RuntimeError(f"no calibration pairs collected for {dataset_tag!r}")
    D_all = torch.stack(diffs_all)   # (n_all, n_layers, d_model)
    D_wrong = (torch.stack(diffs_wrong) if diffs_wrong
               else torch.zeros(0, n_layers, D_all.shape[-1], dtype=D_all.dtype))
    torch.save(D_all, out_dir / "D_all.pt")
    torch.save(D_wrong, out_dir / "D_wrong.pt")
    v_all = D_all.mean(0)
    v_wrong = D_wrong.mean(0) if len(diffs_wrong) else torch.zeros_like(v_all)
    torch.save(torch.stack([v_wrong, v_all]), out_dir / "v.pt")
    sidecar = {
        "model": args.model, "hf_model": args.hf_model,
        "dataset_tag": dataset_tag,
        "n_wrong": len(diffs_wrong), "n_all": len(diffs_all),
        "n_skipped": n_skipped, "n_layers": n_layers,
        "d_model": int(D_all.shape[-1]),
        "D_wrong_shape": list(D_wrong.shape),
        "D_all_shape": list(D_all.shape),
        "v_index_0": "v_wrong", "v_index_1": "v_all",
        "wall_seconds": wall_seconds,
    }
    (out_dir / "v_meta.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[done] {dataset_tag}: D_wrong={tuple(D_wrong.shape)} "
          f"D_all={tuple(D_all.shape)} wall={wall_seconds:.1f}s; saved {out_dir}")


def _phase_smoke_subspace(args, config) -> None:
    """10-pair wiring smoke for subspace projection hook (Method 1)."""
    if not args.subspace_path:
        raise ValueError("--phase smoke-subspace requires --subspace-path")
    if not args.predictions_path:
        raise ValueError("--phase smoke-subspace requires --predictions-path")
    V_all = _load_subspace(args.subspace_path)  # (n_layers, K_max, d_model)
    # Use aggressive parameters: large alpha to ensure discrete output can change
    L_smoke, K_smoke, alpha_smoke = 30, 16, 4.0
    V_K_smoke = V_all[L_smoke, :K_smoke, :]
    print(f"[setup] smoke-subspace: L={L_smoke} K={K_smoke} alpha={alpha_smoke}")

    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    eligible_sids = [
        sid for sid, d in by_sid_cond.items()
        if ("target_only" in d
            and d["target_only"].get("exact_match") == 0
            and "target_plus_irrelevant_number_S1" in d)
    ]
    n_smoke = args.max_samples or 10
    eligible_sids = eligible_sids[:n_smoke]
    print(f"[setup] smoke set: {len(eligible_sids)} wrong-base sids")

    runner, layers = _build_runner_and_layers(args, config)
    dataset_tag = args.dataset_tag or "ds"
    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / f"smoke_subspace_{dataset_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image as _Image
    results = []
    n_changed = 0
    t0 = time.time()
    for sid in eligible_sids:
        r0 = by_sid_cond[sid].get("target_plus_irrelevant_number_S1")
        if r0 is None:
            continue
        images = [_Image.open(p).convert("RGB")
                  for p in (r0.get("input_image_paths") or [])]
        base_out = runner.generate_number(question=r0["question"], images=images,
                                          max_new_tokens=args.max_new_tokens)
        handles = _install_projection_hook(layers, L_smoke, V_K_smoke, alpha_smoke)
        try:
            steer_out = runner.generate_number(question=r0["question"], images=images,
                                               max_new_tokens=args.max_new_tokens)
        finally:
            for h in handles:
                h.remove()
        changed = base_out["raw_text"] != steer_out["raw_text"]
        n_changed += int(changed)
        results.append({"sid": sid, "baseline": base_out["raw_text"],
                        "steered": steer_out["raw_text"], "changed": changed})

    summary = {"model": args.model, "L": L_smoke, "K": K_smoke, "alpha": alpha_smoke,
               "n_pairs": len(results), "n_changed": n_changed,
               "wall_seconds": time.time() - t0, "details": results}
    (out_dir / "smoke_results.json").write_text(json.dumps(summary, indent=2))
    print(f"[smoke-subspace] {n_changed}/{len(results)} predictions changed")
    print(f"[smoke-subspace] saved {out_dir / 'smoke_results.json'}")
    if n_changed == 0:
        raise RuntimeError(
            "smoke-subspace FAIL: 0 predictions changed — projection hook not wiring"
        )


def _phase_sweep_subspace(args, config) -> None:
    """Method 1 sweep: 61 cells × sids × 4 conditions with subspace projection."""
    if not args.subspace_path:
        raise ValueError("--phase sweep-subspace requires --subspace-path")
    if not args.predictions_path:
        raise ValueError("--phase sweep-subspace requires --predictions-path")
    V_all = _load_subspace(args.subspace_path)  # (n_layers, K_max, d_model)
    K_max = V_all.shape[1]
    cells = _subspace_sweep_cells(args.sweep_layers, args.sweep_ks,
                                    args.sweep_alphas)
    print(f"[setup] sweep-subspace: {len(cells)} cells, K_max={K_max}")

    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        try:
            dataset_tag = pred_path.parents[2].name
        except IndexError:
            dataset_tag = "dataset"
    scope = args.subspace_scope
    print(f"[setup] dataset_tag={dataset_tag} scope={scope}")

    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[r["sample_instance_id"]][r["condition"]] = r

    target_conds = (
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
        "target_plus_irrelevant_neutral",
    )
    eligible_sids = [
        sid for sid, d in by_sid_cond.items()
        if (all(c in d for c in target_conds)
            and d["target_only"].get("exact_match") == 0)
    ]
    print(f"[setup] eligible wrong-base sids (all 4 conds): {len(eligible_sids)}")
    if args.max_samples:
        eligible_sids = eligible_sids[:args.max_samples]
        print(f"[setup] capped to {len(eligible_sids)}")

    out_dir = (PROJECT_ROOT / "outputs" / "e6_steering" / args.model
               / f"sweep_subspace_{dataset_tag}_{scope}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    completed = _load_completed_keys_subspace(out_path)
    print(f"[setup] writing to {out_path}; resumed={len(completed)}")

    runner, layers = _build_runner_and_layers(args, config)

    from PIL import Image as _Image
    total_target = len(eligible_sids) * len(target_conds) * len(cells)
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for cell in cells:
            L = cell["layer"]
            K = cell["K"]
            alpha = cell["alpha"]
            handles = []
            if alpha != 0 and K > 0 and K <= K_max:
                V_K = V_all[L, :K, :]  # (K, d)
                handles = _install_projection_hook(layers, L, V_K, alpha)
            try:
                for sid in eligible_sids:
                    for cond_name in target_conds:
                        r0 = by_sid_cond[sid].get(cond_name)
                        if r0 is None:
                            continue
                        key = (sid, cond_name, int(L), int(K), float(alpha))
                        if key in completed:
                            n_skipped += 1
                            continue
                        try:
                            img_paths = r0.get("input_image_paths") or []
                            images = [_Image.open(p).convert("RGB") for p in img_paths]
                            out = runner.generate_number(
                                question=r0["question"], images=images,
                                max_new_tokens=args.max_new_tokens)
                            err = None
                        except Exception as exc:
                            out = {"raw_text": None, "parsed_number": None}
                            err = str(exc)
                        anchor_int = None
                        if r0.get("anchor_value") is not None:
                            try:
                                anchor_int = int(str(r0["anchor_value"]).strip())
                            except (ValueError, TypeError):
                                pass
                        rec = {
                            "model": args.model,
                            "dataset_tag": dataset_tag,
                            "subspace_scope": scope,
                            "sample_instance_id": sid,
                            "question_id": r0.get("question_id"),
                            "condition": cond_name,
                            "ground_truth": r0.get("ground_truth"),
                            "anchor_value": anchor_int,
                            "is_wrong_base": True,
                            "cell_label": cell["label"],
                            "cell_layer": L,
                            "cell_alpha": alpha,
                            "subspace_K": K,
                            "raw_text": out.get("raw_text"),
                            "parsed_number": out.get("parsed_number"),
                            "exact_match": _exact_match(
                                out.get("parsed_number"), r0.get("ground_truth")),
                            "error": err,
                        }
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_done += 1
                        if n_done % 200 == 0 or n_done == 1:
                            elapsed = time.time() - t0
                            rate = n_done / max(elapsed, 1)
                            remaining = total_target - (n_done + n_skipped)
                            eta = remaining / max(rate, 1e-6)
                            print(f"  [progress] cell={cell['label']:28s} "
                                  f"done={n_done} skipped={n_skipped} "
                                  f"rate={rate:.1f}/s eta={eta/60:.1f}min")
            finally:
                for h in handles:
                    h.remove()

    elapsed = time.time() - t0
    print(f"[done] sweep-subspace {dataset_tag}: {n_done} new + {n_skipped} resumed "
          f"= {n_done + n_skipped}/{total_target} wall={elapsed/60:.1f}min")
    print(f"[done] saved {out_path}")


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())

    if args.phase == "calibrate":
        _phase_calibrate(args, config)
    elif args.phase == "smoke":
        _phase_smoke(args, config)
    elif args.phase == "sweep":
        _phase_sweep(args, config)
    elif args.phase == "tiebreaker":
        _phase_tiebreaker(args, config)
    elif args.phase == "calibrate-subspace":
        _phase_calibrate_subspace(args, config)
    elif args.phase == "smoke-subspace":
        _phase_smoke_subspace(args, config)
    elif args.phase == "sweep-subspace":
        _phase_sweep_subspace(args, config)
    else:
        raise ValueError(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
