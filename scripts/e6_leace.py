"""E6 Method 4c — LEACE closed-form linear erasure (arXiv:2306.03819).

Pipeline phases
---------------
  calibrate-leace : Pool Q_wrong.pt + D_wrong.pt from all 3 calibration
                    datasets, construct (no-anchor, with-anchor) pairs per
                    layer, fit LeaceEraser per layer, save erasers.pt.
                    CPU-only; runs in ~1-2 min.

  smoke-leace     : 5-sample wiring check.  Verifies the erasure hook
                    changes at least one digit output.

  sweep-leace     : Full (L × alpha) grid sweep on n=100 wrong-base sids
                    per dataset (TallyQA/ChartQA).

Methodology (LEACE, Belrose et al. 2023):
  Fit per-layer LeaceEraser on approximate (h_b, h_a) pairs:
    X_neg[i, L] = Q_wrong[i, L]                # b-arm repr (no anchor)
    X_pos[i, L] = Q_wrong[i, L] + D_wrong[i, L]  # approx a-arm repr
  Eraser(h) removes the minimal-norm subspace predictive of anchor presence.
  At inference: h ← h − alpha × (h − Eraser(h))
    alpha = 0: no correction; alpha = 1: full LEACE erasure.
    alpha > 1: over-erasure (amplified direction subtraction).

Output:
  outputs/e6_steering/<model>/leace_erasers/
    erasers.pt          # list of LeaceEraser objects, one per layer
    eraser_meta.json

  outputs/e6_steering/<model>/sweep_leace_<dataset>_pooled/
    predictions.jsonl

Usage:
    # 1. Calibrate (CPU, ~1-2 min)
    uv run python scripts/e6_leace.py \\
        --phase calibrate-leace \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf

    # 2. Smoke check on TallyQA
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_leace.py \\
        --phase smoke-leace \\
        --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/experiment_e5e_tallyqa_full/.../predictions.jsonl \\
        --dataset-tag tally

    # 3. Sweep TallyQA n=100
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_leace.py \\
        --phase sweep-leace \\
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
                    choices=("calibrate-leace", "smoke-leace", "sweep-leace"),
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
                    help="Comma-sep cell labels to run (e.g. 'baseline,L28_a1.0').")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _calib_dir(model: str, tag: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / f"calibration_{tag}"


def _eraser_dir(model: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / "leace_erasers"


def _sweep_out_path(model: str, dataset_tag: str, out_tag: str | None) -> Path:
    suffix = f"_{out_tag}" if out_tag else ""
    return (PROJECT_ROOT / "outputs" / "e6_steering" / model
            / f"sweep_leace_{dataset_tag}{suffix}_pooled" / "predictions.jsonl")


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


def _capture_last_token_residuals(runner, question: str, images: list) -> torch.Tensor:
    """Returns (n_layers, d_model) at last input token."""
    _seq_len, inputs = runner._prepare_inputs(question=question, images=images)
    out = runner.model(**inputs, output_hidden_states=True, use_cache=False)
    return torch.stack(
        [h[0, -1, :].detach().float().cpu() for h in out.hidden_states[1:]]
    )


# ---------------------------------------------------------------------------
# Phase: calibrate-leace
# ---------------------------------------------------------------------------

def _phase_calibrate_leace(args) -> None:
    from concept_erasure import LeaceEraser

    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    out_dir = _eraser_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pool Q_wrong + D_wrong across datasets
    Q_list, D_list = [], []
    for tag in calib_tags:
        cdir = _calib_dir(args.model, tag)
        q_path = cdir / "Q_wrong.pt"
        d_path = cdir / "D_wrong.pt"
        if not q_path.exists() or not d_path.exists():
            raise FileNotFoundError(
                f"Missing Q_wrong.pt or D_wrong.pt in {cdir}. "
                f"Run calibrate-qao and calibrate-subspace for tag={tag!r}.")
        Q_tag = torch.load(q_path, map_location="cpu", weights_only=True).float()
        D_tag = torch.load(d_path, map_location="cpu", weights_only=True).float()
        N = min(Q_tag.shape[0], D_tag.shape[0])
        print(f"  {tag}: Q_wrong={tuple(Q_tag.shape)}  D_wrong={tuple(D_tag.shape)}"
              f"  using N={N}")
        Q_list.append(Q_tag[:N])
        D_list.append(D_tag[:N])

    Q_pool = torch.cat(Q_list, dim=0)  # [N_total, n_layers, d]
    D_pool = torch.cat(D_list, dim=0)  # [N_total, n_layers, d]
    N_total = Q_pool.shape[0]
    n_layers = Q_pool.shape[1]
    d_model = Q_pool.shape[2]
    print(f"[calibrate-leace] pooled: N={N_total}  n_layers={n_layers}  d={d_model}")

    # Construct class labels
    Y_neg = torch.zeros(N_total, dtype=torch.long)
    Y_pos = torch.ones(N_total, dtype=torch.long)
    Y = torch.cat([Y_neg, Y_pos])

    # Fit one LeaceEraser per layer
    t0 = time.time()
    erasers = []
    for L in range(n_layers):
        X_neg = Q_pool[:, L, :]              # [N, d] — b-arm (no anchor)
        X_pos = Q_pool[:, L, :] + D_pool[:, L, :]  # [N, d] — approx a-arm
        X = torch.cat([X_neg, X_pos], dim=0)  # [2N, d]
        eraser = LeaceEraser.fit(X, Y)
        erasers.append(eraser)
        if L % 8 == 0:
            elapsed = time.time() - t0
            # quick concept magnitude sanity check
            sample_h = X_neg[:5]
            concept_norm = (sample_h - eraser(sample_h)).norm(dim=-1).mean().item()
            print(f"  L={L:02d}: concept_norm_avg={concept_norm:.4f}  elapsed={elapsed:.1f}s")

    # Save P matrices (LeaceEraser objects aren't torch.save-able easily, save P)
    # P shape: [d, d]; eraser(h) = h @ P
    P_stack = torch.stack([e.P.float() for e in erasers])  # [n_layers, d, d]
    torch.save(P_stack, out_dir / "P_stack.pt")

    elapsed = time.time() - t0
    meta = {
        "model": args.model,
        "calib_tags": calib_tags,
        "N_pooled": N_total,
        "n_layers": n_layers,
        "d_model": d_model,
        "P_stack_shape": list(P_stack.shape),
        "wall_seconds": elapsed,
        "note": ("X_neg=Q_wrong, X_pos=Q_wrong+D_wrong per-layer; "
                 "approximates LEACE on (h_b, h_a) distribution pairs"),
    }
    (out_dir / "eraser_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] calibrate-leace: {n_layers} erasers; "
          f"P_stack={tuple(P_stack.shape)}; wall={elapsed:.1f}s; saved {out_dir}")


# ---------------------------------------------------------------------------
# LEACE hook
# ---------------------------------------------------------------------------

def _make_leace_hook(P_layer: torch.Tensor, alpha: float):
    """Post-hook: h_last ← h_last − alpha × (h_last − h_last @ P).
    Only fires at prefill (seq_len > 1). P: [d, d] on CPU.
    """
    if alpha == 0.0:
        return None

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        if hidden.shape[1] <= 1:
            return output  # decode step — skip
        h = hidden[:, -1, :].float()  # [batch, d]
        P = P_layer.to(device=h.device, dtype=h.dtype)
        h_erased = h @ P              # concept removed (LEACE projection)
        h_new = h - alpha * (h - h_erased)
        hidden[:, -1, :] = h_new.to(hidden.dtype)
        return output

    return hook


def _install_leace_hook(layers, layer_idx: int, P: torch.Tensor, alpha: float):
    if alpha == 0.0:
        return []
    hook = _make_leace_hook(P[layer_idx], alpha)
    if hook is None:
        return []
    return [layers[layer_idx].register_forward_hook(hook)]


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

def _leace_cells() -> list[dict]:
    L_values = [20, 25, 28, 30, 31]
    alphas = [0.3, 0.5, 1.0, 2.0]
    cells: list[dict] = [{"L": -1, "alpha": 0.0, "label": "baseline"}]
    for L in L_values:
        for a in alphas:
            cells.append({"L": L, "alpha": a, "label": f"L{L:02d}_a{a}"})
    return cells


# ---------------------------------------------------------------------------
# Phase: smoke-leace
# ---------------------------------------------------------------------------

def _phase_smoke_leace(args) -> None:
    if not args.predictions_path:
        raise ValueError("--phase smoke-leace requires --predictions-path")
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

    p_dir = _eraser_dir(args.model)
    P_stack = torch.load(p_dir / "P_stack.pt", map_location="cpu", weights_only=True)

    runner = _build_runner(args)
    layers = _get_llm_layers(runner.model)

    L = 28
    alpha = 1.0
    print(f"[smoke-leace] {dataset_tag}: L={L} alpha={alpha}")
    n_changed = 0
    for sid in smoke_sids:
        a_rec = by_sid[sid].get("target_plus_irrelevant_number_S1")
        if not a_rec:
            continue
        imgs = _open_images(a_rec.get("input_image_paths") or [])

        out_base = runner.generate_number(question=a_rec["question"],
                                          images=imgs,
                                          max_new_tokens=args.max_new_tokens)
        handles = _install_leace_hook(layers, L, P_stack, alpha)
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
    print(f"[smoke-leace] {result}: {n_changed}/{len(smoke_sids)} outputs changed")


# ---------------------------------------------------------------------------
# Phase: sweep-leace
# ---------------------------------------------------------------------------

def _load_completed_leace_keys(path: Path) -> set:
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


def _phase_sweep_leace(args) -> None:
    if not args.predictions_path:
        raise ValueError("--phase sweep-leace requires --predictions-path")
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
    print(f"[sweep-leace] {dataset_tag}: {len(eligible)} sids")

    p_dir = _eraser_dir(args.model)
    P_stack = torch.load(p_dir / "P_stack.pt", map_location="cpu", weights_only=True)
    print(f"[sweep-leace] loaded P_stack {tuple(P_stack.shape)} from {p_dir}")

    cells = _leace_cells()
    if args.target_cells:
        keep = {c.strip() for c in args.target_cells.split(",")}
        cells = [c for c in cells if c["label"] in keep]
    print(f"[sweep-leace] {len(cells)} cells ({len(cells)-1} steered + 1 baseline)")

    out_path = _sweep_out_path(args.model, dataset_tag, args.out_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_leace_keys(out_path)
    print(f"[sweep-leace] {len(completed)} records already done")

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
            alpha = cell["alpha"]
            label = cell["label"]
            is_baseline = (label == "baseline")

            if not is_baseline:
                handles = _install_leace_hook(layers, L, P_stack, alpha)
            else:
                handles = []

            try:
                for si, sid in enumerate(eligible):
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
                            "cell_alpha": alpha,
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
            print(f"[sweep-leace] cell {cell_idx+1}/{len(cells)}: {label}  "
                  f"written={n_written}  elapsed={elapsed:.1f}s")

    print(f"[done] sweep-leace: {n_written} new records in {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    if args.phase == "calibrate-leace":
        _phase_calibrate_leace(args)
    elif args.phase == "smoke-leace":
        _phase_smoke_leace(args)
    elif args.phase == "sweep-leace":
        _phase_sweep_leace(args)


if __name__ == "__main__":
    main()
