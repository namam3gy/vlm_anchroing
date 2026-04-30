"""E6 Method 2 — Query-Adaptive Offset (QAO / AFTER-style).

Pipeline phases
---------------
  calibrate-qao : Extract b-arm (target_only) hidden states for calibration
                  samples from a predictions.jsonl.  Saves Q.pt [N, n_layers,
                  d_model] alongside the existing D_wrong.pt in
                  outputs/e6_steering/<model>/calibration_<tag>/.

  train-probe   : PCA + Ridge regression probe.  Loads Q.pt and D_wrong.pt
                  from all three calibration dirs (vqa, tally, chartqa),
                  pools them, fits per-(L_q, L_target) probes, saves
                  outputs/e6_steering/<model>/qao_probe/probe_Lq<q>_Lt<t>.pt.

  smoke-qao     : 5-sample wiring check.  Verifies that the query-adaptive
                  hook changes the output digit on at least one sample.

  sweep-qao     : Full (L_q × L_target × alpha) sweep on n=100 wrong-base
                  sids from a predictions.jsonl (TallyQA or ChartQA first,
                  then VQAv2 if cross-dataset passes).

Methodology (AFTER arXiv:2601.01957):
  h ← h − α · (v_general[L_target] + δ)
  where δ = probe(q), q = b-arm residual at last token, layer L_q.
  v_general = mean of v_wrong across VQA + TallyQA + ChartQA calibration sets.
  No anchor labels at inference — probe operates on the target-only repr.

Usage examples:
    # 1. Extract b-arm reprs for VQA calibration (run from vlm_anchroing/)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_query_adaptive_offset.py \\
        --phase calibrate-qao --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/experiment_e5c_vqa/llava-next-interleaved-7b/<ts>/predictions.jsonl \\
        --dataset-tag vqa

    # 2. Same for TallyQA and ChartQA (--dataset-tag tally / chartqa)

    # 3. Train probe on pooled calibration data (CPU, ~1 min)
    uv run python scripts/e6_query_adaptive_offset.py \\
        --phase train-probe --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf

    # 4. Smoke on TallyQA
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_query_adaptive_offset.py \\
        --phase smoke-qao --model llava-next-interleaved-7b \\
        --hf-model llava-hf/llava-interleave-qwen-7b-hf \\
        --predictions-path outputs/.../predictions.jsonl \\
        --dataset-tag tally

    # 5. Sweep TallyQA n=100
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_query_adaptive_offset.py \\
        --phase sweep-qao --model llava-next-interleaved-7b \\
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
import yaml

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
                    choices=("calibrate-qao", "train-probe",
                             "smoke-qao", "sweep-qao"),
                    required=True)
    ap.add_argument("--model", required=True,
                    help="Project model id (e.g. llava-next-interleaved-7b)")
    ap.add_argument("--hf-model", required=True,
                    help="HuggingFace model id")
    ap.add_argument("--config",
                    default="configs/experiment_e5c_vqa.yaml")
    ap.add_argument("--predictions-path", default=None,
                    help="calibrate-qao / smoke-qao / sweep-qao: path to an "
                         "E5e predictions.jsonl with input_image_paths fields.")
    ap.add_argument("--dataset-tag", default=None,
                    help="Short label (vqa / tally / chartqa). Inferred from "
                         "predictions-path parent dirs if omitted.")
    ap.add_argument("--max-sweep-sids", type=int, default=100,
                    help="Cap wrong-base sids for sweep-qao per dataset.")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--pca-components", type=int, default=100,
                    help="PCA dimensionality for probe input compression.")
    ap.add_argument("--ridge-alpha", type=float, default=1e3,
                    help="Ridge regularisation coefficient for probe fitting.")
    ap.add_argument("--probe-dir", default=None,
                    help="Override probe directory (default: "
                         "outputs/e6_steering/<model>/qao_probe/)")
    ap.add_argument("--calib-tags", default="vqa,tally,chartqa",
                    help="Comma-separated calibration dataset tags used in "
                         "train-probe (must have Q.pt + D_wrong.pt).")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _calib_dir(model: str, tag: str) -> Path:
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / f"calibration_{tag}"


def _probe_dir(model: str, probe_dir_override: str | None) -> Path:
    if probe_dir_override:
        p = Path(probe_dir_override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / "outputs" / "e6_steering" / model / "qao_probe"


def _build_runner(args) -> Any:
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
    runner = build_eager_runner(args.hf_model, inference_config=inference_cfg)
    return runner


def _capture_last_token_residuals(runner: Any, question: str,
                                   images: list) -> torch.Tensor:
    """Forward pass on (question, images); return (n_layers, d_model) at last token."""
    _seq_len, inputs = runner._prepare_inputs(question=question, images=images)
    out = runner.model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states  # tuple of (1, seq_len, d) per layer+1
    return torch.stack(
        [h[0, -1, :].detach().to(torch.float32).cpu() for h in hidden[1:]]
    )  # (n_layers, d_model)


def _load_predictions(pred_path: Path) -> dict[str, dict[str, dict]]:
    """Load predictions.jsonl; return by_sid_cond[sid][condition] = record."""
    by_sid_cond: dict = defaultdict(dict)
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_sid_cond[str(r["sample_instance_id"])][r["condition"]] = r
    return by_sid_cond


def _wrong_sids_from_predictions(by_sid_cond: dict) -> set[str]:
    return {sid for sid, d in by_sid_cond.items()
            if d.get("target_only", {}).get("exact_match") == 0}


def _infer_dataset_tag(pred_path: Path) -> str:
    try:
        return pred_path.parents[2].name
    except IndexError:
        return "alt"


def _open_images(paths: list[str]):
    from PIL import Image as _Image
    return [_Image.open(p).convert("RGB") for p in (paths or [])]


# ---------------------------------------------------------------------------
# v_general: mean v_wrong across datasets
# ---------------------------------------------------------------------------

def _load_v_general(model: str, calib_tags: list[str]) -> torch.Tensor:
    """Average v_wrong across calibration datasets; shape (n_layers, d_model)."""
    vs = []
    for tag in calib_tags:
        p = _calib_dir(model, tag) / "v.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run calibrate-subspace or calibrate-qao first.")
        v = torch.load(p, map_location="cpu", weights_only=True)  # [2, n_layers, d]
        vs.append(v[0].float())  # v_wrong
    return torch.stack(vs).mean(0)  # (n_layers, d_model)


# ---------------------------------------------------------------------------
# Phase: calibrate-qao
# ---------------------------------------------------------------------------

def _phase_calibrate_qao(args) -> None:
    """Extract b-arm (target_only) hidden states for calibration samples.

    Reads an E5e predictions.jsonl; for every sample with a 'target_only'
    record, loads the target image(s) from input_image_paths and runs a
    forward pass.  Saves Q.pt [N, n_layers, d_model] to the calibration dir.
    """
    if not args.predictions_path:
        raise ValueError("--phase calibrate-qao requires --predictions-path")
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path

    dataset_tag = args.dataset_tag or _infer_dataset_tag(pred_path)
    out_dir = _calib_dir(args.model, dataset_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_sid_cond = _load_predictions(pred_path)
    wrong_sids = _wrong_sids_from_predictions(by_sid_cond)

    # Prioritise wrong-base sids (same ordering as calibrate-subspace)
    all_sids = list(by_sid_cond.keys())
    ordered = ([s for s in all_sids if s in wrong_sids]
               + [s for s in all_sids if s not in wrong_sids])

    print(f"[setup] {dataset_tag}: {len(ordered)} total sids, "
          f"{len(wrong_sids)} wrong-base")

    runner = _build_runner(args)

    Q_wrong: list[torch.Tensor] = []
    Q_all: list[torch.Tensor] = []
    n_skipped = 0
    t0 = time.time()

    for i, sid in enumerate(ordered):
        b_rec = by_sid_cond[sid].get("target_only")
        if b_rec is None:
            n_skipped += 1
            continue
        try:
            images = _open_images(b_rec.get("input_image_paths") or [])
            res = _capture_last_token_residuals(
                runner, b_rec["question"], images)
        except Exception as exc:
            print(f"  [error] sid={sid}: {exc}")
            n_skipped += 1
            continue

        Q_all.append(res)
        if sid in wrong_sids:
            Q_wrong.append(res)

        n = len(Q_all)
        if n % 20 == 0 or n == 1:
            elapsed = time.time() - t0
            print(f"  [progress] {n}/{len(ordered)} sids "
                  f"(wrong-base: {len(Q_wrong)}); elapsed={elapsed:.1f}s")

    if not Q_all:
        raise RuntimeError("No samples collected. Check predictions path.")

    Q_all_t = torch.stack(Q_all)    # [N, n_layers, d_model]
    Q_wrong_t = (torch.stack(Q_wrong) if Q_wrong
                 else torch.zeros(0, *Q_all_t.shape[1:]))
    torch.save(Q_all_t, out_dir / "Q_all.pt")
    torch.save(Q_wrong_t, out_dir / "Q_wrong.pt")
    meta = {
        "model": args.model, "dataset_tag": dataset_tag,
        "n_all": len(Q_all), "n_wrong": len(Q_wrong),
        "n_skipped": n_skipped,
        "Q_all_shape": list(Q_all_t.shape),
        "Q_wrong_shape": list(Q_wrong_t.shape),
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "Q_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {dataset_tag}: Q_all={tuple(Q_all_t.shape)} "
          f"Q_wrong={tuple(Q_wrong_t.shape)}; saved {out_dir}")


# ---------------------------------------------------------------------------
# Phase: train-probe
# ---------------------------------------------------------------------------

def _sweep_probe_pairs() -> list[tuple[int, int]]:
    """(L_q, L_target) pairs to train probes for."""
    L_q_values = [20, 25, 30, 31]
    L_target_values = [28, 30, 31]
    return [(lq, lt) for lq in L_q_values for lt in L_target_values]


def _fit_probe(Q: torch.Tensor, Y: torch.Tensor,
               n_components: int, ridge_alpha: float) -> dict:
    """Ridge regression with PCA input compression.

    Q : [N, d_model] — b-arm repr at L_q
    Y : [N, d_model] — D_wrong direction at L_target (NOT normalised; raw diff)
    Returns dict with pca_mean, pca_components, probe_W.
    """
    Q = Q.float()
    Y = Y.float()

    # PCA on Q
    pca_mean = Q.mean(0)  # [d]
    Q_c = Q - pca_mean    # [N, d]
    _, _, Vh = torch.linalg.svd(Q_c, full_matrices=False)
    n_components = min(n_components, Vh.shape[0], Q.shape[0] - 1)
    pca_components = Vh[:n_components]  # [P, d]

    X = Q_c @ pca_components.T  # [N, P] — PCA scores

    # Ridge: (X^T X + λI)^{-1} X^T Y
    P = X.shape[1]
    XtX = X.T @ X                                            # [P, P]
    XtY = X.T @ Y                                            # [P, d]
    reg = ridge_alpha * torch.eye(P, dtype=X.dtype)
    W = torch.linalg.solve(XtX + reg, XtY)                  # [P, d]

    return {
        "pca_mean": pca_mean.to(torch.float32),
        "pca_components": pca_components.to(torch.float32),
        "probe_W": W.to(torch.float32),
    }


def _phase_train_probe(args) -> None:
    """Pool Q_wrong.pt + D_wrong.pt across calibration tags, fit per-(L_q, L_target) probes."""
    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    out_dir = _probe_dir(args.model, args.probe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and pool calibration data
    Q_list, D_list = [], []
    for tag in calib_tags:
        cdir = _calib_dir(args.model, tag)
        q_path = cdir / "Q_wrong.pt"
        d_path = cdir / "D_wrong.pt"
        if not q_path.exists():
            raise FileNotFoundError(
                f"Missing {q_path}. Run --phase calibrate-qao for tag={tag!r}.")
        if not d_path.exists():
            raise FileNotFoundError(
                f"Missing {d_path}. Run --phase calibrate-subspace for tag={tag!r}.")
        Q_tag = torch.load(q_path, map_location="cpu", weights_only=True).float()
        D_tag = torch.load(d_path, map_location="cpu", weights_only=True).float()
        n = min(Q_tag.shape[0], D_tag.shape[0])
        print(f"  {tag}: Q_wrong={tuple(Q_tag.shape)}  D_wrong={tuple(D_tag.shape)}"
              f"  using n={n}")
        Q_list.append(Q_tag[:n])
        D_list.append(D_tag[:n])

    Q_all = torch.cat(Q_list, dim=0)  # [N_pool, n_layers, d]
    D_all = torch.cat(D_list, dim=0)  # [N_pool, n_layers, d]
    N = Q_all.shape[0]
    n_layers = Q_all.shape[1]
    print(f"[train-probe] pooled: N={N}  n_layers={n_layers}  d_model={Q_all.shape[2]}")

    pairs = _sweep_probe_pairs()
    print(f"[train-probe] fitting {len(pairs)} (L_q, L_target) probes "
          f"(pca_components={args.pca_components}  ridge_alpha={args.ridge_alpha})")

    t0 = time.time()
    saved = []
    for L_q, L_target in pairs:
        Q_lq = Q_all[:, L_q, :]       # [N, d]
        D_lt = D_all[:, L_target, :]   # [N, d]
        probe = _fit_probe(Q_lq, D_lt, args.pca_components, args.ridge_alpha)
        fname = f"probe_Lq{L_q:02d}_Lt{L_target:02d}.pt"
        torch.save(probe, out_dir / fname)
        saved.append(fname)
        # quick train residual
        pca_mean = probe["pca_mean"]
        pca_comp = probe["pca_components"]
        W = probe["probe_W"]
        X = (Q_lq - pca_mean) @ pca_comp.T
        Y_pred = X @ W
        rmse = (D_lt - Y_pred).norm(dim=-1).mean().item()
        print(f"  Lq={L_q} Lt={L_target}: rmse(D_wrong)={rmse:.3f}")

    elapsed = time.time() - t0
    meta = {
        "model": args.model,
        "calib_tags": calib_tags,
        "n_pooled": N,
        "pca_components": args.pca_components,
        "ridge_alpha": args.ridge_alpha,
        "pairs": [{"L_q": lq, "L_target": lt} for lq, lt in pairs],
        "files": saved,
        "wall_seconds": elapsed,
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] train-probe: {len(saved)} probes; wall={elapsed:.1f}s; saved {out_dir}")


# ---------------------------------------------------------------------------
# Query-adaptive hook
# ---------------------------------------------------------------------------

def _apply_probe(probe: dict, q: torch.Tensor) -> torch.Tensor:
    """δ = probe(q); q: [d_model] cpu float32; returns [d_model] cpu float32."""
    pca_mean = probe["pca_mean"]       # [d]
    pca_comp = probe["pca_components"] # [P, d]
    W = probe["probe_W"]               # [P, d]
    x = (q - pca_mean) @ pca_comp.T   # [P]
    return x @ W                       # [d]


def _make_qao_hook(correction: torch.Tensor):
    """Hook: h[:, -1, :] -= correction at prefill; skip at decode (seq_len==1)."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        if hidden.shape[1] <= 1:
            return output  # decode step — no-op
        c = correction.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] - c
        return output
    return hook


# ---------------------------------------------------------------------------
# Phase: smoke-qao
# ---------------------------------------------------------------------------

def _phase_smoke_qao(args) -> None:
    """5-sample wiring check: verify hook changes at least one digit output."""
    if not args.predictions_path:
        raise ValueError("--phase smoke-qao requires --predictions-path")
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path

    dataset_tag = args.dataset_tag or _infer_dataset_tag(pred_path)
    by_sid_cond = _load_predictions(pred_path)
    wrong_sids = _wrong_sids_from_predictions(by_sid_cond)

    # Pick 5 wrong-base anchor-arm samples
    smoke_sids = [s for s in wrong_sids
                  if "target_plus_irrelevant_number_S1" in by_sid_cond[s]][:5]
    if not smoke_sids:
        raise RuntimeError("No wrong-base a-arm samples found in predictions.")

    # Load v_general and probe for L_q=30, L_target=31 (near-miss layer)
    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    v_general = _load_v_general(args.model, calib_tags)  # [n_layers, d]
    p_dir = _probe_dir(args.model, args.probe_dir)
    probe_path = p_dir / "probe_Lq30_Lt31.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing probe at {probe_path}. Run train-probe first.")
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)

    runner = _build_runner(args)
    layers = _get_llm_layers(runner.model)
    L_target = 31
    alpha = 4.0

    print(f"[smoke-qao] {dataset_tag}: L_q=30 L_target={L_target} alpha={alpha}")
    n_changed = 0
    for sid in smoke_sids:
        b_rec = by_sid_cond[sid].get("target_only")
        a_rec = by_sid_cond[sid].get("target_plus_irrelevant_number_S1")
        if b_rec is None or a_rec is None:
            continue

        # Get b-arm repr
        try:
            b_imgs = _open_images(b_rec.get("input_image_paths") or [])
            q_repr = _capture_last_token_residuals(runner, b_rec["question"], b_imgs)
            # [n_layers, d]
        except Exception as exc:
            print(f"  [skip] sid={sid}: b-arm capture failed: {exc}")
            continue

        q = q_repr[30]  # [d] at L_q=30
        delta = _apply_probe(probe, q)
        correction = (alpha * (v_general[L_target] + delta))  # [d]

        a_imgs = _open_images(a_rec.get("input_image_paths") or [])

        # Baseline (no hook)
        out_base = runner.generate_number(
            question=a_rec["question"], images=a_imgs,
            max_new_tokens=args.max_new_tokens)

        # Steered
        handle = layers[L_target].register_forward_hook(_make_qao_hook(correction))
        try:
            out_steered = runner.generate_number(
                question=a_rec["question"], images=a_imgs,
                max_new_tokens=args.max_new_tokens)
        finally:
            handle.remove()

        changed = out_base.get("parsed_number") != out_steered.get("parsed_number")
        if changed:
            n_changed += 1
        print(f"  sid={sid}: base={out_base.get('parsed_number')} "
              f"steered={out_steered.get('parsed_number')} "
              f"gt={a_rec.get('ground_truth')}  changed={changed}")

    if n_changed == 0:
        print("[smoke-qao] FAIL: 0 outputs changed under QAO hook")
    else:
        print(f"[smoke-qao] PASS: {n_changed}/{len(smoke_sids)} outputs changed")


# ---------------------------------------------------------------------------
# Phase: sweep-qao
# ---------------------------------------------------------------------------

def _qao_sweep_cells() -> list[dict]:
    """(L_q × L_target × alpha) cells + 1 baseline = 49 total."""
    L_q_values = [20, 25, 30, 31]
    L_target_values = [28, 30, 31]
    alphas = [0.5, 1.0, 2.0, 4.0]
    cells: list[dict] = [{"L_q": -1, "L_target": -1, "alpha": 0.0,
                           "label": "baseline"}]
    for lq in L_q_values:
        for lt in L_target_values:
            for a in alphas:
                cells.append({
                    "L_q": lq, "L_target": lt, "alpha": a,
                    "label": f"Lq{lq:02d}_Lt{lt:02d}_a{a}",
                })
    return cells


def _sweep_qao_output_path(model: str, dataset_tag: str) -> Path:
    return (PROJECT_ROOT / "outputs" / "e6_steering" / model
            / f"sweep_qao_{dataset_tag}_pooled" / "predictions.jsonl")


def _load_completed_qao_keys(path: Path) -> set:
    """(sid, condition, L_q, L_target, alpha) tuples already written."""
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
                    int(r.get("cell_L_q", -1)),
                    int(r.get("cell_L_target", -1)),
                    float(r["cell_alpha"]),
                ))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    return completed


def _phase_sweep_qao(args) -> None:
    if not args.predictions_path:
        raise ValueError("--phase sweep-qao requires --predictions-path")
    pred_path = Path(args.predictions_path)
    if not pred_path.is_absolute():
        pred_path = PROJECT_ROOT / args.predictions_path
    dataset_tag = args.dataset_tag or _infer_dataset_tag(pred_path)

    by_sid_cond = _load_predictions(pred_path)
    wrong_sids = _wrong_sids_from_predictions(by_sid_cond)

    # Select n wrong-base sids with both b-arm and a-arm records
    eligible = [
        s for s in wrong_sids
        if ("target_only" in by_sid_cond[s] and
            "target_plus_irrelevant_number_S1" in by_sid_cond[s])
    ]
    if args.max_sweep_sids:
        eligible = eligible[:args.max_sweep_sids]

    print(f"[sweep-qao] {dataset_tag}: {len(eligible)} wrong-base sids "
          f"(capped at {args.max_sweep_sids})")

    calib_tags = [t.strip() for t in args.calib_tags.split(",") if t.strip()]
    v_general = _load_v_general(args.model, calib_tags)  # [n_layers, d]
    p_dir = _probe_dir(args.model, args.probe_dir)

    # Pre-load all probes
    probes: dict[tuple[int, int], dict] = {}
    for L_q, L_target in _sweep_probe_pairs():
        probe_path = p_dir / f"probe_Lq{L_q:02d}_Lt{L_target:02d}.pt"
        if not probe_path.exists():
            raise FileNotFoundError(
                f"Missing {probe_path}. Run train-probe first.")
        probes[(L_q, L_target)] = torch.load(
            probe_path, map_location="cpu", weights_only=True)
    print(f"[sweep-qao] loaded {len(probes)} probes from {p_dir}")

    cells = _qao_sweep_cells()
    print(f"[sweep-qao] grid: {len(cells)} cells "
          f"(1 baseline + {len(cells)-1} steered)")

    out_path = _sweep_qao_output_path(args.model, dataset_tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_qao_keys(out_path)
    print(f"[sweep-qao] {len(completed)} records already in {out_path}")

    runner = _build_runner(args)
    layers = _get_llm_layers(runner.model)

    # Run all four conditions (including target_only for accuracy-preservation check)
    sweep_conditions = [
        "target_only",
        "target_plus_irrelevant_number_S1",
        "target_plus_irrelevant_number_masked_S1",
    ]

    # Pre-extract b-arm reprs for all eligible sids (shared across cells)
    print(f"[sweep-qao] pre-extracting b-arm reprs for {len(eligible)} sids ...")
    q_cache: dict[str, torch.Tensor] = {}  # sid → [n_layers, d]
    t_repr = time.time()
    for i, sid in enumerate(eligible):
        b_rec = by_sid_cond[sid]["target_only"]
        try:
            b_imgs = _open_images(b_rec.get("input_image_paths") or [])
            q_cache[sid] = _capture_last_token_residuals(
                runner, b_rec["question"], b_imgs)
        except Exception as exc:
            print(f"  [warn] sid={sid} b-arm extraction failed: {exc}")
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_repr
            print(f"  [{i+1}/{len(eligible)}] elapsed={elapsed:.1f}s")
    print(f"[sweep-qao] b-arm cache: {len(q_cache)} sids; "
          f"wall={time.time()-t_repr:.1f}s")

    n_done = 0
    n_skipped = 0
    total_target = len(cells) * len(eligible) * len(sweep_conditions)
    t0 = time.time()

    with out_path.open("a") as fh:
        for cell in cells:
            L_q = cell["L_q"]
            L_target = cell["L_target"]
            alpha = cell["alpha"]
            is_baseline = (alpha == 0.0)

            # Pre-load probe for this (L_q, L_target) if steered
            probe_key = (L_q, L_target)
            probe = probes.get(probe_key) if not is_baseline else None

            for sid in eligible:
                if sid not in q_cache and not is_baseline:
                    n_skipped += len(sweep_conditions)
                    continue

                # Compute per-sample correction (baseline: None)
                correction = None
                if not is_baseline and probe is not None:
                    q = q_cache[sid][L_q]  # [d]
                    delta = _apply_probe(probe, q)
                    correction = alpha * (v_general[L_target] + delta)  # [d]

                for cond_name in sweep_conditions:
                    rec0 = by_sid_cond[sid].get(cond_name)
                    if rec0 is None:
                        n_skipped += 1
                        continue

                    key = (sid, cond_name, L_q, L_target, alpha)
                    if key in completed:
                        n_skipped += 1
                        continue

                    imgs = _open_images(rec0.get("input_image_paths") or [])

                    handle = None
                    if correction is not None:
                        handle = layers[L_target].register_forward_hook(
                            _make_qao_hook(correction))
                    try:
                        out = runner.generate_number(
                            question=rec0["question"], images=imgs,
                            max_new_tokens=args.max_new_tokens)
                        err = None
                    except Exception as exc:
                        out = {"raw_text": None, "parsed_number": None}
                        err = str(exc)
                    finally:
                        if handle is not None:
                            handle.remove()

                    anchor_int = None
                    if rec0.get("anchor_value") is not None:
                        try:
                            anchor_int = int(str(rec0["anchor_value"]).strip())
                        except (ValueError, TypeError):
                            pass

                    record = {
                        "model": args.model,
                        "dataset_tag": dataset_tag,
                        "sample_instance_id": sid,
                        "question_id": rec0.get("question_id"),
                        "condition": cond_name,
                        "ground_truth": rec0.get("ground_truth"),
                        "anchor_value": anchor_int,
                        "is_wrong_base": True,
                        "cell_label": cell["label"],
                        "cell_L_q": L_q,
                        "cell_L_target": L_target,
                        "cell_alpha": alpha,
                        "raw_text": out.get("raw_text"),
                        "parsed_number": out.get("parsed_number"),
                        "exact_match": (
                            1 if (out.get("parsed_number") is not None and
                                  str(out.get("parsed_number")) ==
                                  str(rec0.get("ground_truth")))
                            else 0),
                        "error": err,
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    n_done += 1

                    if n_done % 200 == 0 or n_done == 1:
                        elapsed = time.time() - t0
                        rate = n_done / max(elapsed, 1)
                        remaining = total_target - (n_done + n_skipped)
                        eta = remaining / max(rate, 1e-6)
                        print(f"  [progress] cell={cell['label']:30s} "
                              f"done={n_done} skipped={n_skipped} "
                              f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    elapsed = time.time() - t0
    total_done = n_done + n_skipped
    print(f"[done] sweep-qao {dataset_tag}: {n_done} new + {n_skipped} resumed = "
          f"{total_done}/{total_target} target  wall={elapsed/60:.1f}min")
    print(f"[done] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if args.phase == "calibrate-qao":
        _phase_calibrate_qao(args)
    elif args.phase == "train-probe":
        _phase_train_probe(args)
    elif args.phase == "smoke-qao":
        _phase_smoke_qao(args)
    elif args.phase == "sweep-qao":
        _phase_sweep_qao(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase!r}")


if __name__ == "__main__":
    main()
