"""Per-layer anchor-attention delta across the 4 E1 models.

Reads `per_step_attention.jsonl` files (already written by scripts/extract_attention_mass.py)
and produces, per layer:

    delta_layer = anchor_mass(number, layer l) − anchor_mass(neutral, layer l)

evaluated at both the answer-digit step and step 0 (prompt-integration step). Bootstrap
95 % CI per layer across sample_instance triplets. Also stratifies by the susceptibility
stratum used in the E1 writeup (top vs bottom decile `moved_closer_rate`).

Outputs, under `outputs/attention_analysis/_per_layer/`:

  - per_layer_deltas.csv       one row per (model, layer, step, stratum)
  - fig_delta_by_layer.png     4-panel layer trace, answer step, overall + top/bottom
  - fig_delta_by_layer_step0.png   same but for step 0
  - peak_layer_summary.csv     per model/step/stratum: argmax layer, peak delta, peak CI

Usage:
    uv run python scripts/analyze_attention_per_layer.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATT_ROOT = PROJECT_ROOT / "outputs" / "attention_analysis"

# Canonical runs identified in docs/experiments/E1-preliminary-results.md
# and docs/experiments/E1b-per-layer-localisation.md (after ConvLLaVA + FastVLM
# inputs_embeds-path extension).
CANONICAL_RUNS: dict[str, str] = {
    "gemma4-e4b": "20260424-115147",
    "qwen2.5-vl-7b-instruct": "20260424-120026",
    "llava-1.5-7b": "20260424-121139",
    "internvl3-8b": "20260424-121334",
    "convllava-7b": "20260424-134840",
    "fastvlm-7b": "20260424-135205",
    # OneVision (Phase 1 P0 v3 Main; AnyRes; populated post-run).
    # If absent, falls back to the latest run dir for the model.
}


def _resolve_run(model: str) -> str | None:
    """Return canonical run id for `model`, or auto-pick the latest run dir.

    Auto-fallback lets us add new models (e.g. OneVision) without
    pre-registering a timestamp.
    """
    if model in CANONICAL_RUNS:
        return CANONICAL_RUNS[model]
    model_dir = ATT_ROOT / model
    if not model_dir.exists():
        return None
    runs = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
    return runs[-1] if runs else None

_DIGIT_RE = re.compile(r"\d")


@dataclass
class TripletLayerArrays:
    """Per-layer number/neutral anchor-mass arrays for one step label.

    shape: (n_triplets, n_layers)
    """
    step_label: str
    num: np.ndarray  # anchor mass in number condition
    neut: np.ndarray  # anchor mass in neutral condition
    sample_ids: list[str]
    strata: list[str]
    base_correct: list[int]

    @property
    def delta(self) -> np.ndarray:
        return self.num - self.neut


def _find_answer_digit_step(per_step_tokens: list[dict]) -> int | None:
    for rec in per_step_tokens:
        if _DIGIT_RE.search(rec.get("token_text", "") or ""):
            return int(rec["step"])
    return None


def _load_records(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]


def _build_sample_step_map(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (sample_instance_id, condition) -> record."""
    return {(r["sample_instance_id"], r["condition"]): r for r in records
            if "error" not in r and r.get("per_step")}


def _layer_mass(record: dict, step: int | None, region: str) -> np.ndarray | None:
    if step is None:
        return None
    per_step = record.get("per_step", [])
    if step >= len(per_step):
        return None
    layers = per_step[step].get(region)
    if not layers:
        return None
    return np.asarray(layers, dtype=float)


def _derive_base_correct(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        if r.get("condition") != "target_only":
            continue
        pred = str(r.get("decoded", ""))
        m = re.search(r"-?\d+", pred)
        gt = str(r.get("ground_truth"))
        out[r["sample_instance_id"]] = int(bool(m) and m.group(0) == gt)
    return out


def _build_layer_arrays(
    records: list[dict],
    step_kind: str,
    susceptibility: dict[int, str],
) -> TripletLayerArrays | None:
    smap = _build_sample_step_map(records)
    base_correct = _derive_base_correct(records)
    sample_ids = sorted({sid for (sid, _) in smap.keys()})

    num_rows: list[np.ndarray] = []
    neut_rows: list[np.ndarray] = []
    kept_ids: list[str] = []
    strata: list[str] = []
    base_correct_list: list[int] = []

    for sid in sample_ids:
        r_num = smap.get((sid, "target_plus_irrelevant_number"))
        r_neut = smap.get((sid, "target_plus_irrelevant_neutral"))
        r_base = smap.get((sid, "target_only"))
        if r_num is None or r_neut is None or r_base is None:
            continue

        if step_kind == "answer":
            step_num = _find_answer_digit_step(r_num.get("per_step_tokens", []))
            step_neut = _find_answer_digit_step(r_neut.get("per_step_tokens", []))
        elif step_kind == "step0":
            step_num, step_neut = 0, 0
        else:
            raise ValueError(f"unknown step_kind={step_kind}")

        m_num = _layer_mass(r_num, step_num, "image_anchor")
        m_neut = _layer_mass(r_neut, step_neut, "image_anchor")
        if m_num is None or m_neut is None or m_num.shape != m_neut.shape:
            continue

        num_rows.append(m_num)
        neut_rows.append(m_neut)
        kept_ids.append(sid)
        qid = int(r_num["question_id"])
        strata.append(susceptibility.get(qid, "unknown"))
        base_correct_list.append(base_correct.get(sid, -1))

    if not num_rows:
        return None
    return TripletLayerArrays(
        step_label=step_kind,
        num=np.stack(num_rows),
        neut=np.stack(neut_rows),
        sample_ids=kept_ids,
        strata=strata,
        base_correct=base_correct_list,
    )


def _bootstrap_ci_per_layer(
    delta_matrix: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, ci_low, ci_high) vectors of shape (n_layers,)."""
    n, n_layers = delta_matrix.shape
    rng = np.random.default_rng(seed)
    mean = delta_matrix.mean(axis=0)
    if n <= 1:
        return mean, mean, mean
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = delta_matrix[idx].mean(axis=1)  # (n_boot, n_layers)
    ci_low = np.percentile(draws, 2.5, axis=0)
    ci_high = np.percentile(draws, 97.5, axis=0)
    return mean, ci_low, ci_high


def _budget_decomposition(
    records: list[dict],
    susceptibility: dict[int, str],
    n_layers: int,
) -> dict[str, np.ndarray]:
    """Mean per-layer delta (number − neutral) for each attention region at answer step.

    Returns dict with keys image_target/image_anchor/text/generated → (n_layers,) arrays.
    Lets us check whether a layer-5 anchor-gain is (a) anchor/target trade-off,
    (b) anchor pulling from text, or (c) distributed redistribution.
    """
    smap = _build_sample_step_map(records)
    sids = sorted({sid for (sid, _) in smap.keys()})
    accum = {k: np.zeros(n_layers) for k in ("image_target", "image_anchor", "text", "generated")}
    n_valid = 0
    for sid in sids:
        r_num = smap.get((sid, "target_plus_irrelevant_number"))
        r_neut = smap.get((sid, "target_plus_irrelevant_neutral"))
        if r_num is None or r_neut is None:
            continue
        s_num = _find_answer_digit_step(r_num.get("per_step_tokens", []))
        s_neut = _find_answer_digit_step(r_neut.get("per_step_tokens", []))
        if s_num is None or s_neut is None:
            continue
        if s_num >= len(r_num["per_step"]) or s_neut >= len(r_neut["per_step"]):
            continue
        for k in accum:
            v_num = r_num["per_step"][s_num].get(k)
            v_neut = r_neut["per_step"][s_neut].get(k)
            if not v_num or not v_neut or len(v_num) != n_layers or len(v_neut) != n_layers:
                break
            accum[k] += np.asarray(v_num) - np.asarray(v_neut)
        else:
            n_valid += 1
    if n_valid == 0:
        return {k: np.full(n_layers, np.nan) for k in accum}
    return {k: v / n_valid for k, v in accum.items()}


def _summarize(
    arrays: TripletLayerArrays,
    model: str,
    stratum_label: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    delta = arrays.delta
    mean, ci_low, ci_high = _bootstrap_ci_per_layer(delta, n_boot, seed)
    num_mean = arrays.num.mean(axis=0)
    neut_mean = arrays.neut.mean(axis=0)
    n_layers = delta.shape[1]
    rows = []
    for l in range(n_layers):
        rows.append({
            "model": model,
            "step": arrays.step_label,
            "stratum": stratum_label,
            "layer": l,
            "layer_frac": l / max(n_layers - 1, 1),
            "n": delta.shape[0],
            "delta_mean": float(mean[l]),
            "ci_low": float(ci_low[l]),
            "ci_high": float(ci_high[l]),
            "num_mean": float(num_mean[l]),
            "neut_mean": float(neut_mean[l]),
            "share_pos": float((delta[:, l] > 0).mean()),
        })
    return pd.DataFrame(rows)


def _peak_row(df_model_step: pd.DataFrame) -> dict:
    """Pick the layer with the largest positive delta_mean; fall back to largest |delta|."""
    pos = df_model_step.loc[df_model_step["delta_mean"] > 0]
    target = pos if not pos.empty else df_model_step
    peak = target.loc[target["delta_mean"].idxmax()].to_dict()
    return {
        "model": peak["model"],
        "step": peak["step"],
        "stratum": peak["stratum"],
        "peak_layer": int(peak["layer"]),
        "peak_layer_frac": float(peak["layer_frac"]),
        "peak_delta": float(peak["delta_mean"]),
        "peak_ci_low": float(peak["ci_low"]),
        "peak_ci_high": float(peak["ci_high"]),
        "n": int(peak["n"]),
        "n_layers": int(df_model_step["layer"].max() + 1),
    }


def _plot_grid(
    per_layer_df: pd.DataFrame,
    step_kind: str,
    out_path: Path,
    model_order: list[str],
) -> None:
    n_models = len(model_order)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), sharex=False)
    axes = axes.flatten()
    for ax, model in zip(axes, model_order):
        sub = per_layer_df.loc[
            (per_layer_df["model"] == model) & (per_layer_df["step"] == step_kind)
        ]
        if sub.empty:
            ax.set_visible(False)
            continue
        # Overall
        overall = sub.loc[sub["stratum"] == "all"].sort_values("layer")
        ax.plot(overall["layer"], overall["delta_mean"], color="black", lw=1.8, label="overall")
        ax.fill_between(overall["layer"], overall["ci_low"], overall["ci_high"], color="black", alpha=0.12)

        # Stratum lines
        for stratum, color in [
            ("top_decile_susceptible", "#c0392b"),
            ("bottom_decile_resistant", "#2e86de"),
        ]:
            s = sub.loc[sub["stratum"] == stratum].sort_values("layer")
            if s.empty:
                continue
            ax.plot(s["layer"], s["delta_mean"], color=color, lw=1.2, label=stratum.split("_")[0])
            ax.fill_between(s["layer"], s["ci_low"], s["ci_high"], color=color, alpha=0.10)

        ax.axhline(0.0, color="grey", lw=0.6, ls=":")
        n_layers = int(sub["layer"].max() + 1)
        ax.set_title(f"{model}  (n_layers={n_layers})", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("delta(anchor mass)  number − neutral")
        ax.legend(fontsize=8, loc="best")
    for extra in axes[len(model_order):]:
        extra.set_visible(False)
    fig.suptitle(f"Per-layer anchor-attention delta — {step_kind} step", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--susceptibility-csv",
        type=str,
        default="docs/insights/_data/susceptibility_strata.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/attention_analysis/_per_layer",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    susc_path = PROJECT_ROOT / args.susceptibility_csv
    susc_df = pd.read_csv(susc_path)
    susceptibility = dict(zip(susc_df["question_id"].astype(int), susc_df["susceptibility_stratum"]))

    # Discover models from disk (anything with a run dir under ATT_ROOT) so
    # late-arriving panel additions (e.g. OneVision) are picked up
    # automatically without editing CANONICAL_RUNS.
    discovered = sorted(
        p.name for p in ATT_ROOT.iterdir()
        if p.is_dir() and p.name not in {"_per_layer", "analysis"}
    ) if ATT_ROOT.exists() else []
    model_order = list(CANONICAL_RUNS.keys()) + [
        m for m in discovered if m not in CANONICAL_RUNS
    ]

    per_layer_rows: list[pd.DataFrame] = []
    peak_rows: list[dict] = []
    budget_rows: list[dict] = []

    for model in model_order:
        run_id = _resolve_run(model)
        if run_id is None:
            print(f"[{model}] skipping — no run dir")
            continue
        jsonl = ATT_ROOT / model / run_id / "per_step_attention.jsonl"
        if not jsonl.exists():
            print(f"[{model}] skipping — run {run_id} not present at {jsonl}")
            continue
        print(f"[{model}] loading {jsonl.relative_to(PROJECT_ROOT)}")
        records = _load_records(jsonl)
        print(f"  records={len(records)}")

        for step_kind in ("answer", "step0"):
            arrays = _build_layer_arrays(records, step_kind, susceptibility)
            if arrays is None:
                print(f"  [{step_kind}] no valid triplets")
                continue

            df_overall = _summarize(arrays, model, "all", args.bootstrap_n, args.rng_seed)
            per_layer_rows.append(df_overall)
            peak_rows.append(_peak_row(df_overall))

            for stratum_name in ("top_decile_susceptible", "bottom_decile_resistant"):
                mask = np.array([s == stratum_name for s in arrays.strata])
                if mask.sum() == 0:
                    continue
                sub_arrays = TripletLayerArrays(
                    step_label=arrays.step_label,
                    num=arrays.num[mask],
                    neut=arrays.neut[mask],
                    sample_ids=[s for s, m in zip(arrays.sample_ids, mask) if m],
                    strata=[s for s, m in zip(arrays.strata, mask) if m],
                    base_correct=[b for b, m in zip(arrays.base_correct, mask) if m],
                )
                df_strat = _summarize(sub_arrays, model, stratum_name, args.bootstrap_n, args.rng_seed)
                per_layer_rows.append(df_strat)
                peak_rows.append(_peak_row(df_strat))

            overall_peak = next(
                p for p in reversed(peak_rows)
                if p["model"] == model and p["step"] == step_kind and p["stratum"] == "all"
            )
            print(
                f"  [{step_kind}] n_triplets={arrays.num.shape[0]}  "
                f"n_layers={arrays.num.shape[1]}  "
                f"peak_layer={overall_peak['peak_layer']}  "
                f"peak_delta={overall_peak['peak_delta']:+.5f}"
            )

        # Budget decomposition at the answer step peak — one row per model
        answer_peak = next(
            p for p in peak_rows
            if p["model"] == model and p["step"] == "answer" and p["stratum"] == "all"
        )
        n_layers_m = int(answer_peak["n_layers"])
        budget = _budget_decomposition(records, susceptibility, n_layers_m)
        pl = int(answer_peak["peak_layer"])
        budget_rows.append({
            "model": model,
            "peak_layer": pl,
            "d_image_anchor": float(budget["image_anchor"][pl]),
            "d_image_target": float(budget["image_target"][pl]),
            "d_text": float(budget["text"][pl]),
            "d_generated": float(budget["generated"][pl]),
            "sum_check": float(
                budget["image_anchor"][pl] + budget["image_target"][pl]
                + budget["text"][pl] + budget["generated"][pl]
            ),
        })

    per_layer_df = pd.concat(per_layer_rows, ignore_index=True)
    peak_df = pd.DataFrame(peak_rows)
    budget_df = pd.DataFrame(budget_rows)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_layer_csv = out_dir / "per_layer_deltas.csv"
    peak_csv = out_dir / "peak_layer_summary.csv"
    budget_csv = out_dir / "peak_budget_decomposition.csv"
    per_layer_df.to_csv(per_layer_csv, index=False)
    peak_df.to_csv(peak_csv, index=False)
    budget_df.to_csv(budget_csv, index=False)
    print(f"[write] {per_layer_csv}")
    print(f"[write] {peak_csv}")
    print(f"[write] {budget_csv}")

    _plot_grid(per_layer_df, "answer", out_dir / "fig_delta_by_layer_answer.png", model_order)
    _plot_grid(per_layer_df, "step0", out_dir / "fig_delta_by_layer_step0.png", model_order)
    print(f"[write] {out_dir / 'fig_delta_by_layer_answer.png'}")
    print(f"[write] {out_dir / 'fig_delta_by_layer_step0.png'}")

    print("\n=== Peak-layer summary (overall, answer step) ===")
    view = peak_df.loc[(peak_df["stratum"] == "all")].copy()
    view["peak_layer_frac"] = view["peak_layer_frac"].round(3)
    view["peak_delta"] = view["peak_delta"].round(5)
    view["peak_ci_low"] = view["peak_ci_low"].round(5)
    view["peak_ci_high"] = view["peak_ci_high"].round(5)
    print(view[["model", "step", "n", "n_layers", "peak_layer", "peak_layer_frac",
                "peak_delta", "peak_ci_low", "peak_ci_high"]].to_string(index=False))

    print("\n=== Peak-layer budget decomposition (answer step, number − neutral) ===")
    bv = budget_df.copy()
    for c in ("d_image_anchor", "d_image_target", "d_text", "d_generated", "sum_check"):
        bv[c] = bv[c].round(5)
    print(bv.to_string(index=False))


if __name__ == "__main__":
    main()
