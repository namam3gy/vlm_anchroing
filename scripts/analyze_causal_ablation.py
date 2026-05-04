"""Analyze the causal anchor-ablation run.

Reads predictions.jsonl files written by scripts/causal_anchor_ablation.py and
produces:

  - outputs/causal_ablation/_summary/per_model_per_mode.csv  — direction_follow_rate,
    adoption_rate, mean_distance_to_anchor with bootstrap 95 % CIs; delta vs baseline
    with CI per mode.
  - outputs/causal_ablation/_summary/fig_direction_follow.png — grouped bar chart,
    one bar per model × mode.
  - outputs/causal_ablation/_summary/fig_adoption.png — parallel for adoption.
  - outputs/causal_ablation/_summary/by_stratum.csv — same metrics split by
    susceptibility stratum (top_decile vs bottom_decile).

Usage:
    uv run python scripts/analyze_causal_ablation.py

Canonical runs are read from `outputs/causal_ablation/<model>/<latest-run>/predictions.jsonl`
for each of the 6 models in the E1b panel.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABL_ROOT = PROJECT_ROOT / "outputs" / "causal_ablation"
SUMMARY_DIR = ABL_ROOT / "_summary"

PANEL_MODELS = [
    "gemma4-e4b",
    "internvl3-8b",
    "llava-1.5-7b",
    "convllava-7b",
    "qwen2.5-vl-7b-instruct",
    "fastvlm-7b",
    "llava-onevision-qwen2-7b-ov",  # Phase 1 P0 v3 Main; AnyRes anchor span
]

MODE_ORDER = [
    "baseline",
    "ablate_layer0",
    "ablate_peak",
    "ablate_peak_window",
    "ablate_lower_half",
    "ablate_upper_half",
    "ablate_all",
]

# Phase E (Phase 1 P0 v3) ran the Main model OneVision across 4 datasets in
# separate timestamped run dirs. Other panel models ran on the legacy
# single-dataset (VQAv2-anchored) E1d. The mapping below partitions
# OneVision rows by run timestamp so per-(model, dataset) cells can be
# emitted instead of one all-data row that mixes 4 datasets.
#
# 20260503-002050 → plotqa (orphaned ChartQA-with-PlotQA-CSV bug; bonus 5th cell, footnote only)
# 20260503-111305 → tallyqa
# 20260503-112116 → infovqa
# 20260504-015933 → chartqa (recovery, commit 2d11876)
# 20260504-021140 → mathvista (recovery, commit 2d11876)
ONEVISION_RUN_DATASET = {
    "20260503-002050": "plotqa",
    "20260503-111305": "tallyqa",
    "20260503-112116": "infovqa",
    "20260504-015933": "chartqa",
    "20260504-021140": "mathvista",
}
MULTI_DATASET_MODELS = {
    "llava-onevision-qwen2-7b-ov": ONEVISION_RUN_DATASET,
}
HEADLINE_DATASETS = ["tallyqa", "infovqa", "chartqa", "mathvista"]

_NUM_RE = re.compile(r"-?\d+")


def _parse_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = _NUM_RE.search(str(value))
    return int(m.group(0)) if m else None


def _all_runs(model: str) -> list[Path]:
    base = ABL_ROOT / model
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def _latest_run(model: str) -> Path | None:
    runs = _all_runs(model)
    return runs[-1] if runs else None


def _load_model(model: str) -> pd.DataFrame | None:
    """Load predictions from ALL run dirs and dedupe by (sample, condition, mode).

    Later runs (sorted by directory name, which is a timestamp) override earlier ones for
    overlapping (sample_instance_id, condition, mode) tuples. This lets us layer additive
    control runs (e.g. ablate_layer0) on top of an earlier multi-mode run.

    Multi-dataset models (Phase E OneVision, see ``MULTI_DATASET_MODELS``) are
    tagged with a ``dataset`` column derived from the run timestamp. Other models
    get ``dataset = "vqav2"`` (the legacy single-dataset E1d setup).
    """
    runs = _all_runs(model)
    if not runs:
        return None
    run_to_dataset = MULTI_DATASET_MODELS.get(model)
    rows = []
    for run_dir in runs:
        path = run_dir / "predictions.jsonl"
        if not path.exists():
            continue
        if run_to_dataset is not None:
            dataset = run_to_dataset.get(run_dir.name)
            if dataset is None:
                # Unknown timestamp for a multi-dataset model — skip rather than
                # silently merge into a wrong cell.
                continue
        else:
            dataset = "vqav2"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "model": r["model"],
                "dataset": dataset,
                "sample_instance_id": r["sample_instance_id"],
                "question_id": int(r["question_id"]),
                "condition": r["condition"],
                "mode": r["mode"],
                "anchor_value": _parse_int(r.get("anchor_value")),
                "parsed_number": _parse_int(r.get("parsed_number")),
                "error": r.get("error"),
                # M2 / C-form per-row flags written by reaggregate_paired_adoption.py.
                # Use these directly so this script's rates match the canonical
                # `metrics.py::summarize_condition` definitions used in §3.3 / §5.
                "anchor_adopted_M2": int(r.get("anchor_adopted") or 0),
                "anchor_direction_followed_moved": int(r.get("anchor_direction_followed_moved") or 0),
                "pred_b_equal_anchor": int(r.get("pred_b_equal_anchor") or 0),
                "numeric_distance_to_anchor": r.get("numeric_distance_to_anchor"),
                "_run_order": run_dir.name,
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values("_run_order").drop_duplicates(
        subset=["dataset", "sample_instance_id", "condition", "mode"], keep="last"
    ).drop(columns=["_run_order"])
    return df.reset_index(drop=True)


def _build_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-(dataset, sample_instance_id, mode) rows with base / number / anchor columns.

    base_pred is taken from condition=target_only, mode=baseline (ablations on target_only
    are noops because anchor span is empty for single-image samples, so we can collapse).

    The dataset key is included so multi-dataset models (Phase E OneVision) don't
    cross-pollinate base preds across datasets even if sids accidentally collide.
    """
    base = (
        df.loc[(df["condition"] == "target_only") & (df["mode"] == "baseline"),
               ["dataset", "sample_instance_id", "parsed_number"]]
          .rename(columns={"parsed_number": "base_pred"})
          .drop_duplicates(subset=["dataset", "sample_instance_id"])
    )
    num_rows = df.loc[df["condition"] == "target_plus_irrelevant_number"].copy()
    num_rows = num_rows.rename(columns={"parsed_number": "num_pred"})
    num_rows = num_rows.merge(base, on=["dataset", "sample_instance_id"], how="inner")
    return num_rows


def _compute_metrics(triplets: pd.DataFrame) -> dict[str, float | int]:
    """Per-cell M2 / C-form rates.

    Pre-2026-04-28 this used the Phase-A pull-form
    `(|num_pred − anchor| < |base_pred − anchor|).mean()` plus pre-M1
    marginal adoption `(num_pred == anchor).mean()`. Refactored to read the
    canonical M2 flags written by `reaggregate_paired_adoption.py`:

      direction_follow_rate = #(C-form moved) / #(numeric pair AND anchor present)
      adoption_rate         = #(pa == anchor AND pb != anchor) / #(pb != anchor)
      mean_distance         = mean(|pa − anchor|) over the same numeric-anchor subset

    so that §7 ablation deltas match the §3.3 / §5 headline metrics by
    construction.
    """
    valid = triplets.dropna(subset=["base_pred", "num_pred", "anchor_value"])
    if valid.empty:
        return {"n": 0, "direction_follow_rate": np.nan, "adoption_rate": np.nan,
                "mean_distance_to_anchor": np.nan}

    df_eligible = valid[valid["numeric_distance_to_anchor"].notna()]
    n_df = len(df_eligible)
    pulled = int(df_eligible["anchor_direction_followed_moved"].sum()) if n_df else 0

    adopt_eligible = valid[valid["pred_b_equal_anchor"] == 0]
    n_adopt = len(adopt_eligible)
    adopted = int(adopt_eligible["anchor_adopted_M2"].sum()) if n_adopt else 0

    return {
        "n": int(n_df),
        "n_pb_ne_anchor": int(n_adopt),
        "direction_follow_rate": float(pulled / n_df) if n_df else np.nan,
        "adoption_rate": float(adopted / n_adopt) if n_adopt else np.nan,
        "mean_distance_to_anchor": (
            float(df_eligible["numeric_distance_to_anchor"].astype(float).mean())
            if n_df else np.nan
        ),
    }


def _bootstrap_ci(triplets: pd.DataFrame, metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    """C-form / M2 bootstrap CI matching `_compute_metrics`.

    Each bootstrap draw resamples valid triplets uniformly; the per-draw
    rate uses the same M2 predicates as the point estimate (paired denominators
    for adoption, numeric-pair-with-anchor denominator for direction-follow).
    """
    valid = triplets.dropna(subset=["base_pred", "num_pred", "anchor_value"]).reset_index(drop=True)
    n = len(valid)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    df_moved_arr = valid["anchor_direction_followed_moved"].astype(float).to_numpy()
    adopt_arr = valid["anchor_adopted_M2"].astype(float).to_numpy()
    pb_eq_a_arr = valid["pred_b_equal_anchor"].astype(float).to_numpy()
    dist_arr = valid["numeric_distance_to_anchor"].astype(float).to_numpy()  # NaN where None
    df_eligible_mask = ~np.isnan(dist_arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "direction_follow_rate":
            elig = df_eligible_mask[idx]
            n_elig = int(elig.sum())
            draws[i] = float(df_moved_arr[idx][elig].sum() / n_elig) if n_elig else np.nan
        elif metric == "adoption_rate":
            elig = (pb_eq_a_arr[idx] == 0.0)
            n_elig = int(elig.sum())
            draws[i] = float(adopt_arr[idx][elig].sum() / n_elig) if n_elig else np.nan
        elif metric == "mean_distance_to_anchor":
            elig = df_eligible_mask[idx]
            d = dist_arr[idx][elig]
            draws[i] = float(d.mean()) if d.size else np.nan
        else:
            raise ValueError(metric)
    valid_draws = draws[~np.isnan(draws)]
    if valid_draws.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(valid_draws, 2.5)), float(np.percentile(valid_draws, 97.5)))


def _load_susceptibility() -> dict[int, str]:
    path = PROJECT_ROOT / "docs/insights/_data/susceptibility_strata.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["question_id"].astype(int), df["susceptibility_stratum"]))


def _load_susceptibility_per_dataset() -> dict[str, dict[int, str]]:
    """Load per-dataset OneVision susceptibility CSVs.

    Used for OneVision Phase E E1d cells where each dataset has its own
    top/bottom-decile partition; the legacy single-file
    ``susceptibility_strata.csv`` is panel-wide and doesn't apply.
    """
    out: dict[str, dict[int, str]] = {}
    data_dir = PROJECT_ROOT / "docs/insights/_data"
    for ds in HEADLINE_DATASETS + ["plotqa"]:
        path = data_dir / f"susceptibility_{ds}_onevision.csv"
        if path.exists():
            df = pd.read_csv(path)
            out[ds] = dict(zip(df["question_id"].astype(int),
                               df["susceptibility_stratum"]))
    return out


def _mode_sort_key(m: str) -> int:
    try:
        return MODE_ORDER.index(m)
    except ValueError:
        return 99


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    susceptibility = _load_susceptibility()
    susceptibility_per_ds = _load_susceptibility_per_dataset()

    per_model_rows: list[dict] = []
    stratum_rows: list[dict] = []
    for model in PANEL_MODELS:
        df = _load_model(model)
        if df is None:
            print(f"[{model}] skipping — no run found")
            continue
        runs = _all_runs(model)
        rel_runs = ", ".join(p.name for p in runs)
        print(f"[{model}] loading {len(runs)} run(s) [{rel_runs}] -> {len(df)} unique records")

        # Per-row stratum: panel-wide map for legacy models; per-dataset
        # OneVision susceptibility for Phase E.
        if model in MULTI_DATASET_MODELS:
            df["stratum"] = [
                susceptibility_per_ds.get(ds, {}).get(int(qid))
                for ds, qid in zip(df["dataset"], df["question_id"])
            ]
        else:
            df["stratum"] = df["question_id"].map(susceptibility)

        triplets_all = _build_triplets(df)

        # Iterate per (dataset, mode). Single-dataset models have one dataset cell.
        for dataset in sorted(triplets_all["dataset"].unique()):
            ds_triplets = triplets_all.loc[triplets_all["dataset"] == dataset]
            for mode in sorted(ds_triplets["mode"].unique(), key=_mode_sort_key):
                sub = ds_triplets.loc[ds_triplets["mode"] == mode]
                stats = _compute_metrics(sub)
                ci_df_lo, ci_df_hi = _bootstrap_ci(sub, "direction_follow_rate",
                                                   args.bootstrap_n, args.rng_seed)
                ci_ad_lo, ci_ad_hi = _bootstrap_ci(sub, "adoption_rate",
                                                   args.bootstrap_n, args.rng_seed)
                ci_md_lo, ci_md_hi = _bootstrap_ci(sub, "mean_distance_to_anchor",
                                                   args.bootstrap_n, args.rng_seed)
                per_model_rows.append({
                    "model": model, "dataset": dataset,
                    "mode": mode, "stratum": "all",
                    **stats,
                    "df_ci_low": ci_df_lo, "df_ci_high": ci_df_hi,
                    "adopt_ci_low": ci_ad_lo, "adopt_ci_high": ci_ad_hi,
                    "mean_dist_ci_low": ci_md_lo, "mean_dist_ci_high": ci_md_hi,
                })

                for stratum_name in ("top_decile_susceptible", "bottom_decile_resistant"):
                    mask = sub["stratum"] == stratum_name
                    stratum_sub = sub.loc[mask]
                    s_stats = _compute_metrics(stratum_sub)
                    s_ci_lo, s_ci_hi = _bootstrap_ci(stratum_sub, "direction_follow_rate",
                                                     args.bootstrap_n, args.rng_seed)
                    stratum_rows.append({
                        "model": model, "dataset": dataset,
                        "mode": mode, "stratum": stratum_name,
                        **s_stats,
                        "df_ci_low": s_ci_lo, "df_ci_high": s_ci_hi,
                    })

    df_summary = pd.DataFrame(per_model_rows)
    df_stratum = pd.DataFrame(stratum_rows)

    # Attach delta-vs-baseline columns (per (model, dataset))
    baseline = df_summary.loc[df_summary["mode"] == "baseline",
                              ["model", "dataset", "direction_follow_rate",
                               "adoption_rate", "mean_distance_to_anchor"]]
    baseline = baseline.rename(columns={
        "direction_follow_rate": "base_df",
        "adoption_rate": "base_adopt",
        "mean_distance_to_anchor": "base_md",
    })
    df_summary = df_summary.merge(baseline, on=["model", "dataset"], how="left")
    df_summary["delta_df"] = df_summary["direction_follow_rate"] - df_summary["base_df"]
    df_summary["delta_adopt"] = df_summary["adoption_rate"] - df_summary["base_adopt"]
    df_summary["delta_md"] = df_summary["mean_distance_to_anchor"] - df_summary["base_md"]

    per_csv = SUMMARY_DIR / "per_model_per_mode.csv"
    stratum_csv = SUMMARY_DIR / "by_stratum.csv"
    df_summary.to_csv(per_csv, index=False)
    df_stratum.to_csv(stratum_csv, index=False)
    print(f"[write] {per_csv}")
    print(f"[write] {stratum_csv}")

    # ─── Plots ───
    def metric_to_ci(metric: str) -> str:
        return {
            "direction_follow_rate": "df",
            "adoption_rate": "adopt",
            "mean_distance_to_anchor": "mean_dist",
        }[metric]

    def _bar_plot_panel(df_panel: pd.DataFrame, x_keys: list[str], x_labels: list[str],
                        x_col: str, metric: str, ylabel: str, title: str,
                        out_path: Path) -> None:
        modes = [m for m in MODE_ORDER if (df_panel["mode"] == m).any()]
        x = np.arange(len(x_keys))
        w = 0.8 / max(len(modes), 1)
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(x_keys)), 5))
        colours = plt.cm.tab10(np.linspace(0, 1, len(modes)))
        for i, mode in enumerate(modes):
            sub = df_panel[df_panel["mode"] == mode].set_index(x_col)
            vals = np.array([sub.loc[k, metric] if k in sub.index else np.nan for k in x_keys])
            lo = np.array([sub.loc[k, f"{metric_to_ci(metric)}_ci_low"] if k in sub.index else np.nan for k in x_keys])
            hi = np.array([sub.loc[k, f"{metric_to_ci(metric)}_ci_high"] if k in sub.index else np.nan for k in x_keys])
            yerr = np.abs(np.vstack([vals - lo, hi - vals]))
            ax.bar(x + (i - len(modes) / 2 + 0.5) * w, vals, width=w,
                   color=colours[i], label=mode, yerr=yerr, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.axhline(0, color="grey", lw=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Cross-model panel: vqav2-only (the legacy single-dataset E1d setup; one
    # row per model so the per-(model, dataset) shape collapses cleanly).
    panel_df = df_summary[df_summary["dataset"] == "vqav2"]
    panel_models = [m for m in PANEL_MODELS if (panel_df["model"] == m).any()]
    if panel_models:
        _bar_plot_panel(panel_df, panel_models, panel_models, "model",
                        "direction_follow_rate", "direction_follow rate",
                        "Causal anchor ablation — direction_follow_rate (legacy panel)",
                        SUMMARY_DIR / "fig_direction_follow.png")
        _bar_plot_panel(panel_df, panel_models, panel_models, "model",
                        "adoption_rate", "adoption rate",
                        "Causal anchor ablation — adoption_rate (legacy panel)",
                        SUMMARY_DIR / "fig_adoption.png")
        print(f"[write] {SUMMARY_DIR / 'fig_direction_follow.png'}")
        print(f"[write] {SUMMARY_DIR / 'fig_adoption.png'}")

    # Per-dataset panel for multi-dataset Main models (Phase E OneVision).
    for model in MULTI_DATASET_MODELS:
        ds_df = df_summary[(df_summary["model"] == model) &
                            df_summary["dataset"].isin(HEADLINE_DATASETS)]
        if ds_df.empty:
            continue
        ds_keys = [d for d in HEADLINE_DATASETS if (ds_df["dataset"] == d).any()]
        slug = model.split("/")[-1]
        _bar_plot_panel(ds_df, ds_keys, ds_keys, "dataset",
                        "direction_follow_rate", "direction_follow rate",
                        f"Phase E E1d — {model} per dataset",
                        SUMMARY_DIR / f"fig_direction_follow_{slug}_per_dataset.png")
        _bar_plot_panel(ds_df, ds_keys, ds_keys, "dataset",
                        "adoption_rate", "adoption rate",
                        f"Phase E E1d — {model} per dataset",
                        SUMMARY_DIR / f"fig_adoption_{slug}_per_dataset.png")
        print(f"[write] {SUMMARY_DIR / f'fig_direction_follow_{slug}_per_dataset.png'}")

    # ─── Printable summary ───
    print("\n=== direction_follow by (model, dataset) × mode ===")
    pivot_df = df_summary.pivot_table(index=["model", "dataset"], columns="mode",
                                       values="direction_follow_rate",
                                       aggfunc="first")
    pivot_df = pivot_df.reindex(columns=[m for m in MODE_ORDER if m in pivot_df.columns])
    print(pivot_df.round(3).to_string())
    print("\n=== adoption by (model, dataset) × mode ===")
    pivot_ad = df_summary.pivot_table(index=["model", "dataset"], columns="mode",
                                       values="adoption_rate", aggfunc="first")
    pivot_ad = pivot_ad.reindex(columns=[m for m in MODE_ORDER if m in pivot_ad.columns])
    print(pivot_ad.round(3).to_string())


if __name__ == "__main__":
    main()
