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
    """
    runs = _all_runs(model)
    if not runs:
        return None
    rows = []
    for run_dir in runs:
        path = run_dir / "predictions.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "model": r["model"],
                "sample_instance_id": r["sample_instance_id"],
                "question_id": int(r["question_id"]),
                "condition": r["condition"],
                "mode": r["mode"],
                "anchor_value": _parse_int(r.get("anchor_value")),
                "parsed_number": _parse_int(r.get("parsed_number")),
                "error": r.get("error"),
                "_run_order": run_dir.name,
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.sort_values("_run_order").drop_duplicates(
        subset=["sample_instance_id", "condition", "mode"], keep="last"
    ).drop(columns=["_run_order"])
    return df.reset_index(drop=True)


def _build_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-(sample_instance_id, mode) rows with base / number / anchor columns.

    base_pred is taken from condition=target_only, mode=baseline (ablations on target_only
    are noops because anchor span is empty for single-image samples, so we can collapse).
    """
    base = (
        df.loc[(df["condition"] == "target_only") & (df["mode"] == "baseline"),
               ["sample_instance_id", "parsed_number"]]
          .rename(columns={"parsed_number": "base_pred"})
          .drop_duplicates(subset=["sample_instance_id"])
    )
    num_rows = df.loc[df["condition"] == "target_plus_irrelevant_number"].copy()
    num_rows = num_rows.rename(columns={"parsed_number": "num_pred"})
    num_rows = num_rows.merge(base, on="sample_instance_id", how="inner")
    return num_rows


def _compute_metrics(triplets: pd.DataFrame) -> dict[str, float | int]:
    valid = triplets.dropna(subset=["base_pred", "num_pred", "anchor_value"])
    n = len(valid)
    if n == 0:
        return {"n": 0, "direction_follow_rate": np.nan, "adoption_rate": np.nan,
                "mean_distance_to_anchor": np.nan}
    diff_base = (valid["base_pred"] - valid["anchor_value"]).abs()
    diff_num = (valid["num_pred"] - valid["anchor_value"]).abs()
    pulled = (diff_num < diff_base).sum()
    adopted = (valid["num_pred"] == valid["anchor_value"]).sum()
    mean_dist = float(diff_num.mean())
    return {
        "n": int(n),
        "direction_follow_rate": float(pulled / n),
        "adoption_rate": float(adopted / n),
        "mean_distance_to_anchor": mean_dist,
    }


def _bootstrap_ci(triplets: pd.DataFrame, metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    valid = triplets.dropna(subset=["base_pred", "num_pred", "anchor_value"]).reset_index(drop=True)
    n = len(valid)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    base_pred = valid["base_pred"].to_numpy()
    num_pred = valid["num_pred"].to_numpy()
    anchor = valid["anchor_value"].to_numpy()
    diff_b = np.abs(base_pred - anchor)
    diff_n = np.abs(num_pred - anchor)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "direction_follow_rate":
            draws[i] = (diff_n[idx] < diff_b[idx]).mean()
        elif metric == "adoption_rate":
            draws[i] = (num_pred[idx] == anchor[idx]).mean()
        elif metric == "mean_distance_to_anchor":
            draws[i] = diff_n[idx].mean()
        else:
            raise ValueError(metric)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def _load_susceptibility() -> dict[int, str]:
    path = PROJECT_ROOT / "docs/insights/_data/susceptibility_strata.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["question_id"].astype(int), df["susceptibility_stratum"]))


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
        df["stratum"] = df["question_id"].map(susceptibility)
        triplets_all = _build_triplets(df)

        for mode in sorted(df["mode"].unique(), key=_mode_sort_key):
            sub = triplets_all.loc[triplets_all["mode"] == mode]
            stats = _compute_metrics(sub)
            ci_df_lo, ci_df_hi = _bootstrap_ci(sub, "direction_follow_rate",
                                               args.bootstrap_n, args.rng_seed)
            ci_ad_lo, ci_ad_hi = _bootstrap_ci(sub, "adoption_rate",
                                               args.bootstrap_n, args.rng_seed)
            ci_md_lo, ci_md_hi = _bootstrap_ci(sub, "mean_distance_to_anchor",
                                               args.bootstrap_n, args.rng_seed)
            per_model_rows.append({
                "model": model, "mode": mode, "stratum": "all",
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
                    "model": model, "mode": mode, "stratum": stratum_name,
                    **s_stats,
                    "df_ci_low": s_ci_lo, "df_ci_high": s_ci_hi,
                })

    df_summary = pd.DataFrame(per_model_rows)
    df_stratum = pd.DataFrame(stratum_rows)

    # Attach delta-vs-baseline columns (per-model)
    baseline = df_summary.loc[df_summary["mode"] == "baseline",
                              ["model", "direction_follow_rate", "adoption_rate",
                               "mean_distance_to_anchor"]]
    baseline = baseline.rename(columns={
        "direction_follow_rate": "base_df",
        "adoption_rate": "base_adopt",
        "mean_distance_to_anchor": "base_md",
    })
    df_summary = df_summary.merge(baseline, on="model", how="left")
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
    def _bar_plot(metric: str, ylabel: str, out_path: Path) -> None:
        models = [m for m in PANEL_MODELS if (df_summary["model"] == m).any()]
        modes = [m for m in MODE_ORDER if (df_summary["mode"] == m).any()]
        x = np.arange(len(models))
        w = 0.8 / max(len(modes), 1)
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(models)), 5))
        colours = plt.cm.tab10(np.linspace(0, 1, len(modes)))
        for i, mode in enumerate(modes):
            sub = df_summary[df_summary["mode"] == mode].set_index("model")
            vals = np.array([sub.loc[m, metric] if m in sub.index else np.nan for m in models])
            lo = np.array([sub.loc[m, f"{metric_to_ci(metric)}_ci_low"] if m in sub.index else np.nan for m in models])
            hi = np.array([sub.loc[m, f"{metric_to_ci(metric)}_ci_high"] if m in sub.index else np.nan for m in models])
            yerr = np.abs(np.vstack([vals - lo, hi - vals]))
            ax.bar(x + (i - len(modes) / 2 + 0.5) * w, vals, width=w,
                   color=colours[i], label=mode, yerr=yerr, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Causal anchor ablation — {metric}")
        ax.legend(fontsize=8, loc="best")
        ax.axhline(0, color="grey", lw=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def metric_to_ci(metric: str) -> str:
        return {
            "direction_follow_rate": "df",
            "adoption_rate": "adopt",
            "mean_distance_to_anchor": "mean_dist",
        }[metric]

    _bar_plot("direction_follow_rate", "direction_follow rate",
              SUMMARY_DIR / "fig_direction_follow.png")
    _bar_plot("adoption_rate", "adoption rate",
              SUMMARY_DIR / "fig_adoption.png")
    print(f"[write] {SUMMARY_DIR / 'fig_direction_follow.png'}")
    print(f"[write] {SUMMARY_DIR / 'fig_adoption.png'}")

    # ─── Printable summary ───
    print("\n=== direction_follow by model × mode ===")
    pivot_df = df_summary.pivot(index="model", columns="mode",
                                 values="direction_follow_rate").reindex(
        index=PANEL_MODELS, columns=MODE_ORDER)
    print(pivot_df.round(3).to_string())
    print("\n=== adoption by model × mode ===")
    pivot_ad = df_summary.pivot(index="model", columns="mode",
                                values="adoption_rate").reindex(
        index=PANEL_MODELS, columns=MODE_ORDER)
    print(pivot_ad.round(3).to_string())


if __name__ == "__main__":
    main()
