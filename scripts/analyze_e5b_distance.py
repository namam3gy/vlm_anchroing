"""E5b — per-stratum direction-follow / adoption / em / distance with CIs + plots.

Loads the latest run from `outputs/experiment_distance_vqa/<model>/` and
`outputs/experiment_distance_tally/<model>/`, computes per-stratum stats
with bootstrap CIs, draws two figures (per-dataset distance curve,
cross-dataset overlay), and writes a tidy CSV summary.

Usage: `uv run python scripts/analyze_e5b_distance.py [--model llava-next-interleaved-7b]`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "VQAv2": "experiment_distance_vqa",
    "TallyQA": "experiment_distance_tally",
}
STRATUM_ORDER = ["S1", "S2", "S3", "S4", "S5"]
STRATUM_MIDPOINT = {"S1": 0.5, "S2": 3.5, "S3": 18.0, "S4": 165.0, "S5": 650.0}
STRATUM_LABEL = {"S1": "[0,1]", "S2": "[2,5]", "S3": "[6,30]", "S4": "[31,300]", "S5": "[301,inf)"}
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _latest_run_dir(experiment_dir: str, model: str) -> Path:
    base = PROJECT_ROOT / "outputs" / experiment_dir / model
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run dirs under {base}")
    return runs[-1]


def _load_records(model: str) -> pd.DataFrame:
    frames = []
    for ds_label, ds_dir in DATASETS.items():
        run_dir = _latest_run_dir(ds_dir, model)
        recs = [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]
        df = pd.DataFrame(recs)
        df["dataset"] = ds_label
        df["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.nanmean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_stratum_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in records["dataset"].unique():
        ds = records[records["dataset"] == dataset]
        baseline = ds[ds["condition"] == "target_only"]
        baseline_df_mean = float(np.nanmean(baseline["anchor_direction_followed"].astype(float)))
        for stratum in STRATUM_ORDER:
            cell = ds[ds["condition"] == f"target_plus_irrelevant_number_{stratum}"]
            if cell.empty:
                continue
            df_vals = cell["anchor_direction_followed"].astype(float).to_numpy()
            adoption_vals = cell["anchor_adopted"].astype(float).to_numpy()
            em_vals = cell["exact_match"].astype(float).to_numpy()
            dist_vals = cell["numeric_distance_to_anchor"].astype(float).to_numpy()
            df_lo, df_hi = _bootstrap_ci(df_vals)
            rows.append({
                "dataset": dataset,
                "stratum": stratum,
                "stratum_range": STRATUM_LABEL[stratum],
                "stratum_midpoint": STRATUM_MIDPOINT[stratum],
                "n": len(cell),
                "direction_follow_rate": float(np.nanmean(df_vals)),
                "direction_follow_ci_lo": df_lo,
                "direction_follow_ci_hi": df_hi,
                "adoption_rate": float(np.nanmean(adoption_vals)),
                "exact_match": float(np.nanmean(em_vals)),
                "mean_distance_to_anchor": float(np.nanmean(dist_vals)) if not np.isnan(dist_vals).all() else float("nan"),
                "df_minus_baseline": float(np.nanmean(df_vals)) - baseline_df_mean,
            })
    return pd.DataFrame(rows)


def plot_distance_curve(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        ds = summary[summary["dataset"] == dataset].sort_values("stratum_midpoint")
        x = ds["stratum_midpoint"].to_numpy()
        y = ds["direction_follow_rate"].to_numpy()
        lo = ds["direction_follow_ci_lo"].to_numpy()
        hi = ds["direction_follow_ci_hi"].to_numpy()
        ax.plot(x, y, "o-", color="C0")
        ax.fill_between(x, lo, hi, color="C0", alpha=0.2)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks(ds["stratum_midpoint"].tolist())
        ax.set_xticklabels(ds["stratum"].tolist())
        ax.set_xlabel("Anchor distance stratum (|a - GT|)")
        ax.set_title(f"{dataset}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("direction-follow rate (95% CI)")
    fig.suptitle("E5b - anchor effect vs distance (llava-interleave-7b, n~=1000/dataset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_overlay(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset, color in [("VQAv2", "C0"), ("TallyQA", "C1")]:
        ds = summary[summary["dataset"] == dataset].sort_values("stratum_midpoint")
        x = ds["stratum_midpoint"].to_numpy()
        y = ds["direction_follow_rate"].to_numpy()
        lo = ds["direction_follow_ci_lo"].to_numpy()
        hi = ds["direction_follow_ci_hi"].to_numpy()
        ax.plot(x, y, "o-", color=color, label=dataset)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xscale("symlog", linthresh=2)
    midpoints = sorted(STRATUM_MIDPOINT.values())
    ax.set_xticks(midpoints)
    ax.set_xticklabels(STRATUM_ORDER)
    ax.set_xlabel("Anchor distance stratum (|a - GT|)")
    ax.set_ylabel("direction-follow rate (95% CI)")
    ax.set_title("E5b - cross-dataset anchor-distance comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(model: str = "llava-next-interleaved-7b") -> dict:
    records = _load_records(model)
    summary = per_stratum_summary(records)

    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5b_per_stratum.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_distance_curve(summary, fig_dir / "E5b_distance_curve.png")
    plot_cross_dataset_overlay(summary, fig_dir / "E5b_cross_dataset_overlay.png")

    return {
        "summary": summary,
        "n_records": len(records),
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "figures": [
            "docs/figures/E5b_distance_curve.png",
            "docs/figures/E5b_cross_dataset_overlay.png",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-next-interleaved-7b")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run(model=args.model)
    print(out["summary"].to_string(index=False))
    print(f"\nwrote {out['out_csv']}")
    for f in out["figures"]:
        print(f"wrote {f}")
