"""E5b — anchor-distance analysis with paired conditional adoption.

Loads the latest stratified runs under
outputs/experiment_distance_vqa/<model>/ and
outputs/experiment_distance_tally/<model>/, computes per-stratum
paired adoption rate (case 4 excluded from denominator), stratifies
by base-prediction correctness, and writes:

- docs/insights/_data/E5b_per_stratum.csv      — full per-(dataset, stratum, base) table
- docs/figures/E5b_adopt_cond_curve.png         — per-dataset adopt_cond vs stratum, two lines (correct/wrong)
- docs/figures/E5b_adopt_cond_overlay.png       — cross-dataset overlay (wrong-base only)

Primary metric is `adopt_cond`. Direction-follow numbers are computed
and written to the CSV for reference but are not plotted at the
headline level (they're noisier than adoption — see roadmap §10
2026-04-27 entry).
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
    "VQAv2":   "experiment_distance_vqa",
    "TallyQA": "experiment_distance_tally",
}
STRATUM_ORDER = ["S1", "S2", "S3", "S4", "S5"]
STRATUM_MIDPOINT = {"S1": 0.5, "S2": 3.5, "S3": 18.0, "S4": 165.0, "S5": 650.0}
STRATUM_LABEL = {"S1": "[0,1]", "S2": "[2,5]", "S3": "[6,30]", "S4": "[31,300]", "S5": "[301,inf)"}
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _normalize_int_str(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    return s if s.lstrip("-").lstrip("+").isdigit() else None


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


def _classify(records: pd.DataFrame) -> pd.DataFrame:
    """Add columns: norm_pred, norm_anchor, norm_gt, base_pred, base_correct,
    case_id (1/2/3/4 or NaN for target_only), adopt_eligible (True for cases 1/2/3)."""
    df = records.copy()
    df["norm_pred"]   = df["prediction"].apply(_normalize_int_str)
    df["norm_anchor"] = df["anchor_value"].apply(_normalize_int_str)
    df["norm_gt"]     = df["ground_truth"].apply(_normalize_int_str)

    base_pred_map = (
        df[df["condition"] == "target_only"]
        .set_index("sample_instance_id")["norm_pred"]
        .to_dict()
    )
    df["base_pred"] = df["sample_instance_id"].map(base_pred_map)
    df["base_correct"] = df["base_pred"] == df["norm_gt"]

    def case_id(row):
        if row["condition"] == "target_only":
            return None
        a, p, b = row["norm_anchor"], row["norm_pred"], row["base_pred"]
        if a is None or p is None or b is None:
            return None
        if b != a and p != a: return 1
        if b != a and p == a: return 2
        if b == a and p != a: return 3
        if b == a and p == a: return 4
        return None

    df["case_id"] = df.apply(case_id, axis=1)
    df["adopt_eligible"] = df["case_id"].isin([1, 2, 3])
    df["adopted"] = df["case_id"] == 2
    return df


def _bootstrap_rate(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_cell_summary(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, stratum, base_correct)."""
    df = _classify(records)
    rows = []
    for dataset in df["dataset"].unique():
        for stratum in STRATUM_ORDER:
            for base_correct, base_label in [(True, "correct"), (False, "wrong")]:
                cell = df[
                    (df["dataset"] == dataset)
                    & (df["anchor_stratum_id"] == stratum)
                    & (df["base_correct"] == base_correct)
                ]
                n_total = len(cell)
                if n_total == 0:
                    continue
                eligible = cell[cell["adopt_eligible"]]
                n_elig = len(eligible)
                adopted = eligible[eligible["adopted"]]
                n_adopted = len(adopted)
                # bootstrap CI on the conditional rate via the eligible vector
                if n_elig > 0:
                    elig_vec = eligible["adopted"].astype(float).to_numpy()
                    rate = elig_vec.mean()
                    ci_lo, ci_hi = _bootstrap_rate(elig_vec)
                else:
                    rate, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
                # df_cond and case counts for the CSV
                rows.append({
                    "dataset": dataset,
                    "stratum": stratum,
                    "stratum_range": STRATUM_LABEL[stratum],
                    "stratum_midpoint": STRATUM_MIDPOINT[stratum],
                    "base": base_label,
                    "n_total": n_total,
                    "case1": int((cell["case_id"] == 1).sum()),
                    "case2": int((cell["case_id"] == 2).sum()),
                    "case3": int((cell["case_id"] == 3).sum()),
                    "case4": int((cell["case_id"] == 4).sum()),
                    "n_eligible": n_elig,
                    "n_adopted": n_adopted,
                    "adopt_cond": rate,
                    "adopt_cond_ci_lo": ci_lo,
                    "adopt_cond_ci_hi": ci_hi,
                    "adopt_uncond": cell["adopted"].mean() if n_total > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


def plot_per_dataset(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        for base_label, color, marker in [("wrong", "C3", "o"), ("correct", "C0", "s")]:
            cell = summary[(summary["dataset"] == dataset) & (summary["base"] == base_label)].sort_values("stratum_midpoint")
            x = cell["stratum_midpoint"].to_numpy()
            y = cell["adopt_cond"].to_numpy()
            lo = cell["adopt_cond_ci_lo"].to_numpy()
            hi = cell["adopt_cond_ci_hi"].to_numpy()
            ax.plot(x, y, marker + "-", color=color, label=f"base={base_label}")
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_xscale("symlog", linthresh=2)
        midpoints = list(STRATUM_MIDPOINT.values())
        ax.set_xticks(midpoints)
        ax.set_xticklabels(STRATUM_ORDER)
        ax.set_xlabel("Anchor distance stratum (|a - GT|)")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("paired adoption rate (95% CI)")
    fig.suptitle("E5b - paired anchor adoption vs distance, stratified by base correctness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_wrong_only(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for dataset, color, marker in [("VQAv2", "C0", "o"), ("TallyQA", "C1", "s")]:
        cell = summary[(summary["dataset"] == dataset) & (summary["base"] == "wrong")].sort_values("stratum_midpoint")
        x = cell["stratum_midpoint"].to_numpy()
        y = cell["adopt_cond"].to_numpy()
        lo = cell["adopt_cond_ci_lo"].to_numpy()
        hi = cell["adopt_cond_ci_hi"].to_numpy()
        ax.plot(x, y, marker + "-", color=color, label=dataset)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xscale("symlog", linthresh=2)
    midpoints = list(STRATUM_MIDPOINT.values())
    ax.set_xticks(midpoints)
    ax.set_xticklabels(STRATUM_ORDER)
    ax.set_xlabel("Anchor distance stratum (|a - GT|)")
    ax.set_ylabel("paired adoption rate (95% CI)")
    ax.set_title("E5b - cross-dataset adoption, wrong-base subset only")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(model: str = "llava-next-interleaved-7b") -> dict:
    records = _load_records(model)
    summary = per_cell_summary(records)

    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5b_per_stratum.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_per_dataset(summary, fig_dir / "E5b_adopt_cond_curve.png")
    plot_cross_dataset_wrong_only(summary, fig_dir / "E5b_adopt_cond_overlay.png")

    return {
        "summary": summary,
        "n_records": len(records),
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "figures": [
            "docs/figures/E5b_adopt_cond_curve.png",
            "docs/figures/E5b_adopt_cond_overlay.png",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-next-interleaved-7b")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run(model=args.model)
    pd.set_option("display.float_format", "{:0.4f}".format)
    print(out["summary"].to_string(index=False))
    print(f"\nwrote {out['out_csv']}")
    for f in out["figures"]:
        print(f"wrote {f}")
