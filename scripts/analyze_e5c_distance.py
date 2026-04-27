"""E5c — anchor + masked-anchor + neutral analysis with paired conditional adoption.

Loads the latest stratified runs under
outputs/experiment_e5c_vqa/<model>/ and
outputs/experiment_e5c_tally/<model>/, and computes paired adoption
rates (case 4 excluded from denominator) stratified by:

- dataset (VQAv2 / TallyQA)
- stratum (S1..S5; "all" for neutral cells)
- base_correct (correct / wrong from same sample's target_only)
- condition_type (anchor / masked / neutral) derived from `irrelevant_type`

Writes:

- docs/insights/_data/E5c_per_cell.csv               — full per-(dataset, stratum, base, condition_type) table
- docs/figures/E5c_anchor_vs_masked.png              — wrong-base, anchor vs masked across S1..S5 (per dataset)
- docs/figures/E5c_three_way_comparison.png          — wrong-base, anchor vs masked + neutral horizontal reference
- docs/figures/E5c_correct_vs_wrong.png              — anchor adoption: correct vs wrong base (per dataset)

Primary metric is `adopt_cond`. The headline reading: digit-pixel
causality holds if anchor >> masked, and generic 2-image distraction
holds if masked > neutral. Both gated on wrong-base.
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
    "VQAv2":   "experiment_e5c_vqa",
    "TallyQA": "experiment_e5c_tally",
}
STRATUM_ORDER = ["S1", "S2", "S3", "S4", "S5"]
STRATUM_MIDPOINT = {"S1": 0.5, "S2": 3.5, "S3": 18.0, "S4": 165.0, "S5": 650.0}
STRATUM_LABEL = {"S1": "[0,1]", "S2": "[2,5]", "S3": "[6,30]", "S4": "[31,300]", "S5": "[301,inf)"}
IRRELEVANT_TYPE_TO_CONDITION_TYPE = {
    "number": "anchor",
    "number_masked": "masked",
    "neutral": "neutral",
}
CONDITION_TYPE_ORDER = ["anchor", "masked", "neutral"]
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
    case_id (1/2/3/4 or NaN), adopt_eligible (True for cases 1/2/3),
    condition_type (anchor / masked / neutral / None for target_only)."""
    df = records.copy()
    df["norm_pred"]   = df["prediction"].apply(_normalize_int_str)
    df["norm_anchor"] = df["anchor_value"].apply(_normalize_int_str)
    df["norm_gt"]     = df["ground_truth"].apply(_normalize_int_str)

    base_pred_map = (
        df[df["condition"] == "target_only"]
        .set_index(["dataset", "sample_instance_id"])["norm_pred"]
        .to_dict()
    )
    df["base_pred"] = list(zip(df["dataset"], df["sample_instance_id"]))
    df["base_pred"] = df["base_pred"].map(base_pred_map)
    df["base_correct"] = df["base_pred"] == df["norm_gt"]

    df["condition_type"] = df["irrelevant_type"].map(IRRELEVANT_TYPE_TO_CONDITION_TYPE)

    def case_id(row):
        if row["condition"] == "target_only":
            return None
        ct = row["condition_type"]
        p, b = row["norm_pred"], row["base_pred"]
        if p is None or b is None:
            return None
        if ct == "neutral":
            # neutral has no anchor; case is defined relative to a hypothetical anchor.
            # For the paired-adoption framework neutral cells use the same logic as
            # "no anchor adoption can occur": every record is in case 1 (b!=a, p!=a)
            # if we treat the absent anchor as a sentinel that never equals base or pred.
            # This keeps the denominator semantics consistent.
            return 1
        a = row["norm_anchor"]
        if a is None:
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


def _summarize_cell(cell: pd.DataFrame) -> dict:
    n_total = len(cell)
    eligible = cell[cell["adopt_eligible"]]
    n_elig = len(eligible)
    adopted = eligible[eligible["adopted"]]
    n_adopted = len(adopted)
    if n_elig > 0:
        elig_vec = eligible["adopted"].astype(float).to_numpy()
        rate = float(elig_vec.mean())
        ci_lo, ci_hi = _bootstrap_rate(elig_vec)
    else:
        rate, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
    return {
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
        "adopt_uncond": float(cell["adopted"].mean()) if n_total > 0 else float("nan"),
    }


def per_cell_summary(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, stratum, base_correct, condition_type).

    For condition_type ∈ {anchor, masked}, stratum iterates S1..S5.
    For condition_type == neutral, stratum is a single bucket "all".
    """
    df = _classify(records)
    rows = []
    datasets = list(DATASETS.keys())
    for dataset in datasets:
        for base_correct, base_label in [(True, "correct"), (False, "wrong")]:
            # anchor + masked: per-stratum
            for ct in ["anchor", "masked"]:
                for stratum in STRATUM_ORDER:
                    cell = df[
                        (df["dataset"] == dataset)
                        & (df["condition_type"] == ct)
                        & (df["anchor_stratum_id"] == stratum)
                        & (df["base_correct"] == base_correct)
                    ]
                    if len(cell) == 0:
                        continue
                    summary = _summarize_cell(cell)
                    rows.append({
                        "dataset": dataset,
                        "stratum": stratum,
                        "stratum_range": STRATUM_LABEL[stratum],
                        "stratum_midpoint": STRATUM_MIDPOINT[stratum],
                        "base": base_label,
                        "condition_type": ct,
                        **summary,
                    })
            # neutral: single "all" bucket
            cell = df[
                (df["dataset"] == dataset)
                & (df["condition_type"] == "neutral")
                & (df["base_correct"] == base_correct)
            ]
            if len(cell) > 0:
                summary = _summarize_cell(cell)
                rows.append({
                    "dataset": dataset,
                    "stratum": "all",
                    "stratum_range": "all",
                    "stratum_midpoint": float("nan"),
                    "base": base_label,
                    "condition_type": "neutral",
                    **summary,
                })
    return pd.DataFrame(rows)


def plot_anchor_vs_masked(summary: pd.DataFrame, out_path: Path) -> None:
    """Wrong-base only. Per dataset, two lines: anchor and masked across S1..S5."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, list(DATASETS.keys())):
        for ct, color, marker in [("anchor", "C3", "o"), ("masked", "C2", "s")]:
            cell = summary[
                (summary["dataset"] == dataset)
                & (summary["base"] == "wrong")
                & (summary["condition_type"] == ct)
                & (summary["stratum"].isin(STRATUM_ORDER))
            ].sort_values("stratum_midpoint")
            x = cell["stratum_midpoint"].to_numpy()
            y = cell["adopt_cond"].to_numpy()
            lo = cell["adopt_cond_ci_lo"].to_numpy()
            hi = cell["adopt_cond_ci_hi"].to_numpy()
            ax.plot(x, y, marker + "-", color=color, label=ct)
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
    fig.suptitle("E5c - anchor vs masked-anchor (wrong-base subset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_three_way(summary: pd.DataFrame, out_path: Path) -> None:
    """Wrong-base only. Per dataset, three lines: anchor, masked across S1..S5,
    and neutral as a horizontal reference (its value at stratum=all)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, list(DATASETS.keys())):
        for ct, color, marker in [("anchor", "C3", "o"), ("masked", "C2", "s")]:
            cell = summary[
                (summary["dataset"] == dataset)
                & (summary["base"] == "wrong")
                & (summary["condition_type"] == ct)
                & (summary["stratum"].isin(STRATUM_ORDER))
            ].sort_values("stratum_midpoint")
            x = cell["stratum_midpoint"].to_numpy()
            y = cell["adopt_cond"].to_numpy()
            lo = cell["adopt_cond_ci_lo"].to_numpy()
            hi = cell["adopt_cond_ci_hi"].to_numpy()
            ax.plot(x, y, marker + "-", color=color, label=ct)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        # neutral horizontal reference
        neutral_row = summary[
            (summary["dataset"] == dataset)
            & (summary["base"] == "wrong")
            & (summary["condition_type"] == "neutral")
        ]
        if len(neutral_row) > 0:
            n_y = float(neutral_row["adopt_cond"].iloc[0])
            n_lo = float(neutral_row["adopt_cond_ci_lo"].iloc[0])
            n_hi = float(neutral_row["adopt_cond_ci_hi"].iloc[0])
            ax.axhline(n_y, color="C7", linestyle="--", label=f"neutral (all) = {n_y:.4f}")
            ax.axhspan(n_lo, n_hi, color="C7", alpha=0.10)
        ax.set_xscale("symlog", linthresh=2)
        midpoints = list(STRATUM_MIDPOINT.values())
        ax.set_xticks(midpoints)
        ax.set_xticklabels(STRATUM_ORDER)
        ax.set_xlabel("Anchor distance stratum (|a - GT|)")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("paired adoption rate (95% CI)")
    fig.suptitle("E5c - three-way: anchor vs masked vs neutral (wrong-base subset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correct_vs_wrong(summary: pd.DataFrame, out_path: Path) -> None:
    """Anchor condition only; per dataset, two lines (correct-base / wrong-base)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, list(DATASETS.keys())):
        for base_label, color, marker in [("wrong", "C3", "o"), ("correct", "C0", "s")]:
            cell = summary[
                (summary["dataset"] == dataset)
                & (summary["condition_type"] == "anchor")
                & (summary["base"] == base_label)
                & (summary["stratum"].isin(STRATUM_ORDER))
            ].sort_values("stratum_midpoint")
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
    fig.suptitle("E5c - anchor adoption: correct-base vs wrong-base (sanity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def headline_table(summary: pd.DataFrame) -> str:
    """Distilled wrong-base headline: per (dataset, stratum) anchor vs masked,
    plus a per-dataset neutral row."""
    lines = []
    lines.append("=== E5c headline (wrong-base, paired conditional) ===")
    lines.append(f"{'Dataset':<10s}{'Stratum':<10s}{'anchor':>10s}{'masked':>10s}{'neutral':>10s}")
    for dataset in DATASETS.keys():
        for stratum in STRATUM_ORDER:
            anchor_row = summary[
                (summary["dataset"] == dataset)
                & (summary["base"] == "wrong")
                & (summary["condition_type"] == "anchor")
                & (summary["stratum"] == stratum)
            ]
            masked_row = summary[
                (summary["dataset"] == dataset)
                & (summary["base"] == "wrong")
                & (summary["condition_type"] == "masked")
                & (summary["stratum"] == stratum)
            ]
            a = float(anchor_row["adopt_cond"].iloc[0]) if len(anchor_row) else float("nan")
            m = float(masked_row["adopt_cond"].iloc[0]) if len(masked_row) else float("nan")
            lines.append(f"{dataset:<10s}{stratum:<10s}{a:>10.4f}{m:>10.4f}{'n/a':>10s}")
    lines.append("")
    lines.append(f"{'neutral (across all)':<30s}{'VQAv2':>10s}{'TallyQA':>10s}")
    n_vqa = summary[
        (summary["dataset"] == "VQAv2")
        & (summary["base"] == "wrong")
        & (summary["condition_type"] == "neutral")
    ]
    n_tally = summary[
        (summary["dataset"] == "TallyQA")
        & (summary["base"] == "wrong")
        & (summary["condition_type"] == "neutral")
    ]
    n_v = float(n_vqa["adopt_cond"].iloc[0]) if len(n_vqa) else float("nan")
    n_t = float(n_tally["adopt_cond"].iloc[0]) if len(n_tally) else float("nan")
    lines.append(f"{'wrong-base':<30s}{n_v:>10.4f}{n_t:>10.4f}")
    return "\n".join(lines)


def run(model: str = "llava-next-interleaved-7b") -> dict:
    records = _load_records(model)
    summary = per_cell_summary(records)

    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5c_per_cell.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_anchor_vs_masked(summary, fig_dir / "E5c_anchor_vs_masked.png")
    plot_three_way(summary, fig_dir / "E5c_three_way_comparison.png")
    plot_correct_vs_wrong(summary, fig_dir / "E5c_correct_vs_wrong.png")

    return {
        "summary": summary,
        "n_records": len(records),
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "headline": headline_table(summary),
        "figures": [
            "docs/figures/E5c_anchor_vs_masked.png",
            "docs/figures/E5c_three_way_comparison.png",
            "docs/figures/E5c_correct_vs_wrong.png",
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
    print()
    print(out["headline"])
