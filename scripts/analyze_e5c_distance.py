"""E5c — anchor + masked + neutral analysis with full metric set.

For each (dataset, base, condition_type, stratum) cell, computes:
- adopt_cond   = case2 / (case1 + case2 + case3); paired conditional adoption (M1)
                 — undefined for neutral (anchor_value is None) → returns NaN
- df_uncond    = mean(anchor_direction_followed) over all records in the cell
- df_cond      = mean(anchor_direction_followed) over records where
                 anchor != gt AND not case 4 (case 4 = base=a AND pred=a)
- acc_drop     = baseline_acc - cell_acc, where baseline_acc is target_only's
                 standard_vqa_accuracy on the SAME base subset
- n            = total records in the cell

Stratification: 2 datasets × 3 bases × 3 condition_types × (5 strata for
anchor/masked, 1 'all' bucket for neutral and target_only) = ~40 cells
plus the target_only and neutral cells per (dataset, base).

Reads from outputs/experiment_e5c_*/llava-next-interleaved-7b/<latest>/.
Writes:
  docs/insights/_data/E5c_per_cell.csv
  docs/figures/E5c_anchor_vs_masked_adopt.png
  docs/figures/E5c_anchor_vs_masked_df.png
  docs/figures/E5c_acc_drop_3way.png
  docs/figures/E5c_correct_vs_wrong_adopt.png
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
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _norm(s):
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


def _load(model: str) -> pd.DataFrame:
    frames = []
    for label, exp_dir in DATASETS.items():
        run_dir = _latest_run_dir(exp_dir, model)
        recs = [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]
        df = pd.DataFrame(recs)
        df["dataset"] = label
        df["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm_pred"]   = df["prediction"].apply(_norm)
    df["norm_anchor"] = df["anchor_value"].apply(_norm)
    df["norm_gt"]     = df["ground_truth"].apply(_norm)

    base_pred_map = (
        df[df["condition"] == "target_only"]
        .set_index(["dataset", "sample_instance_id"])["norm_pred"]
        .to_dict()
    )
    df["base_pred"] = df.apply(
        lambda r: base_pred_map.get((r["dataset"], r["sample_instance_id"])),
        axis=1,
    )
    df["base_correct"] = df["base_pred"] == df["norm_gt"]

    def case_id(row):
        if row["irrelevant_type"] in ("none", "neutral", None):
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

    def cond_type(t):
        return {"none": "baseline", "number": "anchor", "number_masked": "masked", "neutral": "neutral"}.get(t, t)
    df["condition_type"] = df["irrelevant_type"].apply(cond_type)
    return df


def _bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _filter_base(df: pd.DataFrame, base: str) -> pd.DataFrame:
    if base == "all":
        return df
    if base == "correct":
        return df[df["base_correct"] == True]
    if base == "wrong":
        return df[df["base_correct"] == False]
    raise ValueError(base)


def per_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = _classify(df)
    rows = []
    for dataset in df["dataset"].unique():
        for base in ["all", "correct", "wrong"]:
            ds_base = _filter_base(df[df["dataset"] == dataset], base)
            # baseline acc on same base subset
            baseline = ds_base[ds_base["condition_type"] == "baseline"]
            baseline_acc = float(baseline["standard_vqa_accuracy"].mean()) if len(baseline) else float("nan")
            # cells: (condition_type, stratum)
            specs = (
                [("anchor", s) for s in STRATUM_ORDER]
                + [("masked", s) for s in STRATUM_ORDER]
                + [("neutral", "all")]
                + [("baseline", "all")]
            )
            for ct, stratum in specs:
                if ct in ("anchor", "masked"):
                    cell = ds_base[(ds_base["condition_type"] == ct) & (ds_base["anchor_stratum_id"] == stratum)]
                else:
                    cell = ds_base[ds_base["condition_type"] == ct]
                n_total = len(cell)
                if n_total == 0:
                    continue
                # case counts (only meaningful for anchor/masked; for neutral/baseline they'll be 0)
                case1 = int((cell["case_id"] == 1).sum())
                case2 = int((cell["case_id"] == 2).sum())
                case3 = int((cell["case_id"] == 3).sum())
                case4 = int((cell["case_id"] == 4).sum())
                eligible = case1 + case2 + case3
                # adopt_cond — undefined for neutral/baseline (no anchor)
                if ct in ("anchor", "masked") and eligible > 0:
                    adopt_cond = case2 / eligible
                    elig_vec = np.array([1.0]*case2 + [0.0]*(case1+case3))
                    ci_lo, ci_hi = _bootstrap_ci(elig_vec)
                else:
                    adopt_cond = float("nan"); ci_lo = float("nan"); ci_hi = float("nan")
                # df metrics
                df_uncond = float(cell["anchor_direction_followed"].astype(float).mean()) if n_total else float("nan")
                # df_cond: anchor != gt AND not case 4
                if ct in ("anchor", "masked"):
                    elig_df = cell[(cell["norm_anchor"] != cell["norm_gt"]) & (cell["case_id"] != 4)]
                    df_cond = float(elig_df["anchor_direction_followed"].astype(float).mean()) if len(elig_df) else float("nan")
                else:
                    df_cond = float("nan")
                cell_acc = float(cell["standard_vqa_accuracy"].mean())
                acc_drop = baseline_acc - cell_acc
                rows.append({
                    "dataset": dataset,
                    "base": base,
                    "condition_type": ct,
                    "stratum": stratum,
                    "stratum_midpoint": STRATUM_MIDPOINT.get(stratum, float("nan")),
                    "n": n_total,
                    "case1": case1, "case2": case2, "case3": case3, "case4": case4,
                    "n_eligible": eligible,
                    "n_adopted": case2,
                    "adopt_cond": adopt_cond,
                    "adopt_cond_ci_lo": ci_lo, "adopt_cond_ci_hi": ci_hi,
                    "df_uncond": df_uncond,
                    "df_cond": df_cond,
                    "acc": cell_acc,
                    "acc_drop": acc_drop,
                    "baseline_acc": baseline_acc,
                })
    return pd.DataFrame(rows)


def plot_anchor_vs_masked_adopt(summary: pd.DataFrame, out_path: Path) -> None:
    """One panel per dataset; lines for base=correct vs base=wrong; anchor vs masked stratified."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        for ct, ls in [("anchor", "-"), ("masked", "--")]:
            for base, color in [("wrong", "C3"), ("correct", "C0")]:
                cell = summary[(summary["dataset"] == dataset)
                              & (summary["base"] == base)
                              & (summary["condition_type"] == ct)
                              & (summary["stratum"].isin(STRATUM_ORDER))].sort_values("stratum_midpoint")
                if cell.empty: continue
                x = cell["stratum_midpoint"].to_numpy()
                y = cell["adopt_cond"].to_numpy()
                ax.plot(x, y, ls, color=color, marker="o" if ct == "anchor" else "s",
                        label=f"{ct} | base={base}")
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks(list(STRATUM_MIDPOINT.values()))
        ax.set_xticklabels(STRATUM_ORDER)
        ax.set_xlabel("|a - GT|")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("paired adopt_cond")
    fig.suptitle("E5c - anchor vs masked x base correctness, paired conditional adoption")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_anchor_vs_masked_df(summary: pd.DataFrame, out_path: Path) -> None:
    """Show df_cond — anchor and masked are nearly identical on wrong-base; this is the key quirk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        for ct, ls in [("anchor", "-"), ("masked", "--")]:
            for base, color in [("wrong", "C3"), ("correct", "C0")]:
                cell = summary[(summary["dataset"] == dataset)
                              & (summary["base"] == base)
                              & (summary["condition_type"] == ct)
                              & (summary["stratum"].isin(STRATUM_ORDER))].sort_values("stratum_midpoint")
                if cell.empty: continue
                x = cell["stratum_midpoint"].to_numpy()
                y = cell["df_cond"].to_numpy()
                ax.plot(x, y, ls, color=color, marker="o" if ct == "anchor" else "s",
                        label=f"{ct} | base={base}")
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks(list(STRATUM_MIDPOINT.values()))
        ax.set_xticklabels(STRATUM_ORDER)
        ax.set_xlabel("|a - GT|")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("df_cond (anchor != gt, not case 4)")
    fig.suptitle("E5c - direction_follow_cond x base correctness x anchor-vs-masked")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acc_drop_3way(summary: pd.DataFrame, out_path: Path) -> None:
    """One panel per (dataset, base). Bars: anchor S1..S5, masked S1..S5, neutral 'all'.
       Goal: show masked acc_drop ≈ neutral acc_drop, anchor > both."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for r, dataset in enumerate(["VQAv2", "TallyQA"]):
        for c, base in enumerate(["all", "correct", "wrong"]):
            ax = axes[r, c]
            sub = summary[(summary["dataset"] == dataset) & (summary["base"] == base)]
            xs, ys, colors = [], [], []
            for ct in ["anchor", "masked"]:
                for s in STRATUM_ORDER:
                    cell = sub[(sub["condition_type"] == ct) & (sub["stratum"] == s)]
                    if cell.empty: continue
                    xs.append(f"{ct[:1]}.{s}")
                    ys.append(float(cell["acc_drop"].iloc[0]))
                    colors.append("C3" if ct == "anchor" else "C2")
            neutral = sub[(sub["condition_type"] == "neutral") & (sub["stratum"] == "all")]
            if not neutral.empty:
                xs.append("neutral")
                ys.append(float(neutral["acc_drop"].iloc[0]))
                colors.append("C7")
            ax.bar(xs, ys, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{dataset} | base={base}")
            ax.set_xticklabels(xs, rotation=60, fontsize=8)
            ax.grid(alpha=0.3, axis="y")
        axes[r, 0].set_ylabel("acc_drop (vs target_only on same base)")
    fig.suptitle("E5c - accuracy drop across conditions: anchor vs masked vs neutral")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correct_vs_wrong_adopt(summary: pd.DataFrame, out_path: Path) -> None:
    """E5b-style figure but on E5c data: confirms uncertainty gate still holds when
       both anchor and masked arms are present."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        for base, color, marker in [("wrong", "C3", "o"), ("correct", "C0", "s")]:
            cell = summary[(summary["dataset"] == dataset)
                          & (summary["base"] == base)
                          & (summary["condition_type"] == "anchor")
                          & (summary["stratum"].isin(STRATUM_ORDER))].sort_values("stratum_midpoint")
            x = cell["stratum_midpoint"].to_numpy()
            y = cell["adopt_cond"].to_numpy()
            lo = cell["adopt_cond_ci_lo"].to_numpy()
            hi = cell["adopt_cond_ci_hi"].to_numpy()
            ax.plot(x, y, marker + "-", color=color, label=f"base={base}")
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks(list(STRATUM_MIDPOINT.values()))
        ax.set_xticklabels(STRATUM_ORDER)
        ax.set_xlabel("|a - GT|")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("adopt_cond (anchor only)")
    fig.suptitle("E5c - uncertainty gate: anchor adopt_cond x base correctness (cf. E5b)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


DEFAULT_MODEL = "llava-next-interleaved-7b"


def run(models: list[str] | str = DEFAULT_MODEL) -> dict:
    if isinstance(models, str):
        models = [models]
    summaries = []
    figures = []
    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_records_total = 0
    for model in models:
        df = _load(model)
        summary = per_cell_summary(df)
        summary.insert(0, "model", model)
        summaries.append(summary)
        n_records_total += len(df)
        suffix = "" if model == DEFAULT_MODEL else f"_{model}"
        for kind, plotter in [
            ("anchor_vs_masked_adopt", plot_anchor_vs_masked_adopt),
            ("anchor_vs_masked_df", plot_anchor_vs_masked_df),
            ("acc_drop_3way", plot_acc_drop_3way),
            ("correct_vs_wrong_adopt", plot_correct_vs_wrong_adopt),
        ]:
            path = fig_dir / f"E5c_{kind}{suffix}.png"
            plotter(summary, path)
            figures.append(str(path.relative_to(PROJECT_ROOT)))
    merged = pd.concat(summaries, ignore_index=True)
    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5c_per_cell.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return {
        "summary": merged,
        "n_records": n_records_total,
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "figures": figures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="One or more model names under outputs/experiment_e5c_*. The default model "
        "writes to E5c_<kind>.png; non-default models write to E5c_<kind>_<model>.png.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run(models=args.models)
    pd.set_option("display.float_format", "{:0.4f}".format)
    print(out["summary"].to_string(index=False))
    print(f"\nwrote {out['out_csv']}")
    for f in out["figures"]:
        print(f"wrote {f}")
