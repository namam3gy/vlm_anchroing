"""§4.4 binning comparison — 4 vs 6 vs 10 bins on a high-n cell.

Picks a representative cell (PlotQA × LLaVA-OneVision Main, n_pair ≈ 4,700)
and re-aggregates df(a) per confidence bin under three binning resolutions.
Plots side-by-side with per-bin n + 95 % bootstrap CI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/20260502-132624"
OUT = ROOT / "docs/figures/paper_4_4_binning_comparison.png"

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_num(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    m = NUM_RE.search(str(x))
    return float(m.group(0)) if m else float("nan")


def load_paired() -> tuple[pd.DataFrame, pd.Series]:
    rows = [json.loads(line) for line in (RUN_DIR / "predictions.jsonl").open()]
    df = pd.DataFrame(rows)
    df["pred_num"] = df["prediction"].apply(parse_num)
    df["gt_num"] = df["ground_truth"].apply(parse_num)
    df["anchor_num"] = df["anchor_value"].apply(parse_num)

    b = df[df["condition"] == "target_only"].set_index("sample_instance_id")
    a = df[df["condition"] == "target_plus_irrelevant_number_S1"].set_index("sample_instance_id")
    common = a.index.intersection(b.index)

    out = pd.DataFrame({
        "pa": a.loc[common, "pred_num"],
        "pb": b.loc[common, "pred_num"],
        "anchor": a.loc[common, "anchor_num"],
        "gt": a.loc[common, "gt_num"],
        "conf_b": b.loc[common, "answer_span_cross_entropy"],
    })
    out = out.dropna(subset=["pa", "pb", "anchor", "conf_b"])
    return out


def df_per_bin(d: pd.DataFrame, n_bins: int, n_boot: int = 2000, seed: int = 13):
    """Equal-frequency bins on conf_b (low entropy = high confidence = bin 0)."""
    rng = np.random.default_rng(seed)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.array(d["conf_b"].quantile(qs).to_numpy(), dtype=float, copy=True)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(d["conf_b"].to_numpy(), edges[1:-1])

    out = []
    for k in range(n_bins):
        mask = bin_idx == k
        n = int(mask.sum())
        if n == 0:
            out.append((k, 0, np.nan, np.nan, np.nan))
            continue
        sub = d[mask]
        pa, pb, anc = sub["pa"].to_numpy(), sub["pb"].to_numpy(), sub["anchor"].to_numpy()
        sign_ok = ((pa - pb) * (anc - pb)) > 0
        pa_ne_pb = pa != pb
        df_pt = float((sign_ok & pa_ne_pb).sum() / n)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            df_b = float(((sign_ok[idx] & pa_ne_pb[idx]).sum()) / n)
            boot.append(df_b)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append((k, n, df_pt, lo, hi))
    return pd.DataFrame(out, columns=["bin", "n", "df", "ci_lo", "ci_hi"])


def monotonicity_score(df_values: np.ndarray) -> tuple[bool, str]:
    """Is the bin sequence monotonically increasing?"""
    diffs = np.diff(df_values)
    fully_mono = bool((diffs > 0).all())
    weakly_mono = bool((diffs >= 0).all())
    label = "fully ↑" if fully_mono else ("weakly ↑" if weakly_mono else "non-mono")
    return fully_mono, label


def main() -> None:
    d = load_paired()
    print(f"Loaded n_pair = {len(d)} from {RUN_DIR.name}")

    bin_configs = [(4, "Q1-Q4 (current paper headline)"),
                   (6, "6 bins"),
                   (8, "8 bins"),
                   (10, "10 bins")]

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6), dpi=150, sharey=True)
    for ax, (n_bins, title) in zip(axes, bin_configs):
        res = df_per_bin(d, n_bins)
        x = np.arange(n_bins)
        df_vals = res["df"].to_numpy()
        ci_lo = res["ci_lo"].to_numpy()
        ci_hi = res["ci_hi"].to_numpy()
        ns = res["n"].to_numpy()
        gap = df_vals[-1] - df_vals[0]
        _, mono_label = monotonicity_score(df_vals)

        ax.bar(x, df_vals, color="#3a78b8", edgecolor="black", linewidth=0.5,
               width=0.78, alpha=0.85)
        ax.errorbar(x, df_vals, yerr=[df_vals - ci_lo, ci_hi - df_vals],
                    fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3)
        for i, (v, n) in enumerate(zip(df_vals, ns)):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
            ax.text(i, -0.018, f"n={n}", ha="center", fontsize=7.5,
                    color="#555555")

        ax.set_xticks(x)
        if n_bins == 4:
            ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=9.5)
        else:
            ax.set_xticklabels([f"B{i+1}" for i in range(n_bins)], fontsize=8.5)
        ax.set_xlabel("confidence bin (low→high entropy = high→low confidence)",
                      fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("direction-follow rate df(a)", fontsize=10)
        ax.set_title(f"{title}\nB(last) − B(1) = {gap:+.3f}  ·  {mono_label}",
                     fontsize=10)
        ax.set_ylim(-0.04, max(ci_hi.max() * 1.15, 0.4))
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    fig.suptitle("§4.4 binning comparison — PlotQA × LLaVA-OneVision *(Main)*, "
                 "S1 anchor (n_pair ≈ 4,700, equal-frequency bins on b-arm "
                 "answer_span_cross_entropy; 95 % bootstrap CI)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
