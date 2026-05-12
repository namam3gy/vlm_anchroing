"""Aggregate the judge anchoring pilot results.

Reads `outputs/judge_pilot/<judge_id>/<timestamp>/predictions.jsonl` files,
joins per-judge per-sample scores into wide format, computes paired-bootstrap
CIs for `(a - m)` Δmean and ΔP(score = 1), writes canonical CSVs and a
2-panel forest figure.

Usage:
    uv run python scripts/analyze_judge_pilot.py --config configs/judge_pilot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from vlm_anchor.judge_pilot_data import paired_bootstrap_ci


def _largest_predictions(judge_root: Path) -> Path:
    """Pick the predictions.jsonl with the most rows (canonical run).

    Memory `feedback_smoke_run_pollution`: smoke runs accumulate alongside
    canonical full runs; mtime / alphabetical selection has shadowed canonical
    runs in past Phase A work. Sort by row count descending, tie-break by
    mtime descending.
    """
    runs = [p for p in judge_root.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists()]
    if not runs:
        raise FileNotFoundError(f"No predictions under {judge_root}")
    def _key(p: Path) -> tuple[int, float]:
        n_rows = sum(1 for _ in (p / "predictions.jsonl").open())
        return (n_rows, p.stat().st_mtime)
    runs.sort(key=_key, reverse=True)
    chosen = runs[0] / "predictions.jsonl"
    n_rows = sum(1 for _ in chosen.open())
    print(f"  selected {chosen} ({n_rows} rows)")
    return chosen


def _load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-out", type=Path, default=Path("docs/insights/_data/judge_pilot_per_sample.csv"))
    parser.add_argument("--ci-out", type=Path, default=Path("docs/insights/_data/judge_pilot_paired_ci.csv"))
    parser.add_argument("--figure-out", type=Path, default=Path("docs/figures/judge_pilot_paired_delta.png"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_root = Path(cfg["output"]["root"])
    boot_cfg = cfg["bootstrap"]
    anchor_value = int(cfg["anchor"]["anchor_value"])

    per_sample_frames: list[pd.DataFrame] = []
    ci_rows: list[dict] = []

    for judge_cfg in cfg["judges"]:
        judge_id = judge_cfg["id"]
        judge_root = out_root / judge_id
        if not judge_root.exists():
            print(f"[skip] no run dir for {judge_id}")
            continue
        preds_path = _largest_predictions(judge_root)
        df = _load_predictions(preds_path)
        wide = df.pivot_table(index=["judge_id", "sample_id"], columns="arm", values="score").reset_index()
        wide.columns.name = None
        for arm_col in ("b", "a", "m"):
            if arm_col not in wide.columns:
                wide[arm_col] = np.nan
        per_sample_frames.append(wide)

        b = wide["b"].to_numpy(dtype=float)
        a = wide["a"].to_numpy(dtype=float)
        m = wide["m"].to_numpy(dtype=float)

        # Headline: (a - m) on score and on P(score == anchor_value)
        for label, x, y in [
            ("delta_mean_a_minus_m", m, a),
            (f"delta_p_score_eq_{anchor_value}_a_minus_m",
             (m == anchor_value).astype(float), (a == anchor_value).astype(float)),
            ("delta_mean_a_minus_b", b, a),  # anchor + extra-image effect (vs no extra image)
            ("delta_mean_m_minus_b", b, m),  # extra-image-only effect
            (f"delta_p_score_eq_{anchor_value}_a_minus_b",
             (b == anchor_value).astype(float), (a == anchor_value).astype(float)),
            (f"delta_p_score_eq_{anchor_value}_m_minus_b",
             (b == anchor_value).astype(float), (m == anchor_value).astype(float)),
        ]:
            ci = paired_bootstrap_ci(
                x=x, y=y,
                n_resamples=int(boot_cfg["n_resamples"]),
                alpha=float(boot_cfg["ci_alpha"]),
                rng_seed=int(boot_cfg["rng_seed"]),
            )
            ci_rows.append({
                "judge_id": judge_id,
                "contrast": label,
                "point": ci.point,
                "lo": ci.lo,
                "hi": ci.hi,
                "n_pairs": ci.n_pairs,
            })

    args.data_out.parent.mkdir(parents=True, exist_ok=True)
    args.ci_out.parent.mkdir(parents=True, exist_ok=True)
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)

    if not per_sample_frames:
        raise SystemExit(
            "No judge predictions found under "
            f"{out_root}. Run `scripts/run_judge_pilot.py` first to produce "
            "predictions.jsonl files for the judges configured in "
            f"{args.config}."
        )
    per_sample_df = pd.concat(per_sample_frames, ignore_index=True)
    per_sample_df.to_csv(args.data_out, index=False)
    print(f"Wrote per-sample CSV: {args.data_out}")

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(args.ci_out, index=False)
    print(f"Wrote paired-CI CSV: {args.ci_out}")
    print(ci_df.to_string(index=False))

    # 2-panel forest: Δmean (left), ΔP(score == anchor) (right), one row per judge
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.6 * len(cfg["judges"]) + 1.5), sharey=True)
    p_label = f"delta_p_score_eq_{anchor_value}_a_minus_m"
    headline_contrasts = ["delta_mean_a_minus_m", p_label]
    headline = ci_df[ci_df["contrast"].isin(headline_contrasts)].copy()
    for ax, contrast, title in [
        (axes[0], "delta_mean_a_minus_m", "Δ mean score (a − m)"),
        (axes[1], p_label, f"Δ P(score = {anchor_value}) (a − m)"),
    ]:
        sub = headline[headline["contrast"] == contrast].reset_index(drop=True)
        ys = np.arange(len(sub))
        ax.errorbar(
            sub["point"], ys,
            xerr=[sub["point"] - sub["lo"], sub["hi"] - sub["point"]],
            fmt="o", color="black", ecolor="gray", capsize=3,
        )
        ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--")
        ax.set_yticks(ys, sub["judge_id"])
        ax.set_title(title)
    fig.suptitle(f"Judge anchoring pilot — paired (a − m), 95% CI, anchor={anchor_value}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.figure_out, dpi=160, bbox_inches="tight")
    print(f"Wrote figure: {args.figure_out}")


if __name__ == "__main__":
    main()
