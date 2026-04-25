"""Analyse E4 mitigation runs — sweep Pareto, full validation summary.

Reads outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl (Phase 1) and
outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl (Phase 2) for each
mid-stack-cluster model. Produces per-strength bootstrap CIs, a Pareto plot,
and a chosen-strength selection per model.

Usage:
    uv run python scripts/analyze_e4_mitigation.py --phase sweep
    uv run python scripts/analyze_e4_mitigation.py --phase full

Strength-selection rule (per design doc):
  Among strengths where
    direction_follow_rate(target_plus_irrelevant_number, s)
      <= (1 - df_drop_target) * direction_follow_rate(target_plus_irrelevant_number, 0)
    AND exact_match(target_plus_irrelevant_number, s)
      >= exact_match(target_plus_irrelevant_number, 0) - em_drop_max,
  pick the smallest |s|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E4_ROOT = PROJECT_ROOT / "outputs" / "e4_mitigation"
SUMMARY_DIR = E4_ROOT / "_summary"

PANEL_MODELS = ["llava-1.5-7b", "convllava-7b", "internvl3-8b"]


def _load_phase(model: str, phase: str) -> pd.DataFrame | None:
    sub = "sweep_n200" if phase == "sweep" else "full_n17730"
    path = E4_ROOT / model / sub / "predictions.jsonl"
    if not path.exists():
        return None
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    return pd.DataFrame(rows) if rows else None


def _to_int(x):
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None


def _build_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Per (sample_instance_id, mask_strength), join target_only baseline (strength=0)
    with target_plus_irrelevant_number under that strength."""
    base = (
        df[(df["condition"] == "target_only") & (df["mask_strength"] == 0.0)]
        [["sample_instance_id", "parsed_number", "ground_truth"]]
        .rename(columns={"parsed_number": "base_pred"})
        .drop_duplicates("sample_instance_id")
    )
    num = df[df["condition"] == "target_plus_irrelevant_number"].copy()
    num = num.rename(columns={"parsed_number": "num_pred"})
    num = num.merge(base, on="sample_instance_id", how="inner",
                    suffixes=("", "_base"))
    return num


def _metrics(triplets: pd.DataFrame) -> dict:
    base = triplets["base_pred"].map(_to_int)
    num = triplets["num_pred"].map(_to_int)
    anchor = triplets["anchor_value"].map(_to_int)
    gt = triplets["ground_truth"].map(_to_int)
    valid = base.notna() & num.notna() & anchor.notna()
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "df_num": np.nan, "adopt_num": np.nan, "em_num": np.nan,
                "mean_dist": np.nan}
    db = (base[valid] - anchor[valid]).abs()
    dn = (num[valid] - anchor[valid]).abs()
    df_num = float((dn < db).mean())
    adopt = float((num[valid] == anchor[valid]).mean())
    mean_dist = float(dn.mean())
    em_valid = num.notna() & gt.notna()
    em = float((num[em_valid] == gt[em_valid]).mean()) if em_valid.any() else np.nan
    return {"n": n, "df_num": df_num, "adopt_num": adopt, "em_num": em,
            "mean_dist": mean_dist}


def _bootstrap(triplets: pd.DataFrame, metric: str, n_boot: int = 2000,
               seed: int = 42) -> tuple[float, float]:
    base = triplets["base_pred"].map(_to_int)
    num = triplets["num_pred"].map(_to_int)
    anchor = triplets["anchor_value"].map(_to_int)
    gt = triplets["ground_truth"].map(_to_int)
    valid = base.notna() & num.notna() & anchor.notna()
    arr_base = base[valid].to_numpy()
    arr_num = num[valid].to_numpy()
    arr_anc = anchor[valid].to_numpy()
    arr_gt = gt[valid].to_numpy()
    n = len(arr_num)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "df_num":
            db = np.abs(arr_base[idx] - arr_anc[idx])
            dn = np.abs(arr_num[idx] - arr_anc[idx])
            draws[i] = (dn < db).mean()
        elif metric == "adopt_num":
            draws[i] = (arr_num[idx] == arr_anc[idx]).mean()
        elif metric == "em_num":
            num_b = arr_num[idx]
            gt_b = arr_gt[idx]
            mask_gt = np.array([v is not None for v in gt_b])
            if mask_gt.sum() == 0:
                draws[i] = np.nan
            else:
                draws[i] = (num_b[mask_gt] == gt_b[mask_gt]).mean()
        elif metric == "mean_dist":
            draws[i] = float(np.abs(arr_num[idx] - arr_anc[idx]).mean())
        else:
            raise ValueError(metric)
    valid_draws = draws[~np.isnan(draws)]
    if len(valid_draws) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(valid_draws, 2.5)),
            float(np.percentile(valid_draws, 97.5)))


def select_optimal_strength(per_strength: dict, baseline_df: float,
                            baseline_em: float, df_drop_target: float = 0.10,
                            em_drop_max: float = 0.02) -> float | None:
    """Pick smallest |strength| satisfying both target criteria. None if none qualify."""
    df_threshold = baseline_df * (1.0 - df_drop_target)
    em_threshold = baseline_em - em_drop_max
    candidates = []
    for s, m in per_strength.items():
        if s == 0.0:
            continue
        df_num = m["df_num"]
        em_num = m["em_num"]
        if np.isnan(df_num) or np.isnan(em_num):
            continue
        if df_num <= df_threshold and em_num >= em_threshold:
            candidates.append(s)
    if not candidates:
        return None
    return min(candidates, key=abs)


def _per_condition_em(df: pd.DataFrame) -> dict:
    """Per (mask_strength, condition) exact_match — used to verify the hook is
    anchor-condition-specific (target_only should be invariant, neutral should
    barely move, number should drop or stay)."""
    out: dict = {}
    for s in sorted(df["mask_strength"].unique()):
        for cond in ("target_only", "target_plus_irrelevant_neutral",
                     "target_plus_irrelevant_number"):
            sub = df[(df["mask_strength"] == s) & (df["condition"] == cond)]
            parsed = sub["parsed_number"].map(_to_int)
            gt = sub["ground_truth"].map(_to_int)
            valid = parsed.notna() & gt.notna()
            n = int(valid.sum())
            em = float((parsed[valid] == gt[valid]).mean()) if n else np.nan
            out[(float(s), cond)] = {"n": n, "em": em}
    return out


def _summarise_phase(phase: str) -> pd.DataFrame:
    rows = []
    for model in PANEL_MODELS:
        df = _load_phase(model, phase)
        if df is None:
            print(f"[{model}] skipping — no {phase} run")
            continue
        triplets = _build_triplets(df)
        em_by_cond = _per_condition_em(df)
        for s in sorted(triplets["mask_strength"].unique()):
            sub = triplets[triplets["mask_strength"] == s]
            stats = _metrics(sub)
            ci_df_lo, ci_df_hi = _bootstrap(sub, "df_num")
            ci_em_lo, ci_em_hi = _bootstrap(sub, "em_num")
            em_target_only = em_by_cond.get((float(s), "target_only"), {}).get("em", np.nan)
            em_neutral = em_by_cond.get((float(s), "target_plus_irrelevant_neutral"),
                                         {}).get("em", np.nan)
            rows.append({
                "model": model, "mask_strength": float(s), **stats,
                "df_ci_low": ci_df_lo, "df_ci_high": ci_df_hi,
                "em_ci_low": ci_em_lo, "em_ci_high": ci_em_hi,
                "em_target_only": em_target_only,
                "em_neutral": em_neutral,
            })
    return pd.DataFrame(rows)


def _pareto_plot(df_summary: pd.DataFrame, out_path: Path,
                 chosen: dict | None = None) -> None:
    fig, axes = plt.subplots(1, len(PANEL_MODELS), figsize=(5 * len(PANEL_MODELS), 4),
                             sharey=False)
    if len(PANEL_MODELS) == 1:
        axes = [axes]
    for ax, model in zip(axes, PANEL_MODELS):
        sub = df_summary[df_summary["model"] == model].sort_values("mask_strength")
        if sub.empty:
            ax.set_title(f"{model} (no data)")
            continue
        x = sub["mask_strength"]
        ax.plot(x, sub["df_num"], "o-", label="direction_follow", color="tab:red")
        ax.fill_between(x, sub["df_ci_low"], sub["df_ci_high"], color="tab:red", alpha=0.2)
        ax.set_xlabel("mask_strength")
        ax.set_ylabel("direction_follow_rate", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax.twinx()
        ax2.plot(x, sub["em_num"], "s-", label="exact_match", color="tab:blue")
        ax2.fill_between(x, sub["em_ci_low"], sub["em_ci_high"], color="tab:blue", alpha=0.2)
        ax2.set_ylabel("exact_match (target_plus_irrelevant_number)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        # Mark chosen strength with a vertical guide
        if chosen is not None and chosen.get(model) is not None:
            s_star = float(chosen[model])
            ax.axvline(s_star, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
            row = sub[sub["mask_strength"] == s_star]
            if not row.empty:
                df_at_star = float(row.iloc[0]["df_num"])
                em_at_star = float(row.iloc[0]["em_num"])
                ax.annotate(f"s*={s_star:g}\ndf={df_at_star:.3f}\nem={em_at_star:.3f}",
                            xy=(s_star, df_at_star), xytext=(8, -8),
                            textcoords="offset points", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3",
                                       fc="white", ec="black", alpha=0.7))
        ax.set_title(model)
        ax.set_xscale("symlog", linthresh=0.5)
    fig.suptitle("E4 strength sweep — direction-follow vs exact-match")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sweep", "full"), default="sweep")
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "sweep":
        df_summary = _summarise_phase("sweep")
        out_csv = SUMMARY_DIR / "sweep_pareto.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")

        chosen: dict = {}
        for model in PANEL_MODELS:
            sub = df_summary[df_summary["model"] == model]
            if sub.empty:
                chosen[model] = None
                continue
            baseline = sub[sub["mask_strength"] == 0.0]
            if baseline.empty:
                chosen[model] = None
                continue
            base_df = float(baseline["df_num"].iloc[0])
            base_em = float(baseline["em_num"].iloc[0])
            per_strength = {
                float(r.mask_strength): {"df_num": float(r.df_num), "em_num": float(r.em_num)}
                for r in sub.itertuples(index=False)
            }
            chosen[model] = select_optimal_strength(per_strength, base_df, base_em)
        out_json = SUMMARY_DIR / "chosen_strength.json"
        out_json.write_text(json.dumps(chosen, indent=2))
        print(f"[write] {out_json}")
        _pareto_plot(df_summary, SUMMARY_DIR / "sweep_pareto.png", chosen=chosen)
        print(f"[write] {SUMMARY_DIR / 'sweep_pareto.png'}")
        print("=== chosen strengths ===")
        for k, v in chosen.items():
            print(f"  {k}: {v}")
    else:
        df_summary = _summarise_phase("full")
        out_csv = SUMMARY_DIR / "full_validation.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")
        print("=== full-scale validation ===")
        print(df_summary.to_string())


if __name__ == "__main__":
    main()
