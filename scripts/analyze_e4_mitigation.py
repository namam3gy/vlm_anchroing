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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E4_ROOT = PROJECT_ROOT / "outputs" / "e4_mitigation"
SUMMARY_DIR = E4_ROOT / "_summary"

PANEL_MODELS = ["llava-1.5-7b", "convllava-7b", "internvl3-8b"]


sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vlm_anchor.utils import extract_first_number  # noqa: E402


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


def _to_int(x, rescue_text: str | None = None):
    """Robust integer parse.

    Tries: strict int(x). On failure, falls back to extract_first_number on the
    raw decoded text (`rescue_text`) — recovers InternVL3-style prose-leak cases
    where the driver's parsed_number landed on a non-numeric word ("based" etc).
    Returns None if neither path produces an integer."""
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        pass
    if rescue_text is not None:
        rescued = extract_first_number(rescue_text)
        if rescued:
            try:
                return int(rescued)
            except ValueError:
                pass
    return None


def _build_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Per (sample_instance_id, mask_strength), join target_only baseline (strength=0)
    with target_plus_irrelevant_number under that strength.

    Carries the raw `decoded` text alongside `parsed_number` on both sides so
    `_to_int` can rescue prose-leak cases (e.g. InternVL3 emits "based on…",
    parser lands on "based", rescue extracts the number from the raw text)."""
    base_cols = ["sample_instance_id", "parsed_number", "ground_truth"]
    if "decoded" in df.columns:
        base_cols = base_cols + ["decoded"]
    base = (
        df[(df["condition"] == "target_only") & (df["mask_strength"] == 0.0)]
        [base_cols]
        .rename(columns={"parsed_number": "base_pred", "decoded": "base_decoded"})
        .drop_duplicates("sample_instance_id")
    )
    num = df[df["condition"] == "target_plus_irrelevant_number"].copy()
    num = num.rename(columns={"parsed_number": "num_pred", "decoded": "num_decoded"})
    num = num.merge(base, on="sample_instance_id", how="inner",
                    suffixes=("", "_base"))
    return num


def _to_int_series(values: pd.Series, rescue: pd.Series | None = None) -> pd.Series:
    """Vectorised _to_int with optional per-row rescue text."""
    if rescue is None:
        return values.map(_to_int)
    out = []
    for v, r in zip(values, rescue):
        out.append(_to_int(v, rescue_text=r))
    return pd.Series(out, index=values.index, dtype=object)


def _metrics(triplets: pd.DataFrame) -> dict:
    base = _to_int_series(triplets["base_pred"], triplets.get("base_decoded"))
    num = _to_int_series(triplets["num_pred"], triplets.get("num_decoded"))
    anchor = _to_int_series(triplets["anchor_value"])
    gt = _to_int_series(triplets["ground_truth"])
    valid = base.notna() & num.notna() & anchor.notna()
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "df_num": np.nan, "adopt_num": np.nan, "em_num": np.nan,
                "mean_dist": np.nan}
    base_v = base[valid].astype(int)
    num_v = num[valid].astype(int)
    anc_v = anchor[valid].astype(int)
    db = (base_v - anc_v).abs()
    dn = (num_v - anc_v).abs()
    df_num = float((dn < db).mean())
    adopt = float((num_v == anc_v).mean())
    mean_dist = float(dn.mean())
    em_valid = num.notna() & gt.notna()
    em = (
        float((num[em_valid].astype(int) == gt[em_valid].astype(int)).mean())
        if em_valid.any() else np.nan
    )
    return {"n": n, "df_num": df_num, "adopt_num": adopt, "em_num": em,
            "mean_dist": mean_dist}


def _bootstrap(triplets: pd.DataFrame, metric: str, n_boot: int = 2000,
               seed: int = 42) -> tuple[float, float]:
    base = _to_int_series(triplets["base_pred"], triplets.get("base_decoded"))
    num = _to_int_series(triplets["num_pred"], triplets.get("num_decoded"))
    anchor = _to_int_series(triplets["anchor_value"])
    gt = _to_int_series(triplets["ground_truth"])
    valid = base.notna() & num.notna() & anchor.notna()
    arr_base = base[valid].astype(int).to_numpy()
    arr_num = num[valid].astype(int).to_numpy()
    arr_anc = anchor[valid].astype(int).to_numpy()
    arr_gt = gt[valid].to_numpy()  # gt may be None; cast happens after mask_gt filter
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


def _paired_anchor_damage(df: pd.DataFrame, s_star: float | None) -> dict | None:
    """Anchor-damage / partial-recovery on the *intersection* of valid samples.

    For a given model, computes em(target_only @ 0), em(num @ 0), em(num @ s*)
    on the same set of sample_instance_ids — the only fair way to compare em
    across conditions. The saturation cell (s = −10⁴) is included when
    available (Phase 1 sweep) and dropped when not (Phase 2 only runs
    {0, s*}).

    Returns None if s_star is missing or any required cell is empty."""
    if s_star is None:
        return None
    required_cells = {
        "to_0": ("target_only", 0.0),
        "num_0": ("target_plus_irrelevant_number", 0.0),
        "num_s": ("target_plus_irrelevant_number", float(s_star)),
    }
    optional_cells = {
        "num_sat": ("target_plus_irrelevant_number", -1e4),
    }
    per_cell: dict[str, dict[str, tuple[int, int]]] = {}
    for key, (cond, strength) in {**required_cells, **optional_cells}.items():
        sub = df[(df["condition"] == cond) & (df["mask_strength"] == strength)]
        if sub.empty:
            continue
        parsed = _to_int_series(sub["parsed_number"], sub.get("decoded"))
        gt = _to_int_series(sub["ground_truth"])
        bag: dict[str, tuple[int, int]] = {}
        for sid, p, g in zip(sub["sample_instance_id"], parsed, gt):
            if p is None or g is None:
                continue
            bag[str(sid)] = (int(p), int(g))
        per_cell[key] = bag
    if not all(k in per_cell and per_cell[k] for k in required_cells):
        return None
    common = set.intersection(*(set(per_cell[k]) for k in per_cell))
    if not common:
        return None
    n = len(common)
    em = {key: sum(1 for sid in common if per_cell[key][sid][0]
                   == per_cell[key][sid][1]) / n
          for key in per_cell}
    damage_pp = (em["num_0"] - em["to_0"]) * 100.0
    recover_pp_s = (em["num_s"] - em["num_0"]) * 100.0
    pct_loss_recovered_s = (
        100.0 * (em["num_s"] - em["num_0"]) / (em["to_0"] - em["num_0"])
        if em["to_0"] > em["num_0"] else None
    )
    out = {
        "n_paired": n,
        "em_target_only": round(em["to_0"], 4),
        "em_num_at_0": round(em["num_0"], 4),
        "em_num_at_s_star": round(em["num_s"], 4),
        "anchor_damage_pp": round(damage_pp, 2),
        "recovery_at_s_star_pp": round(recover_pp_s, 2),
        "fraction_of_damage_recovered_at_s_star_pct":
            round(pct_loss_recovered_s, 1) if pct_loss_recovered_s is not None else None,
    }
    if "num_sat" in em:
        recover_pp_sat = (em["num_sat"] - em["num_0"]) * 100.0
        pct_loss_recovered_sat = (
            100.0 * (em["num_sat"] - em["num_0"]) / (em["to_0"] - em["num_0"])
            if em["to_0"] > em["num_0"] else None
        )
        out.update({
            "em_num_at_saturation": round(em["num_sat"], 4),
            "recovery_at_saturation_pp": round(recover_pp_sat, 2),
            "fraction_of_damage_recovered_at_saturation_pct":
                round(pct_loss_recovered_sat, 1)
                if pct_loss_recovered_sat is not None else None,
        })
    return out


def _per_condition_em(df: pd.DataFrame) -> dict:
    """Per (mask_strength, condition) exact_match — used to verify the hook is
    anchor-condition-specific (target_only should be invariant, neutral should
    barely move, number should drop or stay)."""
    out: dict = {}
    for s in sorted(df["mask_strength"].unique()):
        for cond in ("target_only", "target_plus_irrelevant_neutral",
                     "target_plus_irrelevant_number"):
            sub = df[(df["mask_strength"] == s) & (df["condition"] == cond)]
            parsed = _to_int_series(sub["parsed_number"], sub.get("decoded"))
            gt = _to_int_series(sub["ground_truth"])
            valid = parsed.notna() & gt.notna()
            n = int(valid.sum())
            em = (
                float((parsed[valid].astype(int) == gt[valid].astype(int)).mean())
                if n else np.nan
            )
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

        # Paired anchor-damage table — fair cross-condition em comparison on
        # intersection of valid samples per model.
        pad_rows = []
        for model in PANEL_MODELS:
            sub_df = _load_phase(model, "sweep")
            if sub_df is None:
                continue
            pad = _paired_anchor_damage(sub_df, chosen.get(model))
            if pad is None:
                continue
            pad_rows.append({"model": model, "phase": "sweep",
                             "s_star": chosen.get(model), **pad})
        if pad_rows:
            pad_df = pd.DataFrame(pad_rows)
            out_pad_csv = SUMMARY_DIR / "anchor_damage_paired_sweep.csv"
            pad_df.to_csv(out_pad_csv, index=False)
            print(f"[write] {out_pad_csv}")
            print("=== paired anchor-damage (sweep) ===")
            print(pad_df.to_string(index=False))
    else:
        df_summary = _summarise_phase("full")
        out_csv = SUMMARY_DIR / "full_validation.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")

        # Side-by-side baseline (s=0) vs treated (s=s*) comparison rows.
        # Phase 2 may be partial mid-session; report completion counter so
        # partial runs are interpretable.
        try:
            chosen = json.loads((SUMMARY_DIR / "chosen_strength.json").read_text())
        except FileNotFoundError:
            chosen = {}

        compare_rows = []
        if df_summary.empty:
            print("(no Phase 2 data yet)")
            return
        for model in PANEL_MODELS:
            sub = df_summary[df_summary["model"] == model]
            if sub.empty:
                continue
            s_star = chosen.get(model)
            if s_star is None:
                continue
            base = sub[sub["mask_strength"] == 0.0]
            treat = sub[sub["mask_strength"] == float(s_star)]
            if base.empty or treat.empty:
                continue
            b = base.iloc[0]
            t = treat.iloc[0]
            # Completion counter: expected = n_questions × n_variants × n_conds
            # × n_modes (after target_only-skip). For Phase 2 sweep_n=17730
            # variants and condition layout, the canonical number is
            # 17730 × (1 + 2 + 2) = 88,650 records per model.
            df_path = E4_ROOT / model / "full_n17730" / "predictions.jsonl"
            n_records = 0
            if df_path.exists():
                with df_path.open() as fh:
                    for line in fh:
                        if line.strip():
                            n_records += 1
            expected = 88_650
            completion = (n_records / expected) if expected else 0.0
            compare_rows.append({
                "model": model,
                "s*": float(s_star),
                "n_records": n_records,
                "expected_records": expected,
                "completion_pct": round(100.0 * completion, 1),
                "df_baseline": float(b["df_num"]),
                "df_baseline_ci": [float(b["df_ci_low"]), float(b["df_ci_high"])],
                "df_treated": float(t["df_num"]),
                "df_treated_ci": [float(t["df_ci_low"]), float(t["df_ci_high"])],
                "df_drop_pp": round(100.0 * (float(t["df_num"]) - float(b["df_num"])), 2),
                "df_drop_relative_pct": round(
                    100.0 * (float(t["df_num"]) - float(b["df_num"])) / float(b["df_num"])
                    if float(b["df_num"]) else 0.0, 2),
                "em_baseline": float(b["em_num"]),
                "em_baseline_ci": [float(b["em_ci_low"]), float(b["em_ci_high"])],
                "em_treated": float(t["em_num"]),
                "em_treated_ci": [float(t["em_ci_low"]), float(t["em_ci_high"])],
                "em_delta_pp": round(100.0 * (float(t["em_num"]) - float(b["em_num"])), 2),
                "em_target_only_baseline": float(b["em_target_only"])
                    if not pd.isna(b["em_target_only"]) else None,
                "em_target_only_treated": float(t["em_target_only"])
                    if not pd.isna(t["em_target_only"]) else None,
                "em_neutral_baseline": float(b["em_neutral"])
                    if not pd.isna(b["em_neutral"]) else None,
                "em_neutral_treated": float(t["em_neutral"])
                    if not pd.isna(t["em_neutral"]) else None,
            })

        if compare_rows:
            compare_df = pd.DataFrame(compare_rows)
            out_compare_csv = SUMMARY_DIR / "full_validation_compare.csv"
            compare_df.to_csv(out_compare_csv, index=False)
            print(f"[write] {out_compare_csv}")

        # Paired anchor-damage at full scale — the headline em-side comparison.
        pad_rows = []
        for model in PANEL_MODELS:
            sub_df = _load_phase(model, "full")
            if sub_df is None:
                continue
            s_star = chosen.get(model)
            pad = _paired_anchor_damage(sub_df, s_star)
            if pad is None:
                continue
            pad_rows.append({"model": model, "phase": "full",
                             "s_star": s_star, **pad})
        if pad_rows:
            pad_df = pd.DataFrame(pad_rows)
            out_pad_csv = SUMMARY_DIR / "anchor_damage_paired_full.csv"
            pad_df.to_csv(out_pad_csv, index=False)
            print(f"[write] {out_pad_csv}")
            print("=== paired anchor-damage (full) ===")
            print(pad_df.to_string(index=False))

        print("=== full-scale validation (raw per-strength) ===")
        print(df_summary.to_string())
        if compare_rows:
            print()
            print("=== baseline vs treated comparison (Phase 2 headline) ===")
            for r in compare_rows:
                print(f"\n{r['model']} @ s*={r['s*']}, "
                      f"completed {r['n_records']:,} / {r['expected_records']:,} "
                      f"({r['completion_pct']:.1f}%)")
                print(f"  df_num: {r['df_baseline']:.4f} -> {r['df_treated']:.4f} "
                      f"(Δ {r['df_drop_pp']:+.2f} pp, {r['df_drop_relative_pct']:+.1f}% rel)")
                print(f"  em_num: {r['em_baseline']:.4f} -> {r['em_treated']:.4f} "
                      f"(Δ {r['em_delta_pp']:+.2f} pp)  "
                      f"<-- 'mitigation safe for / improves accuracy?' check")
                if r['em_target_only_baseline'] is not None:
                    print(f"  em(target_only) baseline: {r['em_target_only_baseline']:.4f}"
                          f"  (sanity — should be invariant; treated row is skipped"
                          f" by Phase 2 design)")


if __name__ == "__main__":
    main()
