"""E5d — MathVista distance-validation analysis (relative-cutoff scheme).

Loads the latest stratified run under
outputs/experiment_e5d_mathvista_validation/<model>/, computes per-stratum
paired adoption rate (case 4 excluded from denominator), stratifies by
base-prediction correctness, applies the C1/C2/C3 acceptance criteria
from `docs/insights/E5_anchor_distance_judgment.md` step 4, and writes:

- docs/insights/_data/E5d_mathvista_per_stratum.csv  — per-(stratum, base) table
- docs/figures/E5d_mathvista_decay.png               — decay curve (correct/wrong)

Acceptance criteria (wrong-base subset only):
- C1 (monotonic decay): adopt_cond decreases from S1 to S5 (allow one
      inversion within bootstrap CI).
- C2 (effect size): S1 adopt_cond ≥ 0.05.
- C3 (noise floor):  S4 OR S5 adopt_cond ≤ 0.01.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = "experiment_e5d_mathvista_validation"
STRATUM_ORDER = ["S1", "S2", "S3", "S4", "S5"]
# Strata are GT-relative, so a single midpoint per S-id is approximate. We use
# log-spaced positions for plotting only; the underlying S-id labels carry the
# semantics (per-question fractional bounds: 10%, 30%, 100%, 300%, ∞).
STRATUM_PLOT_POS = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
STRATUM_PLOT_LABEL = {
    "S1": "S1\n(≤10%·gt)",
    "S2": "S2\n(≤30%·gt)",
    "S3": "S3\n(≤100%·gt)",
    "S4": "S4\n(≤300%·gt)",
    "S5": "S5\n(>300%·gt)",
}
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _normalize_int_str(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    return s if s.lstrip("-").lstrip("+").isdigit() else None


def _latest_run_dir(model: str) -> Path:
    base = PROJECT_ROOT / "outputs" / EXPERIMENT_DIR / model
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run dirs under {base}")
    return runs[-1]


def _load_records(model: str) -> pd.DataFrame:
    run_dir = _latest_run_dir(model)
    recs = [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]
    df = pd.DataFrame(recs)
    df["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
    return df


def _classify(records: pd.DataFrame) -> pd.DataFrame:
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
    df = _classify(records)
    rows = []
    for stratum in STRATUM_ORDER:
        for base_correct, base_label in [(True, "correct"), (False, "wrong")]:
            cell = df[
                (df["anchor_stratum_id"] == stratum)
                & (df["base_correct"] == base_correct)
            ]
            n_total = len(cell)
            if n_total == 0:
                rows.append({
                    "stratum": stratum,
                    "base": base_label,
                    "n_total": 0, "case1": 0, "case2": 0, "case3": 0, "case4": 0,
                    "n_eligible": 0, "n_adopted": 0,
                    "adopt_cond": float("nan"), "adopt_cond_ci_lo": float("nan"),
                    "adopt_cond_ci_hi": float("nan"), "adopt_uncond": float("nan"),
                })
                continue
            eligible = cell[cell["adopt_eligible"]]
            n_elig = len(eligible)
            adopted = eligible[eligible["adopted"]]
            n_adopted = len(adopted)
            if n_elig > 0:
                elig_vec = eligible["adopted"].astype(float).to_numpy()
                rate = elig_vec.mean()
                ci_lo, ci_hi = _bootstrap_rate(elig_vec)
            else:
                rate, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
            rows.append({
                "stratum": stratum,
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


def plot_decay(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for base_label, color, marker in [("wrong", "C3", "o"), ("correct", "C0", "s")]:
        cell = summary[summary["base"] == base_label].copy()
        cell["x"] = cell["stratum"].map(STRATUM_PLOT_POS)
        cell = cell.sort_values("x")
        x = cell["x"].to_numpy()
        y = cell["adopt_cond"].to_numpy()
        lo = cell["adopt_cond_ci_lo"].to_numpy()
        hi = cell["adopt_cond_ci_hi"].to_numpy()
        ax.plot(x, y, marker + "-", color=color, label=f"base={base_label}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.set_xticks(list(STRATUM_PLOT_POS.values()))
    ax.set_xticklabels([STRATUM_PLOT_LABEL[s] for s in STRATUM_ORDER])
    ax.set_xlabel("Anchor distance stratum (relative-cutoff scheme)")
    ax.set_ylabel("paired adoption rate (95% CI)")
    ax.set_title("E5d — MathVista distance validation (n=200, llava-interleave-7b)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def apply_acceptance_criteria(summary: pd.DataFrame) -> dict:
    """Apply C1, C2, C3 to the wrong-base subset."""
    wrong = summary[summary["base"] == "wrong"].set_index("stratum")
    wrong = wrong.reindex(STRATUM_ORDER)

    rates = wrong["adopt_cond"].to_dict()
    cis_lo = wrong["adopt_cond_ci_lo"].to_dict()
    cis_hi = wrong["adopt_cond_ci_hi"].to_dict()

    # C1: monotonic decay across S1..S5, allow one inversion within CI overlap
    ordered = [rates[s] for s in STRATUM_ORDER]
    inversions = []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        if pd.isna(a) or pd.isna(b):
            continue
        if b > a:
            # Inversion exists. Allowed if CIs overlap (within noise).
            ci_overlap = (cis_hi[STRATUM_ORDER[i]] >= cis_lo[STRATUM_ORDER[i + 1]] and
                          cis_hi[STRATUM_ORDER[i + 1]] >= cis_lo[STRATUM_ORDER[i]])
            inversions.append({
                "from": STRATUM_ORDER[i], "to": STRATUM_ORDER[i + 1],
                "from_rate": a, "to_rate": b, "ci_overlap": ci_overlap,
            })
    hard_inversions = [inv for inv in inversions if not inv["ci_overlap"]]
    c1_pass = len(hard_inversions) == 0 and len(inversions) <= 1

    # C2: S1 effect size
    s1 = rates.get("S1", float("nan"))
    c2_pass = (not pd.isna(s1)) and s1 >= 0.05

    # C3: S4 or S5 ≤ 0.01
    s4 = rates.get("S4", float("nan"))
    s5 = rates.get("S5", float("nan"))
    s4_ok = (not pd.isna(s4)) and s4 <= 0.01
    s5_ok = (not pd.isna(s5)) and s5 <= 0.01
    c3_pass = s4_ok or s5_ok

    return {
        "C1": {
            "pass": c1_pass, "rates": rates,
            "inversions": inversions, "hard_inversions": hard_inversions,
        },
        "C2": {"pass": c2_pass, "S1": s1, "threshold": 0.05},
        "C3": {"pass": c3_pass, "S4": s4, "S5": s5, "threshold": 0.01,
               "S4_ok": s4_ok, "S5_ok": s5_ok},
    }


def run(model: str = "llava-next-interleaved-7b") -> dict:
    records = _load_records(model)
    summary = per_cell_summary(records)

    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5d_mathvista_per_stratum.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / "E5d_mathvista_decay.png"
    plot_decay(summary, out_png)

    verdicts = apply_acceptance_criteria(summary)

    return {
        "summary": summary,
        "n_records": len(records),
        "verdicts": verdicts,
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "out_png": str(out_png.relative_to(PROJECT_ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-next-interleaved-7b")
    return parser.parse_args()


def _print_verdicts(v: dict) -> None:
    print("\n=== Acceptance criteria (wrong-base subset) ===")
    c1 = v["C1"]
    print(f"C1 (monotonic decay): {'PASS' if c1['pass'] else 'FAIL'}")
    rate_strs = [(s, f"{c1['rates'][s]:.4f}" if not pd.isna(c1['rates'][s]) else "NaN") for s in STRATUM_ORDER]
    print(f"   rates: {rate_strs}")
    if c1["inversions"]:
        print(f"   inversions: {c1['inversions']}")
        print(f"   hard (no CI overlap): {c1['hard_inversions']}")
    c2 = v["C2"]
    print(f"C2 (S1 ≥ {c2['threshold']}): {'PASS' if c2['pass'] else 'FAIL'} — S1={c2['S1']:.4f}")
    c3 = v["C3"]
    print(f"C3 (S4 OR S5 ≤ {c3['threshold']}): {'PASS' if c3['pass'] else 'FAIL'} — S4={c3['S4']:.4f} S5={c3['S5']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    out = run(model=args.model)
    pd.set_option("display.float_format", "{:0.4f}".format)
    print(out["summary"].to_string(index=False))
    _print_verdicts(out["verdicts"])
    print(f"\nwrote {out['out_csv']}")
    print(f"wrote {out['out_png']}")
