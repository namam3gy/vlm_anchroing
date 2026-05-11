"""Paired-bootstrap 95% CI on the ×12.7 correct-base df(a) ratio.

Setting (paper §4.5, E5e MathVista γ-β):
    Same MathVista-testmini sid pool (n=365 sample_instances) was run through
    Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Thinking. The headline thinking-mode
    amplification ×12.7 = df(a)_correct_thinking / df(a)_correct_instruct
    = 0.266667 / 0.021008.

Estimator (matches `analyze_e5e_wrong_correct.py` / `metrics.py`):
    base_correct      = is_int_str(pb) AND is_int_str(gt) AND pb == gt
    direction_follow  = anchor_direction_followed_moved (per-row flag)
    numeric_pair      = numeric_distance_to_anchor is not None
    df(a)_correct     = Σ direction_follow / Σ numeric_pair  on subset {base_correct}

Pairing design (Option 2, sid-level resample + arm-conditional filter):
    Resample sids with replacement from the 365-sid pool. In each bootstrap
    draw, compute df(a)_correct **per arm independently** using each arm's
    own base_correct subset. Pairing is preserved at the sid level (the
    *same* resampled sid contributes to both arms), but the conditioning
    set differs by arm — matching the point-estimate definition.

Outputs CSV + JSON to `docs/insights/_data/`.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "insights" / "_data"


DEFAULT_PREDS = {
    "instruct": ROOT
    / "outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-instruct/20260428-114421/predictions.jsonl",
    "thinking": ROOT
    / "outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-thinking/20260428-114421/predictions.jsonl",
}


def is_int_str(s) -> bool:
    return bool(str(s)) and str(s).lstrip("-").isdigit() if s is not None else False


def build_per_sid_table(
    preds_paths: dict[str, Path],
    a_cond: str = "target_plus_irrelevant_number_S1",
    b_cond: str = "target_only",
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aligned per-sid arrays for both arms.

    Returns
    -------
    sids
        Ordered list of sample_instance_ids that appear in *both* arms with
        a paired b/a row each.
    base_correct
        Shape ``(n_sids, n_arms)`` of bool: is(int(pb)) AND is(int(gt)) AND pb==gt
        evaluated per arm.
    df_moved
        Shape ``(n_sids, n_arms)`` of {0,1}: per-row `anchor_direction_followed_moved`
        flag from the a-row.
    numeric_pair
        Shape ``(n_sids, n_arms)`` of bool: a-row has
        `numeric_distance_to_anchor is not None`.
    arms
        Ordered list of arm names matching the column axis.
    """
    per_arm_rows: dict[str, dict[str, dict]] = {}
    for arm, path in preds_paths.items():
        sid_to_rows: dict[str, dict] = defaultdict(dict)
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                sid = r.get("sample_instance_id")
                if sid is None:
                    continue
                sid_to_rows[sid][r.get("condition", "")] = r
        per_arm_rows[arm] = sid_to_rows

    arms = list(preds_paths.keys())
    sids = sorted(set.intersection(*(set(d.keys()) for d in per_arm_rows.values())))

    base_correct = np.zeros((len(sids), len(arms)), dtype=bool)
    df_moved = np.zeros((len(sids), len(arms)), dtype=np.int8)
    numeric_pair = np.zeros((len(sids), len(arms)), dtype=bool)

    keep_mask = np.ones(len(sids), dtype=bool)
    for j, arm in enumerate(arms):
        for i, sid in enumerate(sids):
            b = per_arm_rows[arm][sid].get(b_cond)
            a = per_arm_rows[arm][sid].get(a_cond)
            if b is None or a is None:
                keep_mask[i] = False
                continue
            pb = (b.get("prediction") or "").strip()
            gt = (b.get("ground_truth") or "").strip()
            base_correct[i, j] = is_int_str(pb) and is_int_str(gt) and pb == gt
            df_moved[i, j] = int(a.get("anchor_direction_followed_moved") or 0)
            numeric_pair[i, j] = a.get("numeric_distance_to_anchor") is not None

    sids = [s for s, k in zip(sids, keep_mask) if k]
    base_correct = base_correct[keep_mask]
    df_moved = df_moved[keep_mask]
    numeric_pair = numeric_pair[keep_mask]

    return sids, base_correct, df_moved, numeric_pair, arms


def df_correct_rate(
    indices: np.ndarray,
    base_correct: np.ndarray,
    df_moved: np.ndarray,
    numeric_pair: np.ndarray,
) -> np.ndarray:
    """Per-arm df(a)_correct rate on a sid resample.

    Vectorised over arms (columns).
    """
    keep = base_correct[indices] & numeric_pair[indices]  # (n_resample, n_arms)
    den = keep.sum(axis=0)
    num = (df_moved[indices] * keep).sum(axis=0)
    rate = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
    return rate, num, den


def run_bootstrap(
    base_correct: np.ndarray,
    df_moved: np.ndarray,
    numeric_pair: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_sids = base_correct.shape[0]
    rates = np.empty((n_resamples, base_correct.shape[1]))
    nums = np.empty((n_resamples, base_correct.shape[1]), dtype=np.int64)
    dens = np.empty((n_resamples, base_correct.shape[1]), dtype=np.int64)
    for b in range(n_resamples):
        idx = rng.integers(0, n_sids, size=n_sids)
        rate, num, den = df_correct_rate(idx, base_correct, df_moved, numeric_pair)
        rates[b] = rate
        nums[b] = num
        dens[b] = den
    return {"rates": rates, "nums": nums, "dens": dens}


def percentile_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    finite = x[np.isfinite(x)]
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-prefix",
        default="qwen3vl_x12_7_paired_ci",
        help="Filename stem under docs/insights/_data/",
    )
    args = parser.parse_args()

    sids, base_correct, df_moved, numeric_pair, arms = build_per_sid_table(DEFAULT_PREDS)
    print(f"[load] n_sids paired={len(sids)} arms={arms}")

    # Point estimate (full sample)
    full_idx = np.arange(len(sids))
    rate_full, num_full, den_full = df_correct_rate(
        full_idx, base_correct, df_moved, numeric_pair
    )
    point = {arm: (float(rate_full[j]), int(num_full[j]), int(den_full[j])) for j, arm in enumerate(arms)}
    for arm in arms:
        r, n, d = point[arm]
        print(f"[point] {arm:10s} df(a) correct = {r:.6f} ({n}/{d})")
    ratio_point = point["thinking"][0] / point["instruct"][0]
    print(f"[point] ratio = {ratio_point:.4f}")

    boot = run_bootstrap(
        base_correct, df_moved, numeric_pair,
        n_resamples=args.n_resamples, seed=args.seed,
    )
    rates = boot["rates"]
    instruct_j = arms.index("instruct")
    thinking_j = arms.index("thinking")
    r_i = rates[:, instruct_j]
    r_t = rates[:, thinking_j]

    # Ratio (linear) — only valid when instruct rate > 0
    valid = (r_i > 0) & np.isfinite(r_i) & np.isfinite(r_t)
    n_invalid = int((~valid).sum())
    ratio_b = np.where(valid, r_t / np.where(r_i == 0, np.nan, r_i), np.nan)
    log_ratio_b = np.where(valid, np.log(r_t) - np.log(np.where(r_i == 0, np.nan, r_i)), np.nan)
    # Replace inf from log(0) at thinking rate=0 with NaN
    log_ratio_b = np.where(np.isfinite(log_ratio_b), log_ratio_b, np.nan)

    ci_instruct = percentile_ci(r_i)
    ci_thinking = percentile_ci(r_t)
    ci_ratio = percentile_ci(ratio_b)
    ci_logratio = percentile_ci(log_ratio_b)

    print(f"\n[CI 95%] instruct df(a) correct: [{ci_instruct[0]:.4f}, {ci_instruct[1]:.4f}]")
    print(f"[CI 95%] thinking df(a) correct: [{ci_thinking[0]:.4f}, {ci_thinking[1]:.4f}]")
    print(f"[CI 95%] ratio (T/I)           : [{ci_ratio[0]:.3f}, {ci_ratio[1]:.3f}]   point={ratio_point:.3f}")
    print(
        f"[CI 95%] log-ratio (T/I)       : [{ci_logratio[0]:.3f}, {ci_logratio[1]:.3f}]   "
        f"exp → [{math.exp(ci_logratio[0]):.3f}, {math.exp(ci_logratio[1]):.3f}]"
    )
    print(f"[diag] invalid draws (instruct rate==0): {n_invalid}/{args.n_resamples}")

    payload = {
        "experiment": "E5e MathVista γ-β reasoning (Qwen3-VL-8B Instruct vs Thinking)",
        "cell": {"cond_class": "a", "stratum": "S1", "base_correct": True},
        "metric": "direction_follow_rate (M2 C-form)",
        "estimand": "thinking df(a) correct / instruct df(a) correct",
        "method": (
            "paired bootstrap (sid-level resample, arm-conditional base_correct filter; "
            "Option 2 in the methods note)"
        ),
        "n_sids_paired": len(sids),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "point": {
            "instruct_df_correct": point["instruct"][0],
            "instruct_num": point["instruct"][1],
            "instruct_den": point["instruct"][2],
            "thinking_df_correct": point["thinking"][0],
            "thinking_num": point["thinking"][1],
            "thinking_den": point["thinking"][2],
            "ratio_T_over_I": ratio_point,
            "log_ratio": math.log(point["thinking"][0] / point["instruct"][0]),
        },
        "ci_95": {
            "instruct_df_correct": list(ci_instruct),
            "thinking_df_correct": list(ci_thinking),
            "ratio_T_over_I": list(ci_ratio),
            "log_ratio": list(ci_logratio),
            "ratio_exp_logCI": [math.exp(ci_logratio[0]), math.exp(ci_logratio[1])],
        },
        "diagnostics": {
            "invalid_ratio_draws_instruct_zero": n_invalid,
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / f"{args.out_prefix}.json"
    csv_path = DATA_DIR / f"{args.out_prefix}.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=float))

    csv_rows = [
        "quantity,point,ci95_lo,ci95_hi",
        f"instruct_df_a_correct,{point['instruct'][0]:.6f},{ci_instruct[0]:.6f},{ci_instruct[1]:.6f}",
        f"thinking_df_a_correct,{point['thinking'][0]:.6f},{ci_thinking[0]:.6f},{ci_thinking[1]:.6f}",
        f"ratio_T_over_I,{ratio_point:.4f},{ci_ratio[0]:.4f},{ci_ratio[1]:.4f}",
        f"log_ratio_T_over_I,{math.log(ratio_point):.4f},{ci_logratio[0]:.4f},{ci_logratio[1]:.4f}",
        f"ratio_exp_logCI,{ratio_point:.4f},{math.exp(ci_logratio[0]):.4f},{math.exp(ci_logratio[1]):.4f}",
    ]
    csv_path.write_text("\n".join(csv_rows) + "\n")
    print(f"\n[write] {json_path.relative_to(ROOT)}")
    print(f"[write] {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
