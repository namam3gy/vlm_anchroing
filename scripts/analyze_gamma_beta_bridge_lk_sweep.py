"""P0-1 Phase D' aggregator: L × K sweep on within-Thinking paired differences.

Reads coefficient-per-layer JSONLs (re-calibrated bridge inference, B'/C' phase),
sweeps over (layer, K) cells, and reports for each cell:

  within-Thinking (T_a − T_d) per sid  ← *the* anchor-specificity test
  within-Instruct (I_a − I_d) per sid  ← input-difference effect on short Instruct trace
  DiD = within-Thinking − within-Instruct (algebraic)

The headline metric is within-Thinking, since that is the pure
"anchor presence effect on Thinking-mode reasoning trace amplitude" measurement.
DiD is a derived statistic that double-counts Instruct-side artifacts.

Each cell is bootstrapped (paired, B=10000) for 95 % CI, with Bonferroni-corrected
adjusted CIs reported as a sensitivity column.

Usage:
  uv run python scripts/analyze_gamma_beta_bridge_lk_sweep.py \\
      --instruct-amp <recal instruct jsonl> \\
      --thinking-amp <recal thinking jsonl> \\
      --layers 14 20 25 29 30 33 34 \\
      --K-list 1 2 4 8 12 16 \\
      --B 10000
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--instruct-amp", required=True)
    p.add_argument("--thinking-amp", required=True)
    p.add_argument(
        "--layers", nargs="+", type=int,
        default=[14, 20, 25, 29, 30, 33, 34],
        help="Layers to sweep over",
    )
    p.add_argument(
        "--K-list", nargs="+", type=int,
        default=[1, 2, 4, 8, 12, 16],
        help="K values to sweep over",
    )
    p.add_argument(
        "--stats", nargs="+", default=["mean", "max"],
        choices=["mean", "max"],
        help="Per-trace statistics to compute",
    )
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-csv",
        default="docs/insights/_data/gamma_beta_bridge_lk_sweep.csv",
    )
    p.add_argument(
        "--out-md",
        default="docs/insights/_data/gamma_beta_bridge_lk_sweep_summary.md",
    )
    return p.parse_args()


def boot_ci_paired(
    arr: np.ndarray, B: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) bound of two-sided (1-alpha) CI on mean(arr).

    Vectorized: sample (B, n) indices, mean along last axis."""
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def load_records(path: Path) -> dict[tuple[str, str], dict]:
    """Returns {(sid, condition) : record_dict}."""
    out: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["sample_instance_id"], r["condition"])] = r
    return out


def amp_at_K(coefs: list[list[float]], K: int) -> np.ndarray:
    """Convert per-token coefficient rows (n_gen, K_full) to per-token L2 amplitude
    truncated to first K dimensions.

    coefs: list-of-list (n_gen × K_full) raw projection coefficients
    Returns: numpy 1-D array of shape (n_gen,) with sqrt(sum_{k<K} c_k^2).
    """
    if not coefs:
        return np.zeros(0)
    arr = np.array(coefs, dtype=np.float32)  # (n_gen, K_full)
    if arr.ndim != 2:
        return np.zeros(0)
    K = min(K, arr.shape[1])
    return np.sqrt((arr[:, :K] ** 2).sum(axis=1))


def trace_stat(amps: np.ndarray, stat: str) -> float:
    if amps.size == 0:
        return float("nan")
    if stat == "mean":
        return float(amps.mean())
    if stat == "max":
        return float(amps.max())
    raise ValueError(f"unknown stat {stat!r}")


def main() -> None:
    args = parse_args()
    inst_path = PROJECT_ROOT / args.instruct_amp
    thnk_path = PROJECT_ROOT / args.thinking_amp
    inst = load_records(inst_path)
    thnk = load_records(thnk_path)
    print(f"[load] instruct: {len(inst)} records | thinking: {len(thnk)} records")

    # Find sids with all 4 cells (a + d × instruct + thinking)
    AC = "target_plus_irrelevant_number_S1"
    DC = "target_plus_irrelevant_neutral"
    sids = sorted({
        sid for sid, c in inst
        if c == AC and (sid, AC) in thnk and (sid, DC) in inst and (sid, DC) in thnk
    })
    print(f"[pair] {len(sids)} sids with full (a + d × instruct + thinking) coverage")

    rows: list[dict] = []

    n_cells = len(args.layers) * len(args.K_list) * len(args.stats)
    bonferroni_alpha = args.alpha / n_cells
    print(f"[sweep] {n_cells} cells; Bonferroni-corrected alpha = {bonferroni_alpha:.6f}")

    for L in args.layers:
        for K in args.K_list:
            for stat in args.stats:
                # For each sid: compute trace stat at (L, K) per condition per model
                t_a, t_d, i_a, i_d = [], [], [], []
                for sid in sids:
                    coefs_t_a = thnk[(sid, AC)]["coefficients_per_layer"].get(str(L), [])
                    coefs_t_d = thnk[(sid, DC)]["coefficients_per_layer"].get(str(L), [])
                    coefs_i_a = inst[(sid, AC)]["coefficients_per_layer"].get(str(L), [])
                    coefs_i_d = inst[(sid, DC)]["coefficients_per_layer"].get(str(L), [])
                    if not (coefs_t_a and coefs_t_d and coefs_i_a and coefs_i_d):
                        continue
                    t_a.append(trace_stat(amp_at_K(coefs_t_a, K), stat))
                    t_d.append(trace_stat(amp_at_K(coefs_t_d, K), stat))
                    i_a.append(trace_stat(amp_at_K(coefs_i_a, K), stat))
                    i_d.append(trace_stat(amp_at_K(coefs_i_d, K), stat))
                t_a = np.array(t_a); t_d = np.array(t_d)
                i_a = np.array(i_a); i_d = np.array(i_d)
                n = len(t_a)

                # Within-Thinking paired diff (anchor specificity test)
                wt = t_a - t_d
                # Within-Instruct paired diff
                wi = i_a - i_d
                # DiD (algebraic)
                did = wt - wi

                wt_m, wt_lo, wt_hi = boot_ci_paired(wt, args.B, args.alpha, args.seed)
                wi_m, wi_lo, wi_hi = boot_ci_paired(wi, args.B, args.alpha, args.seed)
                did_m, did_lo, did_hi = boot_ci_paired(did, args.B, args.alpha, args.seed)

                # Bonferroni-corrected CIs (within-Thinking only — primary metric)
                wt_m_b, wt_lo_b, wt_hi_b = boot_ci_paired(
                    wt, args.B, bonferroni_alpha, args.seed
                )

                rows.append({
                    "layer": L,
                    "K": K,
                    "stat": stat,
                    "n": n,
                    "thinking_a_mean": float(t_a.mean()),
                    "thinking_d_mean": float(t_d.mean()),
                    "instruct_a_mean": float(i_a.mean()),
                    "instruct_d_mean": float(i_d.mean()),
                    "within_thinking_mean": wt_m,
                    "within_thinking_ci_lo": wt_lo,
                    "within_thinking_ci_hi": wt_hi,
                    "within_thinking_excl_zero": (wt_lo > 0) or (wt_hi < 0),
                    "within_thinking_bonf_ci_lo": wt_lo_b,
                    "within_thinking_bonf_ci_hi": wt_hi_b,
                    "within_thinking_bonf_excl_zero": (wt_lo_b > 0) or (wt_hi_b < 0),
                    "within_instruct_mean": wi_m,
                    "within_instruct_ci_lo": wi_lo,
                    "within_instruct_ci_hi": wi_hi,
                    "within_instruct_excl_zero": (wi_lo > 0) or (wi_hi < 0),
                    "did_mean": did_m,
                    "did_ci_lo": did_lo,
                    "did_ci_hi": did_hi,
                    "did_excl_zero": (did_lo > 0) or (did_hi < 0),
                })

    # Write CSV
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[write] {out_csv} ({len(rows)} cells)")

    # Summary MD: cells where within-Thinking CI excludes 0 (95 % primary)
    survivors_95 = [r for r in rows if r["within_thinking_excl_zero"]]
    survivors_bonf = [r for r in rows if r["within_thinking_bonf_excl_zero"]]
    md = ["# γ-β Bridge L × K Sweep — Within-Thinking Anchor Specificity",
          "",
          f"Source data: re-calibrated 3-pool (TallyQA + PlotQA + InfoVQA, n_wrong=3017) V_K subspace.",
          f"Bridge inference: 7 layers × K=16 raw coefficients per generated token.",
          f"Cells tested: {len(rows)} ({len(args.layers)} layers × {len(args.K_list)} K × {len(args.stats)} stat).",
          f"Bonferroni-corrected alpha = {bonferroni_alpha:.6f} (vs primary alpha = {args.alpha}).",
          f"Paired sids: {rows[0]['n'] if rows else 0}.",
          "",
          "## Headline finding: within-Thinking anchor specificity",
          "",
          f"**95 % CI excludes 0**: {len(survivors_95)} / {len(rows)} cells",
          f"**Bonferroni-corrected CI excludes 0**: {len(survivors_bonf)} / {len(rows)} cells",
          ""]

    if survivors_95:
        md.append("### Cells with within-Thinking 95 % CI excluding zero")
        md.append("")
        md.append("| layer | K | stat | n | within-Thinking | 95 % CI | Bonferroni CI | Bonf ✓ |")
        md.append("|---|---:|---|---:|---:|---|---|:-:|")
        for r in sorted(survivors_95, key=lambda x: -abs(x["within_thinking_mean"])):
            bonf = "✓" if r["within_thinking_bonf_excl_zero"] else "✗"
            md.append(
                f"| {r['layer']} | {r['K']} | {r['stat']} | {r['n']} | "
                f"{r['within_thinking_mean']:+.3f} | "
                f"[{r['within_thinking_ci_lo']:+.3f}, {r['within_thinking_ci_hi']:+.3f}] | "
                f"[{r['within_thinking_bonf_ci_lo']:+.3f}, {r['within_thinking_bonf_ci_hi']:+.3f}] | {bonf} |"
            )
        md.append("")

    md.append("## Full cell-level table (top 20 by |within-Thinking|)")
    md.append("")
    md.append("| layer | K | stat | within-Thinking | 95 % CI | within-Instruct (artifact) | DiD |")
    md.append("|---|---:|---|---:|---|---:|---:|")
    for r in sorted(rows, key=lambda x: -abs(x["within_thinking_mean"]))[:20]:
        excl = "✓" if r["within_thinking_excl_zero"] else "✗"
        md.append(
            f"| {r['layer']} | {r['K']} | {r['stat']} | "
            f"**{r['within_thinking_mean']:+.3f}** | "
            f"[{r['within_thinking_ci_lo']:+.3f}, {r['within_thinking_ci_hi']:+.3f}] {excl} | "
            f"{r['within_instruct_mean']:+.3f} | {r['did_mean']:+.3f} |"
        )

    out_md = PROJECT_ROOT / args.out_md
    out_md.write_text("\n".join(md))
    print(f"[write] {out_md}")


if __name__ == "__main__":
    main()
