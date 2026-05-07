"""Aggregator for E8 capability eval.

Pure helpers (`mcnemar_paired`, `verdict`, `bootstrap_score_diff`) are unit-
tested and reused by the driver for partial updates after each benchmark.
The CLI mode at the bottom is the post-run entry point.

Outputs:
- docs/insights/_data/capability_eval_per_benchmark.{csv,md}
- progress.log line per benchmark (when invoked from the driver)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PER_BENCH_THRESHOLD = -0.01  # -1.0pp on proportion scale
MACRO_THRESHOLD = -0.005     # -0.5pp on proportion scale


@dataclass
class BenchRow:
    name: str
    n: int
    acc_baseline: float        # in [0, 1]
    acc_mit: float
    delta: float
    se: float
    ci_low: float
    ci_high: float
    status: str = "OK"         # OK | INCOMPLETE | ERROR
    note: str = ""


def mcnemar_paired(correct_baseline: Iterable[bool],
                   correct_mit: Iterable[bool]) -> tuple[float, float]:
    """Δ = p_mit - p_baseline; SE under McNemar (paired Bernoulli).

    SE on the proportion scale ≈ sqrt(b + c) / n where b is the count of
    pairs only-correct-in-baseline and c is only-correct-in-mit.
    """
    b = list(correct_baseline)
    m = list(correct_mit)
    if len(b) != len(m):
        raise ValueError(f"length mismatch: {len(b)} vs {len(m)}")
    n = len(b)
    if n == 0:
        return 0.0, 0.0
    p_b = sum(b) / n
    p_m = sum(m) / n
    delta = p_m - p_b
    only_b = sum(1 for bi, mi in zip(b, m) if bi and not mi)
    only_m = sum(1 for bi, mi in zip(b, m) if mi and not bi)
    se = math.sqrt(only_b + only_m) / n
    return delta, se


def bootstrap_score_diff(scores_baseline: list[float],
                         scores_mit: list[float],
                         n_bootstrap: int = 1000,
                         seed: int = 0) -> tuple[float, float]:
    """For sum-style scoring (OCRBench): Δ + 95% CI via paired bootstrap."""
    if len(scores_baseline) != len(scores_mit):
        raise ValueError("paired scores must have same length")
    rng = random.Random(seed)
    diffs = [m - b for b, m in zip(scores_baseline, scores_mit)]
    n = len(diffs)
    delta = sum(diffs) / n
    samples = []
    for _ in range(n_bootstrap):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(resample) / n)
    samples.sort()
    lo = samples[int(0.025 * n_bootstrap)]
    hi = samples[int(0.975 * n_bootstrap)]
    se_proxy = (hi - lo) / (2 * 1.96)
    return delta, se_proxy


def verdict(per_bench_deltas: dict[str, float], macro_delta: float) -> str:
    failed = [k for k, v in per_bench_deltas.items() if v < PER_BENCH_THRESHOLD]
    if failed:
        return f"DEGRADED:{','.join(sorted(failed))}"
    if macro_delta < MACRO_THRESHOLD:
        return "DEGRADED:macro"
    return "STRICT_FREE_LUNCH"


# ── Public API used by the driver for partial updates ─────────────────


def make_row(name: str, n: int,
             acc_baseline: float, acc_mit: float,
             se: float) -> BenchRow:
    delta = acc_mit - acc_baseline
    ci = 1.96 * se
    return BenchRow(name=name, n=n,
                    acc_baseline=acc_baseline, acc_mit=acc_mit,
                    delta=delta, se=se,
                    ci_low=delta - ci, ci_high=delta + ci)


def write_partial(rows: list[BenchRow], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "n", "acc_baseline", "acc_mit",
                    "delta", "se", "ci_low", "ci_high", "status", "note"])
        for r in rows:
            w.writerow([r.name, r.n, r.acc_baseline, r.acc_mit,
                        r.delta, r.se, r.ci_low, r.ci_high, r.status, r.note])
    md_lines = [
        "| Benchmark | n | baseline | +mit | Δ | 95% CI | status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        ci_str = f"[{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]"
        md_lines.append(
            f"| {r.name} | {r.n} | {r.acc_baseline*100:.2f} | {r.acc_mit*100:.2f} | "
            f"{r.delta*100:+.2f} | {ci_str} | {r.status} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")


def finalize(partial_csv: Path, partial_md: Path,
             final_csv: Path, final_md: Path) -> str:
    """Rename _partial → final and append verdict line. Returns verdict string."""
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(partial_csv)
    deltas = {r.name: r.delta for r in rows if r.status == "OK"}
    if not deltas:
        v = "DEGRADED:no_complete_rows"
    else:
        macro = sum(deltas.values()) / len(deltas)
        v = verdict(deltas, macro)
    write_partial(rows, final_csv, final_md)
    with final_md.open("a") as f:
        if deltas:
            f.write(f"\n**Macro Δ:** {sum(deltas.values())/len(deltas)*100:+.2f}pp")
        else:
            f.write("\n**Macro Δ:** N/A")
        f.write(f"\n\n**Verdict:** `{v}`\n")
    return v


def _read_rows(csv_path: Path) -> list[BenchRow]:
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(BenchRow(
                name=r["benchmark"], n=int(r["n"]),
                acc_baseline=float(r["acc_baseline"]),
                acc_mit=float(r["acc_mit"]),
                delta=float(r["delta"]), se=float(r["se"]),
                ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                status=r.get("status", "OK"),
                note=r.get("note", ""),
            ))
    return rows


# ── CLI ───────────────────────────────────────────────────────────────


def _cli_finalize(args):
    v = finalize(
        Path(args.partial_csv), Path(args.partial_md),
        Path(args.final_csv), Path(args.final_md),
    )
    print(f"Verdict: {v}")
    return 0 if v == "STRICT_FREE_LUNCH" else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    fin = sub.add_parser("finalize")
    fin.add_argument("--partial-csv", required=True)
    fin.add_argument("--partial-md", required=True)
    fin.add_argument("--final-csv", required=True)
    fin.add_argument("--final-md", required=True)
    fin.set_defaults(func=_cli_finalize)
    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
