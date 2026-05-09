"""P0-1 gamma-beta bridge summary aggregator.

Reads per-trace amplitude JSONL from Phase C (`scripts/run_gamma_beta_bridge.py`);
pairs Instruct ↔ Thinking by (sample_instance_id, condition); computes per-item
Δ = thinking_mean − instruct_mean; runs paired bootstrap CI on the a-S1 (anchor)
and d (neutral) arms; emits canonical CSV + summary MD.

Sub-group analyses: correct-base (Instruct's pred_b == gt on the target_only
condition), wrong-base, all-base. The correct-base column is the §4.5 ×12.7
correct-base amplification region we expect γ-β residual amplitude to track.

Joins with the 2026-04-28 γ-α MathVista Instruct predictions.jsonl to recover
base_correct flags.

Usage:
    uv run python scripts/build_gamma_beta_bridge_summary.py \\
        --instruct-amp outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/<ts>/amplitude_per_trace.jsonl \\
        --thinking-amp outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/<ts>/amplitude_per_trace.jsonl \\
        --instruct-preds outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-instruct/20260428-114421/predictions.jsonl \\
        --B 10000
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--instruct-amp",
        required=True,
        help="amplitude_per_trace.jsonl from Phase C run on Qwen3-VL-Instruct.",
    )
    p.add_argument(
        "--thinking-amp",
        required=True,
        help="amplitude_per_trace.jsonl from Phase C run on Qwen3-VL-Thinking.",
    )
    p.add_argument(
        "--instruct-preds",
        required=True,
        help="γ-α MathVista Instruct predictions.jsonl (target_only used to "
             "recover base_correct flag per sample_instance_id).",
    )
    p.add_argument("--B", type=int, default=10000, help="Bootstrap iterations.")
    p.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha (default 0.05 -> 95 pct CI).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--layer",
        type=int,
        default=33,
        help="Which layer's mean amplitude to use as primary statistic.",
    )
    p.add_argument(
        "--out-csv",
        default="docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv",
    )
    p.add_argument(
        "--out-md",
        default="docs/insights/_data/gamma_beta_bridge_summary.md",
    )
    return p.parse_args()


def paired_bootstrap_ci(
    deltas: np.ndarray,
    B: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Returns (lower, upper) bound of two-sided (1-alpha) CI for mean(deltas).

    Resampling is performed on the per-item Δ vector (paired design — items
    pre-aligned by (sid, condition) before this function is called).
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan")
    # Vectorised: draw all B*n indices in one shot, reshape to (B, n), mean over axis 1.
    idx = rng.integers(0, n, size=(B, n))
    means = deltas[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def filter_by_base_correctness(rows: list[dict], mode: str) -> list[dict]:
    """Filter rows by base_correct flag.

    mode:
        'all'     - return all rows (no filter).
        'correct' - keep only base_correct is True (Python ``is True``);
                    rows with None/False are excluded.
        'wrong'   - keep only base_correct is False; rows with None/True are excluded.

    Rows whose sample_instance_id was missing from the γ-α predictions file
    will have ``base_correct = None`` and are excluded from BOTH 'correct' and
    'wrong' sub-groups (but appear in 'all').
    """
    if mode == "all":
        return list(rows)
    if mode == "correct":
        return [r for r in rows if r.get("base_correct") is True]
    if mode == "wrong":
        return [r for r in rows if r.get("base_correct") is False]
    raise ValueError(f"unknown mode {mode!r}")


def load_amp_jsonl(path: Path, layer: int = 33) -> dict[tuple[str, str], dict]:
    """Read Phase C amplitude_per_trace.jsonl.

    Returns {(sid, condition) : amp_record} where amp_record carries
    'mean', 'max', 'n_tokens', 'raw_text' for the requested layer.
    Records whose layer entry has mean=None (e.g. n_gen=0) are skipped.
    """
    out: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            entry = r.get("amplitude_per_layer", {}).get(str(layer), {})
            mean_amp = entry.get("mean")
            if mean_amp is None:
                continue
            out[(r["sample_instance_id"], r["condition"])] = {
                "mean": float(mean_amp),
                "max": float(entry["max"]) if entry.get("max") is not None else None,
                "n_tokens": r.get("n_generated_tokens"),
                "raw_text": r.get("raw_text", ""),
            }
    return out


def _parse_int(x) -> int | None:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None


def load_base_correctness(preds_path: Path) -> dict[str, bool]:
    """From γ-α MathVista Instruct predictions.jsonl (target_only condition):
    map sample_instance_id → (parsed_pred == parsed_gt).

    Records whose prediction or ground_truth fails int-parse are skipped (not
    inserted at all — caller treats absent sids as base_correct=None).
    """
    out: dict[str, bool] = {}
    with preds_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") != "target_only":
                continue
            sid = r["sample_instance_id"]
            pred = _parse_int(r.get("prediction"))
            gt = _parse_int(r.get("ground_truth"))
            if pred is None or gt is None:
                continue
            out[sid] = (pred == gt)
    return out


def build_per_trace_rows(
    instruct_amp: dict[tuple[str, str], dict],
    thinking_amp: dict[tuple[str, str], dict],
    base_correct: dict[str, bool],
) -> list[dict]:
    """Pair Instruct ↔ Thinking by (sid, condition); attach base_correct."""
    rows: list[dict] = []
    for (sid, cond), inst in instruct_amp.items():
        thnk = thinking_amp.get((sid, cond))
        if thnk is None:
            continue
        inst_mean = inst["mean"]
        thnk_mean = thnk["mean"]
        ratio = (thnk_mean / inst_mean) if inst_mean and inst_mean > 0 else float("nan")
        rows.append({
            "sample_instance_id": sid,
            "condition": cond,
            "instruct_mean_amp": inst_mean,
            "instruct_max_amp": inst["max"],
            "instruct_n_tokens": inst["n_tokens"],
            "thinking_mean_amp": thnk_mean,
            "thinking_max_amp": thnk["max"],
            "thinking_n_tokens": thnk["n_tokens"],
            "delta": thnk_mean - inst_mean,
            "ratio": ratio,
            "base_correct": base_correct.get(sid),
        })
    return rows


def emit_summary(
    rows: list[dict],
    B: int,
    alpha: float,
    seed: int,
    layer: int,
    out_md: Path,
) -> None:
    md: list[str] = [
        "# γ-β Residual-Stream Bridge Summary",
        "",
        f"Per-trace mean amplitude at layer L={layer}; paired bootstrap "
        f"B={B}, α={alpha}, seed={seed}.",
        "",
        f"Total paired rows: {len(rows)}.",
        "",
        "| arm | base | n | mean Δ | 95% CI | mean ratio | CI excludes 0 |",
        "|-----|------|---:|------:|-------|----------:|:-:|",
    ]

    for cond, cond_label in [
        ("target_plus_irrelevant_number_S1", "a-S1"),
        ("target_plus_irrelevant_neutral", "d (control)"),
    ]:
        cond_rows = [r for r in rows if r["condition"] == cond]
        for base_mode in ["all", "correct", "wrong"]:
            sub = filter_by_base_correctness(cond_rows, base_mode)
            n = len(sub)
            if n == 0:
                md.append(f"| {cond_label} | {base_mode} | 0 | n/a | n/a | n/a | n/a |")
                continue
            deltas = np.array([r["delta"] for r in sub], dtype=np.float64)
            inst_means = np.array([r["instruct_mean_amp"] for r in sub], dtype=np.float64)
            thnk_means = np.array([r["thinking_mean_amp"] for r in sub], dtype=np.float64)
            mean_d = float(deltas.mean())
            denom = float(inst_means.mean())
            mean_r = float(thnk_means.mean() / denom) if denom > 0 else float("nan")
            lo, hi = paired_bootstrap_ci(deltas, B=B, alpha=alpha, seed=seed)
            excludes = "✓" if (lo > 0 or hi < 0) else "✗"
            md.append(
                f"| {cond_label} | {base_mode} | {n} | {mean_d:+.3f} | "
                f"[{lo:+.3f}, {hi:+.3f}] | {mean_r:.2f}× | {excludes} |"
            )

    md.extend([
        "",
        "**Acceptance interpretation:**",
        "- Primary positive: a-S1 / all (or correct) row CI excludes 0 with positive mean Δ.",
        "- Quantitative confirm: a-S1 / correct mean ratio ≥ 2×.",
        "- Stronger quantitative: ratio approaches §4.5 ×12.7 correct-base df ratio.",
        "- Alt-1 falsification: d (control) row also shows same positive Δ as a-S1 row "
        "(rules out anchor-specific γ-β bridge interpretation).",
        "",
    ])

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print(f"[write] {out_md}")


def main() -> None:
    args = parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    instruct_path = _resolve(args.instruct_amp)
    thinking_path = _resolve(args.thinking_amp)
    preds_path = _resolve(args.instruct_preds)

    instruct_amp = load_amp_jsonl(instruct_path, layer=args.layer)
    thinking_amp = load_amp_jsonl(thinking_path, layer=args.layer)
    base_correct = load_base_correctness(preds_path)
    print(f"[load] instruct: {len(instruct_amp)} (sid, cond) entries")
    print(f"[load] thinking: {len(thinking_amp)} (sid, cond) entries")
    print(
        f"[load] base_correct: {len(base_correct)} sids "
        f"({sum(base_correct.values())} correct)"
    )

    rows = build_per_trace_rows(instruct_amp, thinking_amp, base_correct)
    print(f"[pair] {len(rows)} paired rows")

    out_csv = _resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {out_csv} ({len(rows)} rows)")
    else:
        print(f"[warn] no paired rows; CSV not written")

    emit_summary(
        rows,
        B=args.B,
        alpha=args.alpha,
        seed=args.seed,
        layer=args.layer,
        out_md=_resolve(args.out_md),
    )


if __name__ == "__main__":
    main()
