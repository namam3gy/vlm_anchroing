"""E8 capability eval driver — per-benchmark interleaving.

Runs baseline (vanilla LLaVA-OneVision-HF) + mit (LLaVAOneVisionMitigated)
on each benchmark in sequence; after each benchmark, updates
docs/insights/_data/capability_eval_partial.{csv,md} and appends a
one-line summary to progress.log so the user can `cat` the partial
file or `tail -f` the log to see live results.

Usage:
    uv run python scripts/run_capability_eval.py \\
        --config configs/capability_eval.yaml \\
        [--max-questions 50]    # optional sub-sampling for smoke runs

Notes:
- We use VLMEvalKit's programmatic API (`vlmeval.inference.infer_data`
  and `vlmeval.dataset.build_dataset`) rather than its CLI, so the same
  Python process holds both variant model instances when needed.
- For memory hygiene, we tear down the model after every variant×bench
  pair before instantiating the next one.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from vlm_anchor.capability_eval import LLaVAOneVisionMitigated  # noqa: E402
import aggregate_capability_eval as agg  # noqa: E402


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _log(progress_log: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with progress_log.open("a") as f:
        f.write(line + "\n")


def _build_model(variant: str, cfg: dict, vlm_kwargs: dict) -> Any:
    """Instantiate baseline or mit. variant ∈ {'baseline', 'mit'}."""
    from vlmeval.vlm.llava.llava import LLaVA_OneVision_HF
    if variant == "baseline":
        return LLaVA_OneVision_HF(**vlm_kwargs)
    if variant == "mit":
        cell = cfg["cell"]
        return LLaVAOneVisionMitigated(
            subspace_path=cfg["subspace_path"],
            layer=cell["layer"], K=cell["K"], alpha=cell["alpha"],
            **vlm_kwargs,
        )
    raise ValueError(f"unknown variant: {variant}")


def _run_one_variant_bench(variant: str, bench_name: str, cfg: dict,
                            run_dir: Path, progress_log: Path,
                            max_questions: int | None) -> dict:
    """Run a single (variant, benchmark) pair and return per-question
    correctness array + scalar accuracy."""
    from vlmeval.dataset import build_dataset

    out_dir = run_dir / variant / bench_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(progress_log, f"BEGIN {variant} × {bench_name}")
    model = _build_model(variant, cfg, cfg["vlm_kwargs"])

    dataset = build_dataset(bench_name)
    if max_questions is not None and hasattr(dataset, "data") and len(dataset.data) > max_questions:
        dataset.data = dataset.data.head(max_questions).reset_index(drop=True)
        _log(progress_log, f"  sub-sampled to {len(dataset.data)} questions")

    # Predictions xlsx — VLMEvalKit's standard output
    preds_path = out_dir / f"{bench_name}_preds.xlsx"
    _run_inference(model, dataset, preds_path, out_dir)

    # Score the predictions; extract scalar accuracy + per-question correctness
    score = dataset.evaluate(str(preds_path))
    acc, correct = _extract_acc_and_correct(score, dataset, preds_path)
    _log(progress_log,
         f"END   {variant} × {bench_name}  n={len(correct)}  acc={acc*100:.2f}")

    if variant == "mit":
        calls = getattr(model, "_mit_calls", None)
        if not calls or calls.get("prefill", 0) == 0:
            raise RuntimeError(f"mit hook did not fire on {bench_name}: {calls}")
        _log(progress_log,
             f"  hook calls on {bench_name}: prefill={calls['prefill']} "
             f"decode={calls['decode']} other={calls['other']}")

    # Free model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"acc": acc, "correct": correct, "preds_path": str(preds_path)}


def _run_inference(model: Any, dataset: Any, preds_path: Path,
                   out_dir: Path) -> None:
    """Call VLMEvalKit's inference helper. Adapt to whichever API name
    exists in our pinned version."""
    from vlmeval import inference
    # VLMEvalKit names this differently across versions. Try the most
    # common variants in order.
    for fn_name in ("infer_data", "infer_data_api", "infer_data_job"):
        fn = getattr(inference, fn_name, None)
        if fn is not None:
            return fn(model=model, dataset=dataset,
                      work_dir=str(out_dir), nproc=1, verbose=False)
    raise RuntimeError(
        f"None of infer_data / infer_data_api / infer_data_job exist in "
        f"vlmeval.inference (version {inference.__file__})"
    )


def _extract_acc_and_correct(score: Any, dataset: Any, preds_path: Path
                              ) -> tuple[float, list[bool]]:
    """Adapter: VLMEvalKit's evaluate() returns benchmark-specific objects.

    We read the predictions xlsx and compute scalar accuracy + per-question
    correctness from standard columns. For OCRBench, we also support per-question
    score columns.
    """
    import pandas as pd
    df = pd.read_excel(preds_path)
    cols_lower = {c.lower(): c for c in df.columns}

    # OCRBench-style: per-question score column
    for cand in ("score", "scores"):
        if cand in cols_lower:
            per_q = df[cols_lower[cand]].astype(float).tolist()
            mx = max(per_q) if per_q else 1.0
            if mx > 1.0:
                per_q = [s / mx for s in per_q]  # normalize 0..1
            correct = [s >= 0.5 for s in per_q]
            acc = sum(per_q) / len(per_q) if per_q else 0.0
            return acc, correct

    # MCQ-style: hit column or prediction vs answer
    if "hit" in cols_lower:
        correct = [bool(int(h)) for h in df[cols_lower["hit"]]]
    elif "prediction" in cols_lower and "answer" in cols_lower:
        correct = [str(p).strip().upper() == str(a).strip().upper()
                   for p, a in zip(df[cols_lower["prediction"]],
                                   df[cols_lower["answer"]])]
    else:
        raise RuntimeError(
            f"Cannot extract correctness from columns: {list(df.columns)[:10]}"
        )
    acc = sum(correct) / len(correct) if correct else 0.0
    return acc, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Sub-sample each benchmark to N questions (smoke mode).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = f"run_{_ts()}"
    run_dir = PROJECT_ROOT / cfg["output_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log = run_dir / "progress.log"

    partial_csv = PROJECT_ROOT / cfg["partial_csv"]
    partial_md = partial_csv.with_suffix(".md")
    final_csv = PROJECT_ROOT / cfg["final_csv"]
    final_md = final_csv.with_suffix(".md")

    _log(progress_log,
         f"START run_id={run_id}  benchmarks={cfg['benchmarks']}  "
         f"cell={cfg['cell']}  max_questions={args.max_questions}")

    rows: list[agg.BenchRow] = []
    for bench in cfg["benchmarks"]:
        try:
            base = _run_one_variant_bench(
                "baseline", bench, cfg, run_dir, progress_log, args.max_questions)
            mit = _run_one_variant_bench(
                "mit", bench, cfg, run_dir, progress_log, args.max_questions)

            if len(base["correct"]) != len(mit["correct"]):
                raise RuntimeError(
                    f"length mismatch on {bench}: baseline {len(base['correct'])} "
                    f"vs mit {len(mit['correct'])}"
                )

            delta, se = agg.mcnemar_paired(base["correct"], mit["correct"])
            row = agg.BenchRow(
                name=bench, n=len(base["correct"]),
                acc_baseline=base["acc"], acc_mit=mit["acc"],
                delta=delta, se=se,
                ci_low=delta - 1.96 * se, ci_high=delta + 1.96 * se,
                status="OK",
            )
            rows.append(row)
            agg.write_partial(rows, partial_csv, partial_md)
            verdict_marker = "PASS" if delta >= agg.PER_BENCH_THRESHOLD else "FAIL"
            _log(progress_log,
                 f"DELTA {bench}  Δ={delta*100:+.2f}pp  "
                 f"95%CI=[{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]  "
                 f"{verdict_marker}")
        except Exception as e:
            _log(progress_log, f"ERROR on {bench}: {type(e).__name__}: {e}")
            rows.append(agg.BenchRow(
                name=bench, n=0,
                acc_baseline=0.0, acc_mit=0.0,
                delta=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
                status="ERROR", note=f"{type(e).__name__}: {e}",
            ))
            agg.write_partial(rows, partial_csv, partial_md)

    v = agg.finalize(partial_csv, partial_md, final_csv, final_md)
    _log(progress_log, f"FINAL VERDICT: {v}")
    print(f"\nFinal table: {final_md}")
    print(f"Verdict: {v}")
    raise SystemExit(0 if v == "STRICT_FREE_LUNCH" else 1)


if __name__ == "__main__":
    main()
