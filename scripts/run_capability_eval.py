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
- We use VLMEvalKit's programmatic API (`vlmeval.inference.infer_data_job`
  and `vlmeval.dataset.build_dataset`) rather than its CLI, so the same
  Python process holds both variant model instances when needed.
- For memory hygiene, we tear down the model after every variant×bench
  pair before instantiating the next one.
- `infer_data_job(model, work_dir, model_name, dataset, ...)` writes its
  output to `{work_dir}/{model_name}_{dataset.dataset_name}.xlsx`.
  We pass `model_name=variant` ("baseline" or "mit") so the two arms
  never clobber each other even in the same out_dir.
- `dataset.evaluate(eval_file)` aggregates results and writes side-files
  but does NOT store per-question correctness for VQA/OCRBench types.
  We read per-question correctness from the appropriate side-file
  (MCQ/YOrN) or replicate the scoring loop ourselves (OCRBench).
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

    try:
        dataset = build_dataset(bench_name)
        if max_questions is not None and hasattr(dataset, "data") and len(dataset.data) > max_questions:
            dataset.data = dataset.data.head(max_questions).reset_index(drop=True)
            _log(progress_log, f"  sub-sampled to {len(dataset.data)} questions")

        # Run inference.  infer_data_job writes predictions to:
        #   {out_dir}/{variant}_{dataset.dataset_name}.xlsx
        # We pass model_name=variant so baseline and mit files are distinct.
        _run_inference(model, dataset, variant, out_dir)
        preds_path = out_dir / f"{variant}_{dataset.dataset_name}.xlsx"

        # Score the predictions; extract scalar accuracy + per-question correctness.
        # evaluate() writes side-files we later read for per-question correctness.
        dataset.evaluate(str(preds_path))
        acc, correct = _extract_acc_and_correct(dataset, preds_path)
        _log(progress_log,
             f"END   {variant} × {bench_name}  n={len(correct)}  acc={acc*100:.2f}")

        if variant == "mit":
            calls = getattr(model, "_mit_calls", None)
            if not calls or calls.get("prefill", 0) == 0:
                raise RuntimeError(f"mit hook did not fire on {bench_name}: {calls}")
            _log(progress_log,
                 f"  hook calls on {bench_name}: prefill={calls['prefill']} "
                 f"decode={calls['decode']} other={calls['other']}")

        return {"acc": acc, "correct": correct, "preds_path": str(preds_path)}
    finally:
        # Always free the model — even on hook-0 fail or inference error —
        # so a 13h sweep doesn't OOM the next benchmark on a single failed pair.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_inference(model: Any, dataset: Any, model_name: str,
                   out_dir: Path) -> None:
    """Call VLMEvalKit's infer_data_job with the correct signature.

    Signature (from vlmeval/inference.py):
        infer_data_job(model, work_dir, model_name, dataset,
                       verbose=False, api_nproc=4, ignore_failed=False)

    Writes predictions to: {work_dir}/{model_name}_{dataset.dataset_name}.xlsx
    Returns the model object (we discard it; we already hold it).
    """
    from vlmeval.inference import infer_data_job
    infer_data_job(
        model, str(out_dir), model_name, dataset,
        verbose=False,
    )


def _extract_acc_and_correct(dataset: Any,
                              preds_path: Path) -> tuple[float, list[bool]]:
    """Extract scalar accuracy and per-question correctness from VLMEvalKit's
    side-files, keyed by dataset type.

    VLMEvalKit's evaluate() writes side-files but does not expose per-question
    correctness in its return value for all benchmark types.  We read the
    appropriate side-file depending on dataset type:

    - MCQ (ImageMCQDataset — MMStar, RealWorldQA, MMBench_DEV_EN):
        evaluate() writes {preds_path stem}_exact_matching_result.xlsx with a
        'hit' column (1/0) per question.  Read that file.

    - YOrN (ImageYOrNDataset — HallusionBench):
        evaluate() writes {preds_path stem}_auxmatch.xlsx with a boolean
        'score' column per question.  Read that file.

    - OCRBench (OCRBench class in image_vqa.py):
        evaluate() writes {preds_path stem}_score.json with category sums
        only — no per-question file.  We replicate the per-question match
        loop from VLMEvalKit's OCRBench.evaluate() inline.
        Correctness per question: bool (any answer string matched).
        Scalar accuracy: mean(correct) in [0, 1].
    """
    import pandas as pd

    dataset_name = dataset.dataset_name

    # Determine dataset type from the class hierarchy.
    cls_names = {c.__name__ for c in type(dataset).__mro__}

    # --- OCRBench: no per-question side-file; replicate the scoring loop ---
    if "OCRBench" in cls_names or dataset_name == "OCRBench":
        return _ocrbench_per_question(preds_path)

    # --- YOrN (HallusionBench and related): _auxmatch.xlsx has 'score' col ---
    if "ImageYOrNDataset" in cls_names or "YOrN" in cls_names:
        auxmatch = preds_path.with_name(
            preds_path.stem + "_auxmatch.xlsx"
        )
        if not auxmatch.exists():
            raise FileNotFoundError(
                f"YOrN side-file not found: {auxmatch}  "
                f"(dataset.evaluate() must have failed silently)"
            )
        df = pd.read_excel(auxmatch)
        correct = [bool(int(s)) for s in df["score"]]
        acc = sum(correct) / len(correct) if correct else 0.0
        return acc, correct

    # --- MCQ (ImageMCQDataset — covers MMStar, RealWorldQA, MMBench): ---
    # evaluate() with model=None uses exact_matching and writes
    # {stem}_exact_matching_result.{suffix}
    if "ImageMCQDataset" in cls_names or "MCQ" in cls_names:
        suffix = preds_path.suffix  # .xlsx
        result_file = Path(str(preds_path).replace(
            suffix, f"_exact_matching_result{suffix}"
        ))
        if not result_file.exists():
            raise FileNotFoundError(
                f"MCQ result file not found: {result_file}  "
                f"(dataset.evaluate() must have failed silently)"
            )
        df = pd.read_excel(result_file)
        correct = [bool(int(h)) for h in df["hit"]]
        acc = sum(correct) / len(correct) if correct else 0.0
        return acc, correct

    # --- Fallback: try to read from the predictions file itself ---
    # This path should not be reached for our 5 benchmarks, but is kept
    # as a safety net with an informative error.
    df = pd.read_excel(preds_path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "hit" in cols_lower:
        correct = [bool(int(h)) for h in df[cols_lower["hit"]]]
        acc = sum(correct) / len(correct) if correct else 0.0
        return acc, correct
    raise RuntimeError(
        f"Cannot determine per-question correctness for dataset type "
        f"{type(dataset).__name__} ({dataset_name}).  "
        f"preds_path columns: {list(df.columns)[:10]}.  "
        f"Add an explicit adapter for this dataset type."
    )


def _ocrbench_per_question(preds_path: Path) -> tuple[float, list[bool]]:
    """Replicate OCRBench per-question match logic from VLMEvalKit.

    VLMEvalKit's OCRBench.evaluate() (image_vqa.py) iterates rows and checks
    whether any answer string appears in the prediction (case-folded for most
    categories; raw for Handwritten Mathematical Expression Recognition).
    It only writes category-level sums to _score.json — no per-question file.

    We replicate the same loop so we get a per-question bool array for
    McNemar paired testing.  Correctness = bool (any answer matched).
    Scalar acc = mean(correct) in [0, 1] — comparable across sub-samples
    since it's a proportion, not a raw sum.  The §7.4.5 table column shows
    this as a % (×100), matching OCRBench's conventional "score/10" display.
    """
    import pandas as pd

    df = pd.read_excel(preds_path)
    correct: list[bool] = []
    for _, row in df.iterrows():
        predict = str(row["prediction"])
        try:
            answers = eval(str(row["answer"]))  # stored as a Python list repr
        except Exception:
            answers = [str(row["answer"])]
        category = str(row.get("category", ""))

        matched = False
        if category == "Handwritten Mathematical Expression Recognition":
            for ans in answers:
                ans_norm = ans.strip().replace("\n", " ").replace(" ", "")
                pred_norm = predict.strip().replace("\n", " ").replace(" ", "")
                if ans_norm in pred_norm:
                    matched = True
                    break
        else:
            for ans in answers:
                ans_norm = ans.lower().strip().replace("\n", " ")
                pred_norm = predict.lower().strip().replace("\n", " ")
                if ans_norm in pred_norm:
                    matched = True
                    break
        correct.append(matched)

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

    # --- C3: Self-test guard ---
    # In smoke mode (--max-questions set) the smoke run IS the test; skip.
    # In full mode, run a 2-question wiring check before the 13h sweep.
    if args.max_questions is None:
        benchmarks = cfg["benchmarks"]
        _log(progress_log,
             f"SELF-TEST: wiring check on {benchmarks[0]} (2 questions) ...")
        try:
            _run_one_variant_bench(
                "baseline", benchmarks[0], cfg, run_dir, progress_log,
                max_questions=2,
            )
            _log(progress_log, "SELF-TEST: PASSED — entering full sweep")
        except Exception as e:
            _log(progress_log,
                 f"SELF-TEST FAILED: {type(e).__name__}: {e}\n"
                 f"Aborting before entering the 13h sweep.  Fix the issue and retry.")
            raise SystemExit(1)

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
            # I4: only print PASS/FAIL for OK rows
            verdict_marker = "PASS" if delta >= agg.PER_BENCH_THRESHOLD else "FAIL"
            _log(progress_log,
                 f"DELTA {bench}  Δ={delta*100:+.2f}pp  "
                 f"95%CI=[{row.ci_low*100:+.2f}, {row.ci_high*100:+.2f}]  "
                 f"{verdict_marker}")
        except Exception as e:
            # I4: ERROR rows get their own log line; no PASS/FAIL marker
            _log(progress_log, f"ERROR on {bench}: {type(e).__name__}: {e}")
            rows.append(agg.BenchRow(
                name=bench, n=0,
                acc_baseline=0.0, acc_mit=0.0,
                delta=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
                status="ERROR", note=f"{type(e).__name__}: {e}",
            ))
            agg.write_partial(rows, partial_csv, partial_md)

    v = agg.finalize(partial_csv, partial_md, final_csv, final_md)
    # I2: route through _log so these land in progress.log
    _log(progress_log, f"FINAL VERDICT: {v}")
    _log(progress_log, f"Final table: {final_md}")
    raise SystemExit(0 if v == "STRICT_FREE_LUNCH" else 1)


if __name__ == "__main__":
    main()
