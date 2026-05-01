#!/usr/bin/env bash
# Phase 1 post-baseline orchestrator — runs §1.2 → §1.3 → §1.4 in order
# once the §1.1 baseline (scripts/_phase1_baseline.sh) has produced fresh
# predictions for PlotQA + InfoVQA + TallyQA at n=5000 / 1147 / 5000.
#
# §1.2: E6 calibrate-subspace on PlotQA + InfoVQA pooled wrong-base
# §1.3: E6 sweep-subspace at L31_K04_α=1.0 across 5 datasets (5000 wb cap)
# §1.4-A: recompute_answer_span_confidence.py on all main-matrix preds
# §1.4-B: per_cell.csv refresh per dataset
# §1.4-C: confidence-anchoring multi-proxy quartile + monotonicity
# §1.4-D: 5-dataset main-matrix summary (per_cell + relaxed/ANLS supp)
#
# All steps are idempotent. Sequential by design (single-GPU constraint).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_baseline.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

note "==== Phase 1 post-baseline orchestrator start ===="

note "---- §1.2 + §1.3 calibrate-subspace + sweep ----"
bash scripts/_phase1_recalibrate_sweep.sh

note "---- §1.4-A recompute_answer_span_confidence.py (CPU, post-hoc proxies) ----"
uv run python scripts/recompute_answer_span_confidence.py \
    --root outputs/experiment_e7_plotqa_full         >> "$LOG" 2>&1
uv run python scripts/recompute_answer_span_confidence.py \
    --root outputs/experiment_e7_infographicvqa_full >> "$LOG" 2>&1
uv run python scripts/recompute_answer_span_confidence.py \
    --root outputs/experiment_e5e_tallyqa_full       >> "$LOG" 2>&1
# Existing chartqa + mathvista already done; re-running is idempotent (skipped).
uv run python scripts/recompute_answer_span_confidence.py \
    --root outputs/experiment_e5e_chartqa_full       >> "$LOG" 2>&1
uv run python scripts/recompute_answer_span_confidence.py \
    --root outputs/experiment_e5e_mathvista_full     >> "$LOG" 2>&1

note "---- §1.4-B per_cell.csv refresh ----"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  note "  per_cell -> $exp"
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done

note "---- §1.4-C confidence anchoring (multi-proxy quartile + monotonicity) ----"
uv run python scripts/analyze_confidence_anchoring.py --print-summary \
    --primary-proxy cross_entropy >> "$LOG" 2>&1

note "---- §1.4-D 5-dataset main-matrix summary ----"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

note "==== Phase 1 post-baseline orchestrator done ===="
