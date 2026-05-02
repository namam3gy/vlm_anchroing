#!/usr/bin/env bash
# Phase 1 P0 v3 post-baseline orchestrator (Main = LLaVA-OneVision-7B-OV).
#
# §1.1-onevision: OneVision Main baseline on 5-dataset matrix
# §1.2:           E6 calibrate-subspace (PlotQA + InfoVQA pooled wrong-base) on OneVision backbone
# §1.3:           E6 sweep-subspace at L31_K04_α=1.0 across 5 datasets (5000 wb cap)
# §1.4-A:         recompute_answer_span_confidence.py on all main-matrix preds
# §1.4-B:         per_cell.csv refresh per dataset
# §1.4-C:         confidence-anchoring multi-proxy quartile + monotonicity
# §1.4-D:         5-dataset main-matrix summary (per_cell + relaxed/ANLS supp)
#
# Sub-A (qwen2.5-vl-7b) + Sub-B (gemma3-27b) data already exists from the
# earlier Phase 1 P0 v2 run; this orchestrator only runs OneVision Main +
# the post-OneVision pipeline.
#
# All steps are idempotent. Sequential by design (single-GPU constraint).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_baseline.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

note "==== Phase 1 P0 v3 post-baseline orchestrator start ===="

note "---- §1.1-onevision: LLaVA-OneVision-7B-OV Main baseline on 5 datasets ----"
bash scripts/_phase1_baseline_onevision.sh

note "---- §1.2 + §1.3 calibrate-subspace + sweep (OneVision backbone) ----"
bash scripts/_phase1_recalibrate_sweep.sh

note "---- §1.4-A recompute_answer_span_confidence.py (CPU, post-hoc proxies) ----"
for exp in experiment_e7_plotqa_full experiment_e7_infographicvqa_full \
           experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full; do
  uv run python scripts/recompute_answer_span_confidence.py \
      --root outputs/$exp >> "$LOG" 2>&1
done

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

note "==== Phase 1 P0 v3 post-baseline orchestrator done ===="
