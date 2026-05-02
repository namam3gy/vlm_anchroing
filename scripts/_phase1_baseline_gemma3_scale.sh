#!/usr/bin/env bash
# Phase 1 P0 v2: Gemma3 family scale-down panel runs.
# Runs gemma3-12b-it + gemma3-4b-it on the 5-dataset main matrix
# (PlotQA + InfoVQA + TallyQA + ChartQA + MathVista). Gemma3-27b-it
# data already collected by the earlier _phase1_baseline.sh run on
# PlotQA / InfoVQA / TallyQA, and pre-existing on ChartQA / MathVista.
#
# Sequential on GPU 1 (single-GPU constraint). Idempotent — skips a
# (model, dataset) cell if its predictions.jsonl already exists with
# matching n records.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gemma3_scale.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

run_one() {
  local cfg="$1" model="$2" tag="$3"
  note "==== $tag :: $model ===="
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
      --config "$cfg" \
      --models "$model"
}

note "==== Phase 1 P0 v2 gemma3-scale baseline start ===="

for model in gemma3-12b-it gemma3-4b-it; do
  for cfg_pair in \
      "configs/experiment_e7_plotqa_full.yaml plotqa" \
      "configs/experiment_e7_infographicvqa_full.yaml infovqa" \
      "configs/experiment_e5e_tallyqa_full.yaml tallyqa" \
      "configs/experiment_e5e_chartqa_full.yaml chartqa" \
      "configs/experiment_e5e_mathvista_full.yaml mathvista"; do
    cfg="${cfg_pair% *}"; tag="${cfg_pair##* }"
    run_one "$cfg" "$model" "$tag"
  done
done

note "==== Phase 1 P0 v2 gemma3-scale baseline done ===="
