#!/usr/bin/env bash
# Phase 1 P0 v3: LLaVA-OneVision-7B-OV Main baseline.
# Runs llava-onevision-qwen2-7b-ov on the 5-dataset main matrix:
#   PlotQA n=5000 stratified, InfoVQA n=1147 full numeric,
#   TallyQA full 38k, ChartQA ~705 numeric, MathVista ~385 numeric.
#
# OneVision uses AnyRes per image: each image yields up to 7 crops × 384×384
# = up to 5103 visual tokens. Multi-image native; 2-image setup (target +
# irrelevant anchor) verified working via 5-sample sanity test.
#
# Sub-A (qwen2.5-vl-7b) + Sub-B (gemma3-27b) data already exists from prior
# Phase 1 runs — this script only adds the OneVision Main cells.
#
# Sequential on GPU 1.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/onevision_baseline.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

run_one() {
  local cfg="$1" tag="$2"
  note "==== $tag :: llava-onevision-qwen2-7b-ov ===="
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
      --config "$cfg" \
      --models llava-onevision-qwen2-7b-ov
}

note "==== Phase 1 P0 v3 OneVision baseline start ===="

run_one configs/experiment_e7_plotqa_full.yaml          plotqa
run_one configs/experiment_e7_infographicvqa_full.yaml  infovqa
run_one configs/experiment_e5e_tallyqa_full.yaml        tallyqa
run_one configs/experiment_e5e_chartqa_full.yaml        chartqa
run_one configs/experiment_e5e_mathvista_full.yaml      mathvista

note "==== Phase 1 P0 v3 OneVision baseline done ===="
