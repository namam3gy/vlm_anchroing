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

cell_done() {
  # Returns 0 (true) if a complete predictions.jsonl exists for this
  # (exp, model) and the file size is non-trivial (> 1MB sanity).
  # Idempotency for resume-after-pod-restart.
  local exp="$1"
  local model_dir="outputs/$exp/llava-onevision-qwen2-7b-ov"
  [ -d "$model_dir" ] || return 1
  for ts in "$model_dir"/*; do
    [ -d "$ts" ] || continue
    local f="$ts/predictions.jsonl"
    local s="$ts/summary.json"
    if [ -f "$f" ] && [ -f "$s" ]; then
      local sz
      sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
      if [ "$sz" -gt 1000000 ]; then
        return 0
      fi
    fi
  done
  return 1
}

run_one() {
  local cfg="$1" tag="$2" exp="$3"
  if cell_done "$exp"; then
    note "skip $tag :: llava-onevision-qwen2-7b-ov (already has complete predictions)"
    return 0
  fi
  note "==== $tag :: llava-onevision-qwen2-7b-ov ===="
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
      --config "$cfg" \
      --models llava-onevision-qwen2-7b-ov
}

note "==== Phase 1 P0 v3 OneVision baseline start ===="

run_one configs/experiment_e7_plotqa_full.yaml         plotqa    experiment_e7_plotqa_full
run_one configs/experiment_e7_infographicvqa_full.yaml infovqa   experiment_e7_infographicvqa_full
run_one configs/experiment_e5e_tallyqa_full.yaml       tallyqa   experiment_e5e_tallyqa_full
run_one configs/experiment_e5e_chartqa_full.yaml       chartqa   experiment_e5e_chartqa_full
run_one configs/experiment_e5e_mathvista_full.yaml     mathvista experiment_e5e_mathvista_full

note "==== Phase 1 P0 v3 OneVision baseline done ===="
