#!/usr/bin/env bash
# Phase 1 P0 baseline: 3-model × 2-dataset sequential on GPU 1.
# Sequence: PlotQA (llava → qwen → gemma) then InfoVQA (llava → qwen → gemma).
# Each model loaded fresh per sub-run via run_experiment.py --models <name>.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"

run_one() {
  local cfg="$1" model="$2" tag="$3"
  echo "==== $(date '+%H:%M:%S') ${tag} :: ${model} ===="
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
      --config "$cfg" \
      --models "$model"
}

echo "==== Phase 1 P0 baseline start: $(date '+%Y-%m-%d %H:%M:%S') ===="

for cfg_pair in \
    "configs/experiment_e7_plotqa_full.yaml plotqa" \
    "configs/experiment_e7_infographicvqa_full.yaml infovqa" \
    "configs/experiment_e5e_tallyqa_full.yaml tallyqa"; do
  cfg="${cfg_pair% *}"; tag="${cfg_pair##* }"
  for model in llava-next-interleaved-7b qwen2.5-vl-7b-instruct gemma3-27b-it; do
    run_one "$cfg" "$model" "$tag"
  done
done

echo "==== Phase 1 P0 baseline done: $(date '+%Y-%m-%d %H:%M:%S') ===="
