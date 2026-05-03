#!/usr/bin/env bash
# Retry the two Phase D panel models that failed due to wrong HF ids:
#   - llava-1.5-7b   (was: liuhaotian/llava-v1.5-7b → llava-hf/llava-1.5-7b-hf)
#   - convllava-7b   (was: ConvLLaVA/ConvLLaVA-Stage5-7B-LoRA → ConvLLaVA/ConvLLaVA-sft-1536)
#
# Runs sequentially on GPU 3 only so we never collide with the master queue
# (which uses GPU 0 for OneVision×VQAv2 and GPU 0/1/2 for Phase E sharding).
# 2 models × 3 datasets = 6 jobs × ~5 min ≈ 30 min wall.
#
# After both succeed, re-runs analyze_attention_per_layer + analyze_attention_patch
# so panel summaries pick up the new data. The next gcommit in the master queue
# (Phase E) will then include the updated panel files.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/retry_failed_panel.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

note "Retry pipeline starting on GPU 3 only (sequential per dataset)."

run_e1_one() {
  local model="$1" hf="$2" cfg="$3" susc="$4" tag="$5"
  local outlog="$LOG_DIR/retry_${model}_${tag}.log"
  note "[GPU3] retry $model on $tag"
  CUDA_VISIBLE_DEVICES=3 uv run python scripts/extract_attention_mass.py \
      --model "$model" --hf-model "$hf" \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      > "$outlog" 2>&1
}

declare -A FIXED_HF=(
  [llava-1.5-7b]="llava-hf/llava-1.5-7b-hf"
  [convllava-7b]="ConvLLaVA/ConvLLaVA-sft-1536"
)

for model in llava-1.5-7b convllava-7b; do
  hf="${FIXED_HF[$model]}"
  run_e1_one "$model" "$hf" \
      configs/experiment_e7_plotqa_full.yaml \
      docs/insights/_data/susceptibility_plotqa_onevision.csv \
      plotqa
  run_e1_one "$model" "$hf" \
      configs/experiment_e5e_tallyqa_full.yaml \
      docs/insights/_data/susceptibility_tallyqa_onevision.csv \
      tallyqa
  run_e1_one "$model" "$hf" \
      configs/experiment_e7_infographicvqa_full.yaml \
      docs/insights/_data/susceptibility_infovqa_onevision.csv \
      infovqa
  note "$model 3-dataset retry done"
done

note "Re-running analyze_attention_per_layer + analyze_attention_patch"
uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1
uv run python scripts/analyze_attention_patch.py --print-summary >> "$LOG" 2>&1 || true
note "Retry pipeline finished."
