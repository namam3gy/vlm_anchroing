#!/usr/bin/env bash
# Recovers gemma4-e4b PlotQA + InfoVQA Phase D outputs that were destroyed by
# a directory-name collision (two parallel processes writing the same
# %Y%m%d-%H%M%S timestamped dir → corrupt JSONL line 7).
# Fix in extract_attention_mass.py adds microseconds + PID to output dir.
#
# Polls for the panel retry (_phase1_retry_failed_panel.sh) to finish first
# so we don't compete for GPU 3, then runs gemma4 sequentially on GPU 3.
# Re-runs analyze_attention_per_layer + analyze_attention_patch at end.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/recover_gemma4_collision.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

note "Waiting for _phase1_retry_failed_panel.sh to complete..."
while pgrep -f "_phase1_retry_failed_panel" > /dev/null; do
  sleep 30
done
note "Retry done — starting gemma4 PlotQA + InfoVQA recovery on GPU 3."

run_e1_one() {
  local cfg="$1" susc="$2" tag="$3"
  local outlog="$LOG_DIR/recover_gemma4-e4b_${tag}.log"
  note "[GPU3] gemma4-e4b on $tag"
  CUDA_VISIBLE_DEVICES=3 uv run python scripts/extract_attention_mass.py \
      --model gemma4-e4b --hf-model google/gemma-4-E4B-it \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      > "$outlog" 2>&1
}

run_e1_one configs/experiment_e7_plotqa_full.yaml \
    docs/insights/_data/susceptibility_plotqa_onevision.csv \
    plotqa
run_e1_one configs/experiment_e7_infographicvqa_full.yaml \
    docs/insights/_data/susceptibility_infovqa_onevision.csv \
    infovqa

note "Re-running analyze_attention_per_layer + analyze_attention_patch (with full panel)"
uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1
uv run python scripts/analyze_attention_patch.py --print-summary >> "$LOG" 2>&1 || true
note "gemma4 recovery + analyze done."
