#!/usr/bin/env bash
# C1 follow-up: extend OneVision E1-patch attention extraction to ChartQA +
# MathVista (the two §7.1-7.3 main-matrix datasets currently missing from
# Phase D's OneVision × 4 cells = TallyQA + InfoVQA + PlotQA + VQAv2).
#
# Wall: ~30 min/dataset × 2 = ~1h on H200.

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

LOG_DIR="outputs/_logs/phase3_e1patch_onevision"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
SUMMARY_LOG="$LOG_DIR/run_${TS}.log"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

note() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

run_dataset() {
  local ds="$1" cfg="$2" susc="$3"
  note "=== $ds: starting ==="
  uv run python scripts/extract_attention_mass.py \
      --model llava-onevision-qwen2-7b-ov \
      --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      >> "$LOG_DIR/${ds}_${TS}.log" 2>&1
  note "=== $ds: done ==="
}

run_dataset chartqa configs/experiment_e5e_chartqa_full.yaml \
  docs/insights/_data/susceptibility_chartqa_onevision.csv

run_dataset mathvista configs/experiment_e5e_mathvista_full.yaml \
  docs/insights/_data/susceptibility_mathvista_onevision.csv

note "Both done. Re-running analyzer."
uv run python scripts/analyze_attention_patch.py --print-summary \
    >> "$LOG_DIR/analyze_${TS}.log" 2>&1

note "ALL DONE. Check docs/insights/_data/E1_patch_per_layer.csv"
