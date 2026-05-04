#!/usr/bin/env bash
# Re-run OneVision Phase E E1d on the 4 headline datasets with --attn-implementation eager.
# Prior sdpa runs (20260503-111305+, 20260504-015933, 20260504-021140) had ablation no-op
# due to commit 7f8ebb6 SDPA dispatch silently dropping the mask-bias modification.
# Verified 2026-05-04 on 5 chartqa sids: eager 3/5 differ vs sdpa 1/5.
#
# Wall: ~30-45 min/dataset on H200 (eager is 2-3x slower than sdpa).

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

LOG_DIR="outputs/_logs/phase2_e1d_eager_rerun"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
SUMMARY_LOG="$LOG_DIR/run_${TS}.log"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

note() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

run_dataset() {
  local ds="$1" cfg="$2" susc="$3"
  note "=== $ds: starting ==="
  uv run python scripts/causal_anchor_ablation.py \
      --model llava-onevision-qwen2-7b-ov \
      --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
      --peak-layer 27 \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --max-new-tokens 8 \
      --attn-implementation eager \
      >> "$LOG_DIR/${ds}_${TS}.log" 2>&1
  note "=== $ds: done ==="
}

run_dataset tallyqa configs/experiment_e5e_tallyqa_full.yaml \
  docs/insights/_data/susceptibility_tallyqa_onevision.csv

run_dataset infographicvqa configs/experiment_e7_infographicvqa_full.yaml \
  docs/insights/_data/susceptibility_infovqa_onevision.csv

run_dataset chartqa configs/experiment_e5e_chartqa_full.yaml \
  docs/insights/_data/susceptibility_chartqa_onevision.csv

run_dataset mathvista configs/experiment_e5e_mathvista_full.yaml \
  docs/insights/_data/susceptibility_mathvista_onevision.csv

note "All 4 datasets done. Reaggregating + analyzing."
uv run python scripts/reaggregate_paired_adoption.py --apply \
    --root outputs/causal_ablation/llava-onevision-qwen2-7b-ov \
    >> "$LOG_DIR/reaggregate_${TS}.log" 2>&1
uv run python scripts/analyze_causal_ablation.py \
    >> "$LOG_DIR/analyze_${TS}.log" 2>&1

note "ALL DONE. Check outputs/causal_ablation/_summary/per_model_per_mode.csv"
