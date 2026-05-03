#!/usr/bin/env bash
# Phase 1 P0 §1.2 + §1.3: E6 Subspace recalibration on PlotQA + InfoVQA
# pooled, then headline-cell sweep across 5 datasets at full gt range.
#
# Pre-req: Phase 1 baseline (scripts/_phase1_baseline.sh) has produced llava
# preds for experiment_e7_plotqa_full and experiment_e7_infographicvqa_full.
#
# Idempotency: each sub-step is skipped if its primary output exists. Re-run
# safe; pass FORCE=1 to override.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k
LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/recalibrate_sweep.log"
FORCE="${FORCE:-0}"

note() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
exists_or_force() { [ "$FORCE" = "1" ] || [ ! -e "$1" ]; }

note "==== §1.2 calibrate-subspace ===="

# Pick latest llava E7 run dir (largest predictions.jsonl, fall back to lexically last).
latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  # largest by record count, ties broken by name (lexical = ts-newest)
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

PLOT_TS="$(latest_run experiment_e7_plotqa_full)"
INFO_TS="$(latest_run experiment_e7_infographicvqa_full)"
[ -n "$PLOT_TS" ] || { note "ERR: no llava plotqa run"; exit 1; }
[ -n "$INFO_TS" ] || { note "ERR: no llava infographicvqa run"; exit 1; }
PLOT_PRED="outputs/experiment_e7_plotqa_full/$MODEL/$PLOT_TS/predictions.jsonl"
INFO_PRED="outputs/experiment_e7_infographicvqa_full/$MODEL/$INFO_TS/predictions.jsonl"
note "PlotQA  llava preds: $PLOT_PRED ($(wc -l <"$PLOT_PRED") records)"
note "InfoVQA llava preds: $INFO_PRED ($(wc -l <"$INFO_PRED") records)"

CALIB_PLOT_DIR="outputs/e6_steering/$MODEL/calibration_plotqa"
CALIB_INFO_DIR="outputs/e6_steering/$MODEL/calibration_infographicvqa"

if exists_or_force "$CALIB_PLOT_DIR/D_wrong.pt"; then
  note "calibrate-subspace plotqa (--max-calibrate-pairs 2500)"
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \
      --phase calibrate-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$PLOT_PRED")" \
      --predictions-path "$PLOT_PRED" \
      --dataset-tag plotqa \
      --max-calibrate-pairs 2500 \
      --config configs/experiment_e7_plotqa_full.yaml >> "$LOG" 2>&1
else
  note "skip calibrate-subspace plotqa (already exists)"
fi

if exists_or_force "$CALIB_INFO_DIR/D_wrong.pt"; then
  note "calibrate-subspace infographicvqa (--max-calibrate-pairs 1147)"
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \
      --phase calibrate-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$INFO_PRED")" \
      --predictions-path "$INFO_PRED" \
      --dataset-tag infographicvqa \
      --max-calibrate-pairs 1147 \
      --config configs/experiment_e7_infographicvqa_full.yaml >> "$LOG" 2>&1
else
  note "skip calibrate-subspace infographicvqa (already exists)"
fi

note "==== §1.2 SVD pooled subspace ===="
SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"
if exists_or_force "$SUBSPACE_PT"; then
  uv run python scripts/e6_compute_subspace.py \
      --model "$MODEL" --scope "$TAG" --tags plotqa,infographicvqa --K-max 16 >> "$LOG" 2>&1
else
  note "skip e6_compute_subspace ($SUBSPACE_PT exists)"
fi
[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt not produced"; exit 1; }
note "subspace ready: $SUBSPACE_PT"

note "==== §1.3 sweep-subspace L31_K04_α=1.0 across 5 datasets ===="

declare -A DS_CFG=(
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
  [tallyqa]="configs/experiment_e5e_tallyqa_full.yaml"
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
)

declare -A DS_EXP=(
  [plotqa]="experiment_e7_plotqa_full"
  [infographicvqa]="experiment_e7_infographicvqa_full"
  [tallyqa]="experiment_e5e_tallyqa_full"
  [chartqa]="experiment_e5e_chartqa_full"
  [mathvista]="experiment_e5e_mathvista_full"
)

for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  exp="${DS_EXP[$ds]}"
  cfg="${DS_CFG[$ds]}"
  ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "WARN: no llava run for $ds — skipping"
    continue
  fi
  preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}"
  if ! exists_or_force "$sweep_dir/predictions.jsonl"; then
    note "skip sweep-subspace $ds (already exists)"
    continue
  fi
  note "sweep-subspace $ds  preds=$preds (--max-samples 5000)"
  # Cap to 5000 wrong-base sids per the 2026-05-02 sweep-cap revision
  # (was project.md §0.4.3 §7.4.5 "500 wrong-base"; raised to 5000 for
  # better statistical power per evidence: wrong-base df is 3-5× stronger
  # than correct-base, so concentrating sweep there preserves signal).
  # Smaller datasets (ChartQA 416, MathVista 270 eligible-4cond wb) cap
  # naturally below 5000.
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --config "$cfg" >> "$LOG" 2>&1
done

note "==== §1.3 analyze sweeps ===="
for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}"
  [ -f "$sweep_dir/predictions.jsonl" ] || continue
  note "analyze $ds"
  uv run python scripts/analyze_e6_subspace.py --sweep-dir "$sweep_dir" >> "$LOG" 2>&1
done

note "==== Phase 1 §1.2 + §1.3 done ===="
