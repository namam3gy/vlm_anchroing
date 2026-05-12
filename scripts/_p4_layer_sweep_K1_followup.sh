#!/usr/bin/env bash
# P4 follow-up: §5.4 routing-vs-integration framework verification
#
# Tier 2 layer sweep (P3) + K=1 cell (P2) on llava-onevision-qwen2-7b-ov.
# Reuses existing _subspace/subspace_plotqa_infovqa_pooled_n5k_K16.pt.
#
# 7 new cells per dataset:
#   L ∈ {5, 10, 15, 20, 25, 27} × K=8 × α=1.0   (P3 layer sweep — early-null / late-significant)
#   L=26 × K=1 × α=1.0                          (P2 single-direction failure on OneVision Main)
#
# Note: chosen cell L=26 × K=8 × α=1.0 already in
# sweep_subspace_<ds>_plotqa_infovqa_pooled_n5k_chosen/ — NOT re-run here.
#
# Idempotent: each dataset's sweep dir is skipped if predictions.jsonl exists.
# FORCE=1 to override.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k
SWEEP_TAG=p4_layer_sweep_K1
LOG_DIR=logs/p4_layer_sweep_K1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run.log"
FORCE="${FORCE:-0}"

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exists_or_force() { [ "$FORCE" = "1" ] || [ ! -e "$1" ]; }

SUBSPACE="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"
[ -f "$SUBSPACE" ] || { note "ERR: subspace .pt missing — $SUBSPACE"; exit 1; }

note "==== P4 Tier 2 layer-sweep + K=1 follow-up ===="
note "subspace: $SUBSPACE"
note "cells: L∈{5,10,15,20,25,27}×K=8×α=1.0  +  L=26×K=1×α=1.0  (7 new cells)"

# Pick latest E7 / E5e per dataset (largest predictions.jsonl, ties → name-newest)
latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

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

# Two separate sweeps to avoid cartesian-product waste:
#  - "layers_K8":  L∈{5,10,15,20,25,27} × K=8 × α=1.0  (6 cells × 4 conds × 1 baseline)
#  - "L26_K1":     L=26                  × K=1 × α=1.0  (1 cell × 4 conds × 1 baseline)
# Output dirs differ by suffix; 2× baseline overhead vs. one combined call,
# but avoids 14-cell cartesian explosion of `--sweep-ks 1,8 --sweep-layers ...`.
#
# Smaller datasets first — quick early signal + less risk of long jam.
# mathvista(170) → chartqa(224) → infographicvqa(443) → plotqa(2306) → tallyqa(4978)

run_sweep() {
  local ds="$1"
  local sweep_subtag="$2"   # "layers_K8" or "L26_K1"
  local layers="$3"
  local ks="$4"

  local exp="${DS_EXP[$ds]}"
  local cfg="${DS_CFG[$ds]}"
  local ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "WARN: no llava run for $ds — skipping"
    return
  fi
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  local sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}_${SWEEP_TAG}_${sweep_subtag}"
  if ! exists_or_force "$sweep_dir/predictions.jsonl"; then
    note "skip $ds/$sweep_subtag (exists: $sweep_dir/predictions.jsonl)"
    return
  fi
  note "==== $ds / $sweep_subtag  preds=$preds  ($(wc -l <"$preds") records) ===="
  note "  out_dir=$sweep_dir   layers=$layers  ks=$ks"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --subspace-path "$SUBSPACE" \
      --subspace-scope "$TAG" \
      --sweep-layers "$layers" \
      --sweep-ks "$ks" \
      --sweep-alphas 1.0 \
      --max-samples 5000 \
      --output-dir "$sweep_dir" \
      --config "$cfg" >> "$LOG" 2>&1
  note "  $ds/$sweep_subtag done — $(wc -l <"$sweep_dir/predictions.jsonl") records"
}

# Order: smaller datasets first, K=1 cell before layer sweep within each
# dataset (K=1 is cheaper and gives the P2 falsification quickly).
for ds in mathvista chartqa infographicvqa plotqa tallyqa; do
  run_sweep "$ds" "L26_K1"    "26"               "1"
  run_sweep "$ds" "layers_K8" "5,10,15,20,25,27" "8"
done

note "==== P4 layer-sweep + K=1 done ===="
note "Aggregation: scripts/aggregate_e6_layer_sweep_K1_p4.py (TBD)"
