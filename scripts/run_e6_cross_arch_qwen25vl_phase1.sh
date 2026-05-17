#!/usr/bin/env bash
# E6 cross-arch on Qwen2.5-VL-7B-Instruct — Phase 1 pilot grid driver.
#
# Mirrors scripts/_phase1_pilot_grid.sh (OneVision) with two expansions:
#   1. Model swap → qwen2.5-vl-7b-instruct (Qwen2 ViT NaViT encoder).
#   2. Grid L bin expanded from {25, 26, 27} → {14, 20, 25, 26, 27}
#      per user direction 2026-05-17 PM. L=14 tests outline §5.3
#      dataset-dependent peak heterogeneity (InfoVQA L=14 on OneVision);
#      L=20 tests §5.4 P4 framework mid-stack negative-effect signal.
#
# Total: 5 L × 3 K × 3 α = 45 cells per dataset, run on PlotQA + InfoVQA
# pilots (within-distribution at this stage). Sharded across 5 GPUs.
#
# Pre-req: Phase 0 complete — needs
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt
#
# Output:
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/pilot_grid_plotqa_n250/predictions.jsonl
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/pilot_grid_infographicvqa_n250/predictions.jsonl
#   docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv
#
# Budget: ~1h per dataset wall (5-GPU sharded; ~0.3-0.4s/forward × 45 cells ×
#         250 sids × 4 conds ÷ 5 GPUs). Two datasets sequential ≈ 2h wall.
#
# Usage:
#   bash scripts/run_e6_cross_arch_qwen25vl_phase1.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=qwen2.5-vl-7b-instruct
HF=Qwen/Qwen2.5-VL-7B-Instruct
TAG=plotqa_infovqa_pooled

LAYERS="14,20,25,26,27"
KS="2,4,8"
ALPHAS="0.5,1.0,2.0"
N_PILOT=250
GPUS="${PHASE1_GPUS:-0,1,2,3,4}"

LOG_DIR=outputs/e6_steering/$MODEL
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/_phase1_pilot.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"
[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt not found at $SUBSPACE_PT (run Phase 0 first)"; exit 1; }

run_pilot() {
  local ds="$1" cfg="$2" exp="$3"
  local ts; ts="$(latest_run "$exp")"
  [ -n "$ts" ] || { note "ERR: no $MODEL run for $exp"; exit 1; }
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"

  # Pilot output goes to a name distinct from canonical sweep_subspace_*
  local pilot_dir="outputs/e6_steering/$MODEL/pilot_grid_${ds}_n${N_PILOT}"
  if [ -f "$pilot_dir/predictions.jsonl" ]; then
    note "skip $ds pilot (predictions.jsonl exists at $pilot_dir)"
    return 0
  fi

  note "==== Pilot $ds (N=$N_PILOT wb sids, 45 cells, GPUs=$GPUS) ===="
  uv run python scripts/run_sweep_subspace_sharded.py \
      --config "$cfg" \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds" \
      --dataset-tag "${ds}_pilot_n${N_PILOT}" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers "$LAYERS" --sweep-ks "$KS" --sweep-alphas "$ALPHAS" \
      --max-samples "$N_PILOT" \
      --gpus "$GPUS" >> "$LOG" 2>&1

  # Move from auto-generated tag dir to our canonical pilot dir name
  local auto_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_pilot_n${N_PILOT}_${TAG}"
  if [ -d "$auto_dir" ]; then
    mkdir -p "$pilot_dir"
    mv "$auto_dir"/* "$pilot_dir/"
    rmdir "$auto_dir"
  fi
  note "$ds pilot complete -> $pilot_dir"
}

note "==== Qwen2.5-VL-7B Phase 1 pilot grid (45 cells, L=14,20,25,26,27) start ===="
run_pilot plotqa configs/experiment_e7_plotqa_full.yaml experiment_e7_plotqa_full
run_pilot infographicvqa configs/experiment_e7_infographicvqa_full.yaml experiment_e7_infographicvqa_full

note "==== Pilot grid done — running cell-selection aggregator ===="
uv run python scripts/analyze_e6_pilot_cells.py \
    --plotqa-dir   "outputs/e6_steering/$MODEL/pilot_grid_plotqa_n${N_PILOT}" \
    --infovqa-dir  "outputs/e6_steering/$MODEL/pilot_grid_infographicvqa_n${N_PILOT}" \
    --baseline-plotqa  "outputs/experiment_e7_plotqa_full/$MODEL" \
    --baseline-infovqa "outputs/experiment_e7_infographicvqa_full/$MODEL" \
    --em-drop-deal-breaker 0.06 \
    --output docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv >> "$LOG" 2>&1 || \
    note "WARN: aggregator failed — check log + run analyze_e6_pilot_cells.py manually"

note "==== Phase 1 COMPLETE — inspect docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv ===="
