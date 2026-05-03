#!/usr/bin/env bash
# Phase 1 P0 v3 §7.4.5 pilot grid sweep — cell selection on calibration scope.
#
# Design:
#  - Calibration scope = PlotQA + InfoVQA pooled (Stage 2 sourced V_K from
#    these). Pilot must be in-distribution for principled cell selection.
#  - 27-cell grid: L ∈ {25, 26, 27} × K ∈ {2, 4, 8} × α ∈ {0.5, 1.0, 2.0}
#  - Two separate sweeps (option a) so we can analyze per-dataset AND pooled:
#      sweep #1: PlotQA  N=250 wb sids
#      sweep #2: InfoVQA N=250 wb sids
#  - Sharded K=3 GPU each. Sequential (one dataset at a time). ~38min each.
#
# Pre-req: Stage 4A (single-cell sweep_tally @ L=27 K=4 α=1) finished and
# v3 SIGTERMed before Stage 4B starts. Pilot writes to a separate output
# dir so Stage 4-final's confirmatory sweep_subspace_<ds>_<TAG> remains
# free.
#
# Output:
#   outputs/e6_steering/<model>/pilot_grid_plotqa_n250/predictions.jsonl
#   outputs/e6_steering/<model>/pilot_grid_infographicvqa_n250/predictions.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k

LAYERS="25,26,27"
KS="2,4,8"
ALPHAS="0.5,1.0,2.0"
N_PILOT=250

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pilot_grid.log"
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
[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt not found at $SUBSPACE_PT"; exit 1; }

run_pilot() {
  local ds="$1" cfg="$2" exp="$3"
  local ts; ts="$(latest_run "$exp")"
  [ -n "$ts" ] || { note "ERR: no $MODEL run for $exp"; exit 1; }
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"

  # Pilot output goes to a name distinct from the canonical sweep_subspace_*
  # so Stage 4-final's confirmatory sweep dirs aren't blocked.
  local pilot_dir="outputs/e6_steering/$MODEL/pilot_grid_${ds}_n${N_PILOT}"
  if [ -f "$pilot_dir/predictions.jsonl" ]; then
    note "skip $ds pilot (predictions.jsonl exists)"
    return 0
  fi

  note "==== Pilot $ds (N=$N_PILOT wb sids, 27 cells) ===="
  uv run python scripts/run_sweep_subspace_sharded.py \
      --config "$cfg" \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds" \
      --dataset-tag "${ds}_pilot_n${N_PILOT}" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers "$LAYERS" --sweep-ks "$KS" --sweep-alphas "$ALPHAS" \
      --max-samples "$N_PILOT" \
      --gpus 0,1,2 >> "$LOG" 2>&1

  # Move from auto-generated tag dir to our canonical pilot dir name
  local auto_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_pilot_n${N_PILOT}_${TAG}"
  if [ -d "$auto_dir" ]; then
    mkdir -p "$pilot_dir"
    mv "$auto_dir"/* "$pilot_dir/"
    rmdir "$auto_dir"
  fi
  note "$ds pilot complete -> $pilot_dir"
}

note "==== Phase 1 §7.4.5 pilot-grid start ===="
run_pilot plotqa configs/experiment_e7_plotqa_full.yaml experiment_e7_plotqa_full
run_pilot infographicvqa configs/experiment_e7_infographicvqa_full.yaml experiment_e7_infographicvqa_full

note "==== Pilot grid done — running cell-selection aggregator ===="
uv run python scripts/analyze_e6_pilot_cells.py \
    --plotqa-dir   "outputs/e6_steering/$MODEL/pilot_grid_plotqa_n${N_PILOT}" \
    --infovqa-dir  "outputs/e6_steering/$MODEL/pilot_grid_infographicvqa_n${N_PILOT}" \
    --baseline-plotqa  "outputs/experiment_e7_plotqa_full/$MODEL" \
    --baseline-infovqa "outputs/experiment_e7_infographicvqa_full/$MODEL" \
    --em-drop-deal-breaker 0.06 \
    --output docs/insights/_data/pilot_grid_cell_selection.csv >> "$LOG" 2>&1

note "==== Pilot done. Inspect docs/insights/_data/pilot_grid_cell_selection.csv ===="

# Auto-chain to post-pilot master queue (Stage 4-final → Stage 5 → Phase 1.5
# → E1d ext → new model baselines → qwen2.5-vl §7.1-7.3 → final summary →
# branch merge + push). Per user spec 2026-05-03.
note "==== chaining to _phase1_post_pilot_master_queue.sh ===="
bash scripts/_phase1_post_pilot_master_queue.sh
MASTER_RC=$?
note "==== master queue exit code: $MASTER_RC ===="
exit $MASTER_RC
