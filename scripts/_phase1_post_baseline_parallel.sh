#!/usr/bin/env bash
# Phase 1 P0 v3 post-baseline orchestrator — multi-GPU (0,1,2).
#
# Differences vs _phase1_post_baseline.sh:
#  Stage 1  TallyQA Main is sharded 3-way across GPU 0/1/2 (~3.5h vs ~10h
#           single-GPU). chart + math + plotqa + infovqa Main cells either
#           pre-existed or run in stage 2.
#  Stage 2  chart_base on GPU0, math_base on GPU1; calib_plotqa on GPU1
#           (after math), calib_infovqa on GPU2 — all three GPUs active.
#  Stage 3  SVD subspace (fast, single).
#  Stage 4  5 sweep-subspace cells distributed across 3 GPUs by size.
#  Stage 5  CPU finalization (recompute_confidence ×5 in parallel,
#           per_cell ×5, confidence anchoring, 5-dataset summary).
#
# All steps are idempotent. Safe to re-run after pod restart.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_baseline_parallel.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Idempotency helper: returns 0 if any timestamped subdir of $1 has
# predictions.jsonl + summary.json with predictions > 1MB.
cell_done() {
  local model_dir="$1"
  [ -d "$model_dir" ] || return 1
  for ts in "$model_dir"/*; do
    [ -d "$ts" ] || continue
    local f="$ts/predictions.jsonl"
    local s="$ts/summary.json"
    if [ -f "$f" ] && [ -f "$s" ]; then
      local sz; sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
      [ "$sz" -gt 1000000 ] && return 0
    fi
  done
  return 1
}

# Pick the timestamp dir with the most predictions for given (exp, model).
latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

note "==== Phase 1 P0 v3 PARALLEL post-baseline orchestrator start ===="

# ---------------------------------------------------------------------------
# Stage 1: TallyQA Main, sharded 3-way on GPU 0/1/2 (~3.5h)
# ---------------------------------------------------------------------------
note "---- Stage 1: TallyQA Main (sharded GPU 0/1/2) ----"
if cell_done "outputs/experiment_e5e_tallyqa_full/$MODEL"; then
  note "skip tally Main :: $MODEL (already complete)"
else
  uv run python scripts/run_experiment_sharded.py \
      --config configs/experiment_e5e_tallyqa_full.yaml \
      --model "$MODEL" \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------------------
# Stage 2: chart + math + 2 calibrations, parallelized across 3 GPUs
# ---------------------------------------------------------------------------
note "---- Stage 2: chart/math baselines + calibs (parallel 3 GPUs) ----"

PLOT_TS_NOW="$(latest_run experiment_e7_plotqa_full)"
INFO_TS_NOW="$(latest_run experiment_e7_infographicvqa_full)"
[ -n "$PLOT_TS_NOW" ] || { note "ERR: no $MODEL plotqa run"; exit 1; }
[ -n "$INFO_TS_NOW" ] || { note "ERR: no $MODEL infographicvqa run"; exit 1; }
PLOT_PRED="outputs/experiment_e7_plotqa_full/$MODEL/$PLOT_TS_NOW/predictions.jsonl"
INFO_PRED="outputs/experiment_e7_infographicvqa_full/$MODEL/$INFO_TS_NOW/predictions.jsonl"

CALIB_PLOT_DIR="outputs/e6_steering/$MODEL/calibration_plotqa"
CALIB_INFO_DIR="outputs/e6_steering/$MODEL/calibration_infographicvqa"

# GPU 0: chart_base then math_base (both ~15min combined)
(
  if cell_done "outputs/experiment_e5e_chartqa_full/$MODEL"; then
    note "[GPU0] skip chart_base"
  else
    note "[GPU0] chart_base"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
        --config configs/experiment_e5e_chartqa_full.yaml --models "$MODEL" \
        >> "$LOG_DIR/stage2_gpu0.log" 2>&1
  fi
  if cell_done "outputs/experiment_e5e_mathvista_full/$MODEL"; then
    note "[GPU0] skip math_base"
  else
    note "[GPU0] math_base"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
        --config configs/experiment_e5e_mathvista_full.yaml --models "$MODEL" \
        >> "$LOG_DIR/stage2_gpu0.log" 2>&1
  fi
) &
PID0=$!

# GPU 1: calib_plotqa (~1h, biggest calib)
(
  if [ -f "$CALIB_PLOT_DIR/D_wrong.pt" ]; then
    note "[GPU1] skip calib_plotqa (D_wrong.pt exists)"
  else
    note "[GPU1] calib_subspace plotqa (--max-calibrate-pairs 2500)"
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/e6_steering_vector.py \
        --phase calibrate-subspace \
        --model "$MODEL" --hf-model "$HF" \
        --e5c-run-dir "$(dirname "$PLOT_PRED")" \
        --predictions-path "$PLOT_PRED" \
        --dataset-tag plotqa \
        --max-calibrate-pairs 2500 \
        --config configs/experiment_e7_plotqa_full.yaml \
        >> "$LOG_DIR/stage2_gpu1.log" 2>&1
  fi
) &
PID1=$!

# GPU 2: calib_infovqa (~30min)
(
  if [ -f "$CALIB_INFO_DIR/D_wrong.pt" ]; then
    note "[GPU2] skip calib_infovqa (D_wrong.pt exists)"
  else
    note "[GPU2] calib_subspace infographicvqa (--max-calibrate-pairs 1147)"
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/e6_steering_vector.py \
        --phase calibrate-subspace \
        --model "$MODEL" --hf-model "$HF" \
        --e5c-run-dir "$(dirname "$INFO_PRED")" \
        --predictions-path "$INFO_PRED" \
        --dataset-tag infographicvqa \
        --max-calibrate-pairs 1147 \
        --config configs/experiment_e7_infographicvqa_full.yaml \
        >> "$LOG_DIR/stage2_gpu2.log" 2>&1
  fi
) &
PID2=$!

wait $PID0 $PID1 $PID2
note "Stage 2 complete"

# ---------------------------------------------------------------------------
# Stage 3: SVD pooled subspace (CPU, fast)
# ---------------------------------------------------------------------------
note "---- Stage 3: SVD subspace ----"
SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"
if [ -f "$SUBSPACE_PT" ]; then
  note "skip SVD ($SUBSPACE_PT exists)"
else
  uv run python scripts/e6_compute_subspace.py \
      --model "$MODEL" --scope "$TAG" --tags plotqa,infographicvqa --K-max 16 \
      >> "$LOG" 2>&1
fi
[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt not produced"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 4: 5 sweep-subspace cells distributed across 3 GPUs by expected cost
# ---------------------------------------------------------------------------
note "---- Stage 4: sweep-subspace × 5 datasets (parallel 3 GPUs) ----"

# Returns the sweep_subspace command for one dataset, omitting the
# CUDA_VISIBLE_DEVICES which the caller sets.
sweep_one() {
  local ds="$1" cfg="$2" exp="$3"
  local sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}"
  if [ -f "$sweep_dir/predictions.jsonl" ]; then
    note "[sweep] skip $ds (already exists)"
    return 0
  fi
  local ts; ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "[sweep] WARN: no $MODEL run for $ds — skipping"
    return 0
  fi
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  note "[sweep] $ds  preds=$preds"
  uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --config "$cfg"
}

# GPU 0: tallyqa (largest wb pool, capped at 5000)
(
  CUDA_VISIBLE_DEVICES=0
  export CUDA_VISIBLE_DEVICES
  sweep_one tallyqa configs/experiment_e5e_tallyqa_full.yaml experiment_e5e_tallyqa_full \
    >> "$LOG_DIR/stage4_gpu0.log" 2>&1
) &
SP0=$!

# GPU 1: plotqa → chartqa
(
  CUDA_VISIBLE_DEVICES=1
  export CUDA_VISIBLE_DEVICES
  sweep_one plotqa configs/experiment_e7_plotqa_full.yaml experiment_e7_plotqa_full \
    >> "$LOG_DIR/stage4_gpu1.log" 2>&1
  sweep_one chartqa configs/experiment_e5e_chartqa_full.yaml experiment_e5e_chartqa_full \
    >> "$LOG_DIR/stage4_gpu1.log" 2>&1
) &
SP1=$!

# GPU 2: infographicvqa → mathvista
(
  CUDA_VISIBLE_DEVICES=2
  export CUDA_VISIBLE_DEVICES
  sweep_one infographicvqa configs/experiment_e7_infographicvqa_full.yaml experiment_e7_infographicvqa_full \
    >> "$LOG_DIR/stage4_gpu2.log" 2>&1
  sweep_one mathvista configs/experiment_e5e_mathvista_full.yaml experiment_e5e_mathvista_full \
    >> "$LOG_DIR/stage4_gpu2.log" 2>&1
) &
SP2=$!

wait $SP0 $SP1 $SP2
note "Stage 4 complete"

note "---- Stage 4b: analyze sweeps ----"
for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}"
  [ -f "$sweep_dir/predictions.jsonl" ] || continue
  note "analyze $ds"
  uv run python scripts/analyze_e6_subspace.py --sweep-dir "$sweep_dir" >> "$LOG" 2>&1
done

# ---------------------------------------------------------------------------
# Stage 5: CPU finalization
# ---------------------------------------------------------------------------
note "---- Stage 5: CPU finalization ----"

note "  §1.4-A recompute_answer_span_confidence ×5 (parallel)"
PIDS=()
for exp in experiment_e7_plotqa_full experiment_e7_infographicvqa_full \
           experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full; do
  uv run python scripts/recompute_answer_span_confidence.py \
      --root "outputs/$exp" >> "$LOG_DIR/stage5_recompute_${exp}.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"

note "  §1.4-B per_cell.csv refresh ×5 (sequential)"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done

note "  §1.4-C confidence anchoring (multi-proxy quartile + monotonicity)"
uv run python scripts/analyze_confidence_anchoring.py --print-summary \
    --primary-proxy cross_entropy >> "$LOG" 2>&1

note "  §1.4-D 5-dataset main-matrix summary"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

note "==== Phase 1 P0 v3 PARALLEL post-baseline orchestrator done ===="
