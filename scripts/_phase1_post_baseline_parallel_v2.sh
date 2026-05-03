#!/usr/bin/env bash
# Phase 1 P0 v3 post-baseline orchestrator — v2 multi-GPU with sharded
# calibrations and sharded sweep_tally.
#
# Differences vs _phase1_post_baseline_parallel.sh (v1):
#  Stage 2  Sharded calibrations: calib_plotqa + calib_infovqa each run
#           K=3 sharded across GPU 0/1/2 (calib_plotqa ~20min, calib_infovqa
#           ~5min vs 60+30min single-GPU). chart_base + math_base run
#           on GPU 0 first (~15min) since they need a single GPU each.
#  Stage 4  Sharded sweep_tally GPU 0/1/2 (~30min vs ~90min single-GPU).
#           Non-tally sweeps run in parallel after tally finishes
#           (sweep_plotqa GPU0, sweep_infovqa→chartqa GPU1,
#           sweep_mathvista GPU2 — all small).
#
# Tally Main is NOT re-run: v1's stage 1 output remains canonical and is
# detected via cell_done(). To switch from v1: SIGTERM v1 once its
# stage 1 completes, then launch this script.
#
# All steps idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_baseline_parallel_v2.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

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

latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

note "==== Phase 1 P0 v3 PARALLEL_v2 orchestrator start ===="

# ---------------------------------------------------------------------------
# Stage 1: TallyQA Main (re-shard if v1 didn't complete; otherwise skip)
# ---------------------------------------------------------------------------
note "---- Stage 1: TallyQA Main (sharded) ----"
if cell_done "outputs/experiment_e5e_tallyqa_full/$MODEL"; then
  note "skip tally Main :: $MODEL (already complete from v1 or earlier)"
else
  uv run python scripts/run_experiment_sharded.py \
      --config configs/experiment_e5e_tallyqa_full.yaml \
      --model "$MODEL" \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------------------
# Stage 2: chart + math (GPU 0 sequential) → sharded calibs (all 3 GPUs)
# ---------------------------------------------------------------------------
note "---- Stage 2A: chart_base + math_base on GPU 0 (sequential ~15min) ----"

if cell_done "outputs/experiment_e5e_chartqa_full/$MODEL"; then
  note "[GPU0] skip chart_base"
else
  note "[GPU0] chart_base"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
      --config configs/experiment_e5e_chartqa_full.yaml --models "$MODEL" \
      >> "$LOG_DIR/v2_stage2A_chart.log" 2>&1
fi

if cell_done "outputs/experiment_e5e_mathvista_full/$MODEL"; then
  note "[GPU0] skip math_base"
else
  note "[GPU0] math_base"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
      --config configs/experiment_e5e_mathvista_full.yaml --models "$MODEL" \
      >> "$LOG_DIR/v2_stage2A_math.log" 2>&1
fi

note "---- Stage 2B: sharded calib_plotqa on GPU 0/1/2 (~20min) ----"
PLOT_TS_NOW="$(latest_run experiment_e7_plotqa_full)"
INFO_TS_NOW="$(latest_run experiment_e7_infographicvqa_full)"
[ -n "$PLOT_TS_NOW" ] || { note "ERR: no $MODEL plotqa run"; exit 1; }
[ -n "$INFO_TS_NOW" ] || { note "ERR: no $MODEL infographicvqa run"; exit 1; }
PLOT_PRED="outputs/experiment_e7_plotqa_full/$MODEL/$PLOT_TS_NOW/predictions.jsonl"
INFO_PRED="outputs/experiment_e7_infographicvqa_full/$MODEL/$INFO_TS_NOW/predictions.jsonl"
CALIB_PLOT_DIR="outputs/e6_steering/$MODEL/calibration_plotqa"
CALIB_INFO_DIR="outputs/e6_steering/$MODEL/calibration_infographicvqa"

if [ -f "$CALIB_PLOT_DIR/D_wrong.pt" ] && [ -f "$CALIB_PLOT_DIR/v.pt" ]; then
  note "skip calib_plotqa (D_wrong.pt + v.pt exist)"
else
  uv run python scripts/run_calibrate_subspace_sharded.py \
      --config configs/experiment_e7_plotqa_full.yaml \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$PLOT_PRED" \
      --dataset-tag plotqa \
      --max-calibrate-pairs 2500 \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

note "---- Stage 2C: sharded calib_infovqa on GPU 0/1/2 (~5min) ----"
if [ -f "$CALIB_INFO_DIR/D_wrong.pt" ] && [ -f "$CALIB_INFO_DIR/v.pt" ]; then
  note "skip calib_infovqa (D_wrong.pt + v.pt exist)"
else
  uv run python scripts/run_calibrate_subspace_sharded.py \
      --config configs/experiment_e7_infographicvqa_full.yaml \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$INFO_PRED" \
      --dataset-tag infographicvqa \
      --max-calibrate-pairs 1147 \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

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
# Stage 4: sharded sweep_tally → parallel non-tally sweeps
# ---------------------------------------------------------------------------
note "---- Stage 4A: sharded sweep_tally on GPU 0/1/2 (~30min) ----"
SWEEP_TALLY_DIR="outputs/e6_steering/$MODEL/sweep_subspace_tallyqa_${TAG}"
TALLY_TS="$(latest_run experiment_e5e_tallyqa_full)"
[ -n "$TALLY_TS" ] || { note "ERR: no $MODEL tallyqa baseline"; exit 1; }
TALLY_PRED="outputs/experiment_e5e_tallyqa_full/$MODEL/$TALLY_TS/predictions.jsonl"

if [ -f "$SWEEP_TALLY_DIR/predictions.jsonl" ]; then
  note "skip sweep_tally (predictions.jsonl exists)"
else
  uv run python scripts/run_sweep_subspace_sharded.py \
      --config configs/experiment_e5e_tallyqa_full.yaml \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$TALLY_PRED" \
      --dataset-tag tallyqa \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

note "---- Stage 4B: parallel non-tally sweeps (GPU 0/1/2) ----"

sweep_one_single() {
  local ds="$1" cfg="$2" exp="$3" gpu="$4" log_tag="$5"
  local sweep_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${TAG}"
  if [ -f "$sweep_dir/predictions.jsonl" ]; then
    note "[GPU$gpu] skip $ds (already exists)"
    return 0
  fi
  local ts; ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "[GPU$gpu] WARN: no $MODEL run for $ds — skipping"
    return 0
  fi
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  note "[GPU$gpu] sweep_${ds}  preds=$preds"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model "$MODEL" --hf-model "$HF" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$TAG" \
      --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --config "$cfg" >> "$LOG_DIR/$log_tag.log" 2>&1
}

# GPU 0: plotqa (largest non-tally, ~30min — saves total wall by going alone)
(
  sweep_one_single plotqa configs/experiment_e7_plotqa_full.yaml \
                   experiment_e7_plotqa_full 0 v2_stage4B_gpu0
) &
SP0=$!

# GPU 1: infographicvqa (~10min) → chartqa (~5min)
(
  sweep_one_single infographicvqa configs/experiment_e7_infographicvqa_full.yaml \
                   experiment_e7_infographicvqa_full 1 v2_stage4B_gpu1
  sweep_one_single chartqa configs/experiment_e5e_chartqa_full.yaml \
                   experiment_e5e_chartqa_full 1 v2_stage4B_gpu1
) &
SP1=$!

# GPU 2: mathvista (~5min) → done early
(
  sweep_one_single mathvista configs/experiment_e5e_mathvista_full.yaml \
                   experiment_e5e_mathvista_full 2 v2_stage4B_gpu2
) &
SP2=$!

wait $SP0 $SP1 $SP2
note "Stage 4B complete"

note "---- Stage 4c: analyze sweeps ----"
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
      --root "outputs/$exp" >> "$LOG_DIR/v2_stage5_recompute_${exp}.log" 2>&1 &
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

note "==== Phase 1 P0 v3 PARALLEL_v2 orchestrator done ===="
