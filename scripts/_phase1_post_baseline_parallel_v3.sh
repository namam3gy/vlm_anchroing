#!/usr/bin/env bash
# Phase 1 P0 v3 post-baseline orchestrator — v3 with §7.1-7.3 mechanism panel.
#
# Differences vs v2:
#  Stage M (NEW, between Stage 3 and Stage 4):
#   - Build TallyQA susceptibility CSV from merged tally preds
#   - Run §7.1-7.3 attention extraction (E1 + E1-patch) on PlotQA, TallyQA,
#     InfoVQA across GPU 0/1/2 in parallel (~1.5h)
#   - Run analyze_attention_per_layer + analyze_attention_patch (CPU)
#   - Run causal_anchor_ablation (E1d) on PlotQA at empirical peak layer (~1h)
#  Stage 4 then uses the empirical OneVision peak layer for sweep_tally
#  cell selection (instead of inheriting L=31 from prior models).
#
# All steps idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_baseline_parallel_v3.log"
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

note "==== Phase 1 P0 v3 v3-orchestrator start (with §7.1-7.3) ===="

# ---------------------------------------------------------------------------
# Stage 1: Tally Main (skip if already done from v1 or earlier)
# ---------------------------------------------------------------------------
note "---- Stage 1: TallyQA Main (sharded) ----"
if cell_done "outputs/experiment_e5e_tallyqa_full/$MODEL"; then
  note "skip tally Main :: $MODEL (already complete)"
else
  uv run python scripts/run_experiment_sharded.py \
      --config configs/experiment_e5e_tallyqa_full.yaml \
      --model "$MODEL" \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------------------
# Stage 2: chart+math (GPU 0) → sharded calibs (GPU 0/1/2)
# ---------------------------------------------------------------------------
note "---- Stage 2A: chart_base + math_base on GPU 0 ----"

if cell_done "outputs/experiment_e5e_chartqa_full/$MODEL"; then
  note "[GPU0] skip chart_base"
else
  note "[GPU0] chart_base"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
      --config configs/experiment_e5e_chartqa_full.yaml --models "$MODEL" \
      >> "$LOG_DIR/v3_stage2A_chart.log" 2>&1
fi

if cell_done "outputs/experiment_e5e_mathvista_full/$MODEL"; then
  note "[GPU0] skip math_base"
else
  note "[GPU0] math_base"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
      --config configs/experiment_e5e_mathvista_full.yaml --models "$MODEL" \
      >> "$LOG_DIR/v3_stage2A_math.log" 2>&1
fi

note "---- Stage 2B: sharded calib_plotqa on GPU 0/1/2 ----"
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

note "---- Stage 2C: sharded calib_infovqa on GPU 0/1/2 ----"
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
# Stage 3: SVD pooled subspace (CPU)
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
# Stage M: §7.1-7.3 mechanism panel — empirical peak layer for OneVision
# ---------------------------------------------------------------------------
note "---- Stage M: §7.1-7.3 mechanism panel ----"

# Build TallyQA susceptibility CSV from merged tally preds.
TALLY_TS="$(latest_run experiment_e5e_tallyqa_full)"
[ -n "$TALLY_TS" ] || { note "ERR: no $MODEL tallyqa baseline"; exit 1; }
TALLY_PRED="outputs/experiment_e5e_tallyqa_full/$MODEL/$TALLY_TS/predictions.jsonl"
TALLY_SUSC="docs/insights/_data/susceptibility_tallyqa_onevision.csv"
if [ -f "$TALLY_SUSC" ]; then
  note "skip TallyQA susceptibility (already exists)"
else
  note "build TallyQA susceptibility from $TALLY_PRED"
  uv run python scripts/build_dataset_susceptibility.py \
      --predictions "$TALLY_PRED" \
      --output "$TALLY_SUSC" \
      --top-n 100 --bottom-n 100 >> "$LOG" 2>&1
fi

# Run E1 + E1d + analyses via the dedicated orchestrator.
note "running _phase1_e1_extraction.sh ..."
bash scripts/_phase1_e1_extraction.sh

# Read OneVision peak layer (answer-step, overall) from analyses.
PEAK_LAYER="$(uv run python -c "
import pandas as pd, sys
p = 'outputs/attention_analysis/_per_layer/peak_layer_summary.csv'
df = pd.read_csv(p)
mask = (df['model'] == '$MODEL') & (df['stratum'] == 'all') & (df['step'] == 'answer')
sub = df[mask]
if len(sub) == 0:
    sys.stderr.write('no peak row for OneVision\\n'); sys.exit(1)
print(int(sub.iloc[0]['peak_layer']))
" 2>&1 | tail -1)"
if ! [[ "$PEAK_LAYER" =~ ^[0-9]+$ ]]; then
  note "ERR: failed to extract peak layer; defaulting to 31"
  PEAK_LAYER=31
fi
note "OneVision empirical peak layer: L=$PEAK_LAYER (used for stage 4 sweep)"

# ---------------------------------------------------------------------------
# Stage 4: sharded sweep_tally + parallel non-tally sweeps
# ---------------------------------------------------------------------------
note "---- Stage 4A: sharded sweep_tally @ L=$PEAK_LAYER on GPU 0/1/2 ----"
SWEEP_TALLY_DIR="outputs/e6_steering/$MODEL/sweep_subspace_tallyqa_${TAG}"
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
      --sweep-layers "$PEAK_LAYER" --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --gpus 0,1,2 >> "$LOG" 2>&1
fi

note "---- Stage 4B: parallel non-tally sweeps @ L=$PEAK_LAYER ----"

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
      --sweep-layers "$PEAK_LAYER" --sweep-ks 4 --sweep-alphas 1.0 \
      --max-samples 5000 \
      --config "$cfg" >> "$LOG_DIR/$log_tag.log" 2>&1
}

(
  sweep_one_single plotqa configs/experiment_e7_plotqa_full.yaml \
                   experiment_e7_plotqa_full 0 v3_stage4B_gpu0
) &
SP0=$!
(
  sweep_one_single infographicvqa configs/experiment_e7_infographicvqa_full.yaml \
                   experiment_e7_infographicvqa_full 1 v3_stage4B_gpu1
  sweep_one_single chartqa configs/experiment_e5e_chartqa_full.yaml \
                   experiment_e5e_chartqa_full 1 v3_stage4B_gpu1
) &
SP1=$!
(
  sweep_one_single mathvista configs/experiment_e5e_mathvista_full.yaml \
                   experiment_e5e_mathvista_full 2 v3_stage4B_gpu2
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
      --root "outputs/$exp" >> "$LOG_DIR/v3_stage5_recompute_${exp}.log" 2>&1 &
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

note "==== Phase 1 P0 v3 v3-orchestrator done (peak_layer=L=$PEAK_LAYER) ===="
