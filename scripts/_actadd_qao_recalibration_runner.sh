#!/usr/bin/env bash
# ActAdd + QAO recalibration runner on OneVision Main (PlotQA + InfoVQA pool).
#
# Mirrors LEACE re-cal substrate (PlotQA + InfoVQA, OneVision Main, L=26).
# Tests whether the §7 fail-table verdicts ("ActAdd: ChartQA backfires +57 %",
# "QAO: probe overfits training query distribution") were calibration-pool
# artifacts.
#
# Phases:
#   A. ActAdd
#     A.0  Pool v_wrong / v_all into calibration_plotqa_infovqa_pooled/  (CPU)
#     A.1-5  Tiebreaker per dataset (5 GPU runs, single direction at L=26)
#   B. QAO
#     B.1-2  calibrate-qao on PlotQA + InfoVQA  (extract Q_wrong.pt)
#     B.3    train-probe (CPU, 4 (L_q, L_target=26) probes)
#     B.4    smoke-qao  (wiring check on PlotQA, 1 cell)
#     B.5-9  sweep-qao per dataset (5 GPU runs, n=200 wrong-base sids)
#
# Idempotent — skip if output exists. FORCE=1 to override.
# Logs to logs/_actadd_qao_recal/run.log (tee).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/_actadd_qao_recalibration_runner.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
POOL_TAG=plotqa_infovqa_pooled
LOG_DIR=logs/_actadd_qao_recal
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run.log"
FORCE="${FORCE:-0}"
CUDA="${CUDA_VISIBLE_DEVICES:-0}"

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exists_or_force() { [ "$FORCE" = "1" ] || [ ! -e "$1" ]; }

# Pick the largest-by-line predictions file for a given experiment dir
# (per memory feedback_smoke_run_pollution: avoid alphabetically-latest pollution).
latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

declare -A DS_EXP=(
  [plotqa]="experiment_e7_plotqa_full"
  [infographicvqa]="experiment_e7_infographicvqa_full"
  [tallyqa]="experiment_e5e_tallyqa_full"
  [chartqa]="experiment_e5e_chartqa_full"
  [mathvista]="experiment_e5e_mathvista_full"
)
declare -A DS_CFG=(
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
  [tallyqa]="configs/experiment_e5e_tallyqa_full.yaml"
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
)

# Smallest first → fastest early signal, lowest jam risk
DATASETS_BY_SIZE=(mathvista chartqa infographicvqa plotqa tallyqa)

note "==== ActAdd + QAO recalibration runner ===="
note "model=$MODEL  pool=$POOL_TAG  CUDA_VISIBLE_DEVICES=$CUDA  FORCE=$FORCE"

#---------------------------------------------------------------------------
# A.0  Pool v_wrong / v_all  (CPU)
#---------------------------------------------------------------------------
POOL_DIR="outputs/e6_steering/$MODEL/calibration_${POOL_TAG}"
if exists_or_force "$POOL_DIR/v.pt"; then
  note "[A.0] pooling v_wrong / v_all → $POOL_DIR/v.pt"
  uv run python scripts/_pool_v_actadd_recal.py >> "$LOG" 2>&1
  note "[A.0] done"
else
  note "[A.0] skip (exists: $POOL_DIR/v.pt)"
fi

#---------------------------------------------------------------------------
# A.1-5  ActAdd tiebreaker per dataset (single mean direction at L=26)
#---------------------------------------------------------------------------
ACTADD_CELLS="26:0.5:0,26:1.0:0,26:2.0:0"
note "[A.tb] ActAdd tiebreaker cells: baseline + $ACTADD_CELLS"

for ds in "${DATASETS_BY_SIZE[@]}"; do
  exp="${DS_EXP[$ds]}"
  cfg="${DS_CFG[$ds]}"
  ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "[A.tb/$ds] WARN: no $exp/$MODEL run — skipping"; continue
  fi
  preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  out_dir="outputs/e6_steering/$MODEL/tiebreaker_${ds}__from_${POOL_TAG}"
  out_path="$out_dir/predictions.jsonl"
  if ! exists_or_force "$out_path"; then
    note "[A.tb/$ds] skip (exists: $out_path)"; continue
  fi
  note "[A.tb/$ds] tiebreaker  preds=$preds  out=$out_path"
  CUDA_VISIBLE_DEVICES="$CUDA" uv run python scripts/e6_steering_vector.py \
      --phase tiebreaker \
      --model "$MODEL" --hf-model "$HF" \
      --calibration-tag "$POOL_TAG" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --cells "$ACTADD_CELLS" \
      --config "$cfg" >> "$LOG" 2>&1
  note "[A.tb/$ds] done — $(wc -l <"$out_path") records"
done

#---------------------------------------------------------------------------
# B.1-2  calibrate-qao on PlotQA + InfoVQA  (extract Q_wrong.pt b-arm reprs)
#---------------------------------------------------------------------------
for ds in plotqa infographicvqa; do
  q_path="outputs/e6_steering/$MODEL/calibration_${ds}/Q_wrong.pt"
  if ! exists_or_force "$q_path"; then
    note "[B.cal/$ds] skip (exists: $q_path)"; continue
  fi
  exp="${DS_EXP[$ds]}"
  cfg="${DS_CFG[$ds]}"
  ts="$(latest_run "$exp")"
  preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  note "[B.cal/$ds] calibrate-qao  preds=$preds"
  CUDA_VISIBLE_DEVICES="$CUDA" uv run python scripts/e6_query_adaptive_offset.py \
      --phase calibrate-qao \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --config "$cfg" >> "$LOG" 2>&1
  note "[B.cal/$ds] done"
done

#---------------------------------------------------------------------------
# B.3  train-probe (CPU)  — pooled PlotQA + InfoVQA, L_target=26
#---------------------------------------------------------------------------
PROBE_PAIRS="20:26,24:26,26:26,27:26"
PROBE_DIR="outputs/e6_steering/$MODEL/qao_probe_${POOL_TAG}"
if exists_or_force "$PROBE_DIR/probe_meta.json"; then
  note "[B.train] train-probe  pairs=$PROBE_PAIRS  out=$PROBE_DIR"
  CUDA_VISIBLE_DEVICES="" uv run python scripts/e6_query_adaptive_offset.py \
      --phase train-probe \
      --model "$MODEL" --hf-model "$HF" \
      --calib-tags plotqa,infographicvqa \
      --probe-pairs "$PROBE_PAIRS" \
      --probe-dir "$PROBE_DIR" >> "$LOG" 2>&1
  note "[B.train] done"
else
  note "[B.train] skip (exists: $PROBE_DIR/probe_meta.json)"
fi

#---------------------------------------------------------------------------
# B.4  smoke-qao on PlotQA (1 cell, ~1 min)
#---------------------------------------------------------------------------
SMOKE_DONE_FILE="$LOG_DIR/.smoke_done"
if exists_or_force "$SMOKE_DONE_FILE"; then
  ts_pq="$(latest_run experiment_e7_plotqa_full)"
  preds_pq="outputs/experiment_e7_plotqa_full/$MODEL/$ts_pq/predictions.jsonl"
  note "[B.smoke] smoke-qao L_q=26 L_target=26 alpha=4.0 on plotqa"
  CUDA_VISIBLE_DEVICES="$CUDA" uv run python scripts/e6_query_adaptive_offset.py \
      --phase smoke-qao \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds_pq" \
      --dataset-tag plotqa \
      --calib-tags plotqa,infographicvqa \
      --probe-dir "$PROBE_DIR" \
      --smoke-pair "26:26:4.0" \
      --config "${DS_CFG[plotqa]}" >> "$LOG" 2>&1
  : > "$SMOKE_DONE_FILE"
  note "[B.smoke] done"
else
  note "[B.smoke] skip (sentinel exists)"
fi

#---------------------------------------------------------------------------
# B.5-9  sweep-qao per dataset  (16 cells + baseline, n=200 sids)
#---------------------------------------------------------------------------
QAO_CELLS="20:26:0.5,20:26:1.0,20:26:2.0,20:26:4.0,24:26:0.5,24:26:1.0,24:26:2.0,24:26:4.0,26:26:0.5,26:26:1.0,26:26:2.0,26:26:4.0,27:26:0.5,27:26:1.0,27:26:2.0,27:26:4.0"
SWEEP_N=200
SWEEP_OUT_TAG="from_${POOL_TAG}_n${SWEEP_N}"
note "[B.sw] sweep-qao 16 cells + baseline  max_sweep_sids=$SWEEP_N"

for ds in "${DATASETS_BY_SIZE[@]}"; do
  exp="${DS_EXP[$ds]}"
  cfg="${DS_CFG[$ds]}"
  ts="$(latest_run "$exp")"
  if [ -z "$ts" ]; then
    note "[B.sw/$ds] WARN: no $exp/$MODEL run — skipping"; continue
  fi
  preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"
  # _sweep_qao_output_path produces: sweep_qao_<ds>_<out_tag>_pooled/predictions.jsonl
  out_dir="outputs/e6_steering/$MODEL/sweep_qao_${ds}_${SWEEP_OUT_TAG}_pooled"
  out_path="$out_dir/predictions.jsonl"
  if ! exists_or_force "$out_path"; then
    note "[B.sw/$ds] skip (exists: $out_path)"; continue
  fi
  note "[B.sw/$ds] sweep-qao  preds=$preds  out=$out_path"
  CUDA_VISIBLE_DEVICES="$CUDA" uv run python scripts/e6_query_adaptive_offset.py \
      --phase sweep-qao \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --calib-tags plotqa,infographicvqa \
      --probe-dir "$PROBE_DIR" \
      --sweep-cells "$QAO_CELLS" \
      --max-sweep-sids "$SWEEP_N" \
      --out-tag "$SWEEP_OUT_TAG" \
      --config "$cfg" >> "$LOG" 2>&1
  note "[B.sw/$ds] done — $(wc -l <"$out_path") records"
done

note "==== ActAdd + QAO recalibration runner DONE ===="
note "Aggregation: TBD — see scripts/aggregate_e6_*.py for templates"
