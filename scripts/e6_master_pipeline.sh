#!/usr/bin/env bash
# E6 Tally-only rerun — master pipeline (S0+S0.5+S1+S2+S3)
# All stages run sequentially on GPU 1.
# Run from /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing.
#
# Stages:
#   S0a   calibrate-subspace TallyQA E5e N=5000   (GPU)
#   S0b   calibrate-qao TallyQA E5e N=5000        (GPU)
#   S0.5  pick top-K peak layers from norms       (CPU)
#   S1a   LEACE calibrate-leace Tally-only        (CPU)
#   S1b   LEACE sweep Tally n=500                  (GPU)
#   S1c   LEACE sweep ChartQA n=500                (GPU)
#   S1d   LEACE analyze (one-sided + two-sided)   (CPU)
#   S2a   Subspace SVD compute Tally-only         (CPU)
#   S2b   Subspace sweep Tally n=500               (GPU)
#   S2c   Subspace sweep ChartQA n=500             (GPU)
#   S2d   Subspace analyze                          (CPU)
#   S3a   DPO build pairs (case_by_case rejected) (CPU)
#   S3b   DPO LoRA training (Tally-only)          (GPU)
#   S3c   DPO sweep Tally + ChartQA                (GPU)
#   S3d   DPO analyze                              (CPU)

set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

export CUDA_VISIBLE_DEVICES=1
MODEL=llava-next-interleaved-7b
HF_MODEL=llava-hf/llava-interleave-qwen-7b-hf
TAG=tally_e5e_n5k
N_CALIB=5000
N_SWEEP=500
TOP_K=5

TALLY_PREDS_E5E="outputs/experiment_e5e_tallyqa_full/$MODEL/20260427-171240/predictions.jsonl"
CHARTQA_PREDS_E5E="outputs/experiment_e5e_chartqa_full/$MODEL/20260427-171240/predictions.jsonl"
E5C_DIR="outputs/experiment_e5c_vqa/$MODEL/20260427-123331"
PEAK_JSON="outputs/e6_steering/$MODEL/_subspace/peak_layers_${TAG}.json"

PROGRESS=/tmp/e6_master_progress.log
LOG=/tmp/e6_master.log
echo "=== [E6 master pipeline] start $(date) ===" | tee "$PROGRESS"

#-----------------------------------------------------------------------------
# Helper: only run if calibration_<tag> doesn't already have D_wrong / Q_wrong
#-----------------------------------------------------------------------------
need_calibrate_subspace() {
    [[ ! -f "outputs/e6_steering/$MODEL/calibration_${TAG}/D_wrong.pt" ]]
}
need_calibrate_qao() {
    [[ ! -f "outputs/e6_steering/$MODEL/calibration_${TAG}/Q_wrong.pt" ]]
}

#-----------------------------------------------------------------------------
# S0a: calibrate-subspace
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S0a: calibrate-subspace TallyQA E5e N=$N_CALIB ===" | tee -a "$PROGRESS"
if need_calibrate_subspace; then
    echo "[S0a START] $(date)" | tee -a "$PROGRESS"
    uv run python scripts/e6_steering_vector.py \
        --phase calibrate-subspace \
        --e5c-run-dir "$E5C_DIR" \
        --model "$MODEL" --hf-model "$HF_MODEL" \
        --predictions-path "$TALLY_PREDS_E5E" \
        --dataset-tag "$TAG" \
        --max-calibrate-pairs "$N_CALIB" \
        --config configs/experiment_e5e_tallyqa_full.yaml \
        > "$LOG" 2>&1
    echo "[S0a DONE] $(date)" | tee -a "$PROGRESS"
else
    echo "[S0a SKIP] D_wrong.pt already exists" | tee -a "$PROGRESS"
fi

#-----------------------------------------------------------------------------
# S0b: calibrate-qao
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S0b: calibrate-qao TallyQA E5e N=$N_CALIB ===" | tee -a "$PROGRESS"
if need_calibrate_qao; then
    echo "[S0b START] $(date)" | tee -a "$PROGRESS"
    uv run python scripts/e6_query_adaptive_offset.py \
        --phase calibrate-qao \
        --model "$MODEL" --hf-model "$HF_MODEL" \
        --predictions-path "$TALLY_PREDS_E5E" \
        --dataset-tag "$TAG" \
        --max-calibrate-pairs "$N_CALIB" \
        --config configs/experiment_e5e_tallyqa_full.yaml \
        >> "$LOG" 2>&1
    echo "[S0b DONE] $(date)" | tee -a "$PROGRESS"
else
    echo "[S0b SKIP] Q_wrong.pt already exists" | tee -a "$PROGRESS"
fi

#-----------------------------------------------------------------------------
# S0.5: peak-layer selection
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S0.5: pick top-$TOP_K peak layers ===" | tee -a "$PROGRESS"
uv run python scripts/e6_pick_peak_layers.py \
    --model "$MODEL" --tag "$TAG" --top-k "$TOP_K" \
    --out "$PEAK_JSON" \
    >> "$LOG" 2>&1
PEAK_LAYERS=$(uv run python -c "
import json
print(','.join(str(L) for L in json.load(open('$PEAK_JSON'))['top_layers']))
")
echo "[S0.5 DONE] peak layers: $PEAK_LAYERS  $(date)" | tee -a "$PROGRESS"

#-----------------------------------------------------------------------------
# S1: LEACE Tally-only
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S1: LEACE Tally-only ===" | tee -a "$PROGRESS"

echo "[S1a calibrate-leace] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase calibrate-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    >> "$LOG" 2>&1
echo "[S1a DONE] $(date)" | tee -a "$PROGRESS"

echo "[S1b LEACE sweep Tally n=$N_SWEEP layers=$PEAK_LAYERS] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$TALLY_PREDS_E5E" \
    --dataset-tag tally \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    --max-sweep-sids "$N_SWEEP" --out-tag "$TAG" \
    --layers "$PEAK_LAYERS" \
    --config configs/experiment_e5e_tallyqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S1b DONE] $(date)" | tee -a "$PROGRESS"

echo "[S1c LEACE sweep ChartQA n=$N_SWEEP] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$CHARTQA_PREDS_E5E" \
    --dataset-tag chartqa \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    --max-sweep-sids "$N_SWEEP" --out-tag "$TAG" \
    --layers "$PEAK_LAYERS" \
    --config configs/experiment_e5e_chartqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S1c DONE] $(date)" | tee -a "$PROGRESS"

for d in tally chartqa; do
  for rule in one_sided two_sided; do
    uv run python scripts/analyze_e6_methods.py \
        --sweep-dir "outputs/e6_steering/$MODEL/sweep_leace_${d}_${TAG}_pooled" \
        --em-rule "$rule" >> "$LOG" 2>&1 || true
  done
done
echo "[S1 ALL DONE] $(date)" | tee -a "$PROGRESS"

#-----------------------------------------------------------------------------
# S2: Subspace Method 1 Tally-only
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S2: Subspace Tally-only ===" | tee -a "$PROGRESS"

echo "[S2a compute-subspace] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_compute_subspace.py \
    --model "$MODEL" --tags "$TAG" --scope "$TAG" --K-max 16 \
    >> "$LOG" 2>&1
SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"
echo "[S2a DONE]  $(date)" | tee -a "$PROGRESS"

echo "[S2b Subspace sweep Tally n=$N_SWEEP layers=$PEAK_LAYERS] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_steering_vector.py \
    --phase sweep-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --e5c-run-dir "$E5C_DIR" \
    --predictions-path "$TALLY_PREDS_E5E" \
    --dataset-tag tally \
    --subspace-path "$SUBSPACE_PT" --subspace-scope "$TAG" \
    --max-samples "$N_SWEEP" \
    --sweep-layers "$PEAK_LAYERS" \
    --config configs/experiment_e5e_tallyqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S2b DONE] $(date)" | tee -a "$PROGRESS"

echo "[S2c Subspace sweep ChartQA n=$N_SWEEP] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_steering_vector.py \
    --phase sweep-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --e5c-run-dir "$E5C_DIR" \
    --predictions-path "$CHARTQA_PREDS_E5E" \
    --dataset-tag chartqa \
    --subspace-path "$SUBSPACE_PT" --subspace-scope "$TAG" \
    --max-samples "$N_SWEEP" \
    --sweep-layers "$PEAK_LAYERS" \
    --config configs/experiment_e5e_chartqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S2c DONE] $(date)" | tee -a "$PROGRESS"

uv run python scripts/analyze_e6_subspace.py \
    --sweep-dir "outputs/e6_steering/$MODEL/sweep_subspace_tally_${TAG}" \
    >> "$LOG" 2>&1 || true
uv run python scripts/analyze_e6_subspace.py \
    --sweep-dir "outputs/e6_steering/$MODEL/sweep_subspace_chartqa_${TAG}" \
    >> "$LOG" 2>&1 || true
echo "[S2 ALL DONE] $(date)" | tee -a "$PROGRESS"

#-----------------------------------------------------------------------------
# S3: DPO Method 3 with case-by-case rejected, Tally-only training
#-----------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S3: DPO Method 3 v2 ===" | tee -a "$PROGRESS"

echo "[S3a build-pairs case_by_case Tally-only] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase build-pairs --model "$MODEL" \
    --calib-tags tally \
    --rejected-mode case_by_case \
    --out-tag "v2_tally_only" \
    --max-pairs 5000 \
    >> "$LOG" 2>&1
echo "[S3a DONE] $(date)" | tee -a "$PROGRESS"

echo "[S3b train-dpo] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase train-dpo --model "$MODEL" --hf-model "$HF_MODEL" \
    --calib-tags tally --out-tag "v2_tally_only" \
    --adapter-dir "outputs/e6_dpo/$MODEL/adapter_v2_tally_only" \
    >> "$LOG" 2>&1
echo "[S3b DONE] $(date)" | tee -a "$PROGRESS"

for d in tally chartqa; do
  preds_var="${d^^}_PREDS_E5E"
  echo "[S3c sweep-adapter $d n=$N_SWEEP] $(date)" | tee -a "$PROGRESS"
  uv run python scripts/e6_dpo_lora.py \
      --phase sweep-adapter --model "$MODEL" --hf-model "$HF_MODEL" \
      --predictions-path "${!preds_var}" \
      --dataset-tag "$d" \
      --max-sweep-sids "$N_SWEEP" \
      --adapter-dir "outputs/e6_dpo/$MODEL/adapter_v2_tally_only" \
      --calib-tags tally \
      --config "configs/experiment_e5e_${d}qa_full.yaml" \
      >> "$LOG" 2>&1 || echo "[S3c $d FAILED]" | tee -a "$PROGRESS"
  echo "[S3c $d DONE] $(date)" | tee -a "$PROGRESS"
done
echo "[S3 ALL DONE] $(date)" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [E6 master pipeline] FINISHED $(date) ===" | tee -a "$PROGRESS"
