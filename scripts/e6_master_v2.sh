#!/usr/bin/env bash
# E6 Tally-only rerun — v2 master pipeline (LEACE supplement + Subspace + DPO)
# New peak-layer set: L 7, 16, 24, 30, 31 (replaces 27,28,29,30,31).
# - LEACE 30, 31 already done → only sweep L 7, 16, 24 (append to existing dir)
# - Subspace partial output (L27-L31) backed up; full re-sweep with new layers
# - DPO unchanged
set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

export CUDA_VISIBLE_DEVICES=1
MODEL=llava-next-interleaved-7b
HF_MODEL=llava-hf/llava-interleave-qwen-7b-hf
TAG=tally_e5e_n5k
N_SWEEP=500

LEACE_LAYERS_NEW="7,16,24"          # supplement only (30, 31 already done)
LEACE_LAYERS_FULL="7,16,24,30,31"
SUBSPACE_LAYERS="7,16,24,30,31"

TALLY_PREDS_E5E="outputs/experiment_e5e_tallyqa_full/$MODEL/20260427-171240/predictions.jsonl"
CHARTQA_PREDS_E5E="outputs/experiment_e5e_chartqa_full/$MODEL/20260427-171240/predictions.jsonl"
E5C_DIR="outputs/experiment_e5c_vqa/$MODEL/20260427-123331"

PROGRESS=/tmp/e6_master_v2_progress.log
LOG=/tmp/e6_master_v2.log
echo "=== [E6 master v2] start $(date) ===" | tee "$PROGRESS"

# ---------------------------------------------------------------------------
# S1' LEACE supplement: sweep L=7,16,24 on Tally + ChartQA (append to existing)
# ---------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S1' LEACE supplement L=$LEACE_LAYERS_NEW ===" | tee -a "$PROGRESS"

echo "[S1'b sweep TallyQA] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$TALLY_PREDS_E5E" \
    --dataset-tag tally \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    --max-sweep-sids "$N_SWEEP" --out-tag "$TAG" \
    --layers "$LEACE_LAYERS_NEW" \
    --config configs/experiment_e5e_tallyqa_full.yaml \
    > "$LOG" 2>&1
echo "[S1'b done] $(date)" | tee -a "$PROGRESS"

echo "[S1'c sweep ChartQA] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$CHARTQA_PREDS_E5E" \
    --dataset-tag chartqa \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    --max-sweep-sids "$N_SWEEP" --out-tag "$TAG" \
    --layers "$LEACE_LAYERS_NEW" \
    --config configs/experiment_e5e_chartqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S1'c done] $(date)" | tee -a "$PROGRESS"

# Re-analyze (full set: includes existing L27-31 + new L7/16/24)
for d in tally chartqa; do
  for rule in one_sided two_sided; do
    uv run python scripts/analyze_e6_methods.py \
        --sweep-dir "outputs/e6_steering/$MODEL/sweep_leace_${d}_${TAG}_pooled" \
        --em-rule "$rule" >> "$LOG" 2>&1 || true
  done
done
echo "[S1' analyze done] $(date)" | tee -a "$PROGRESS"

# ---------------------------------------------------------------------------
# S2 Subspace Tally-only (full re-sweep with new layers)
# ---------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S2 Subspace Tally-only L=$SUBSPACE_LAYERS ===" | tee -a "$PROGRESS"

SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${TAG}_K16.pt"

echo "[S2b sweep TallyQA] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_steering_vector.py \
    --phase sweep-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --e5c-run-dir "$E5C_DIR" \
    --predictions-path "$TALLY_PREDS_E5E" \
    --dataset-tag tally \
    --subspace-path "$SUBSPACE_PT" --subspace-scope "$TAG" \
    --max-samples "$N_SWEEP" \
    --sweep-layers "$SUBSPACE_LAYERS" \
    --config configs/experiment_e5e_tallyqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S2b done] $(date)" | tee -a "$PROGRESS"

echo "[S2c sweep ChartQA] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_steering_vector.py \
    --phase sweep-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --e5c-run-dir "$E5C_DIR" \
    --predictions-path "$CHARTQA_PREDS_E5E" \
    --dataset-tag chartqa \
    --subspace-path "$SUBSPACE_PT" --subspace-scope "$TAG" \
    --max-samples "$N_SWEEP" \
    --sweep-layers "$SUBSPACE_LAYERS" \
    --config configs/experiment_e5e_chartqa_full.yaml \
    >> "$LOG" 2>&1
echo "[S2c done] $(date)" | tee -a "$PROGRESS"

uv run python scripts/analyze_e6_subspace.py \
    --sweep-dir "outputs/e6_steering/$MODEL/sweep_subspace_tally_${TAG}" \
    >> "$LOG" 2>&1 || true
uv run python scripts/analyze_e6_subspace.py \
    --sweep-dir "outputs/e6_steering/$MODEL/sweep_subspace_chartqa_${TAG}" \
    >> "$LOG" 2>&1 || true
echo "[S2 ALL DONE] $(date)" | tee -a "$PROGRESS"

# ---------------------------------------------------------------------------
# S3 DPO Tally-only with case_by_case rejected
# ---------------------------------------------------------------------------
echo "" | tee -a "$PROGRESS"
echo "=== S3 DPO Tally-only case_by_case ===" | tee -a "$PROGRESS"

echo "[S3a build-pairs] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase build-pairs --model "$MODEL" \
    --calib-tags tally \
    --rejected-mode case_by_case \
    --out-tag "v2_tally_only" \
    --max-pairs 5000 \
    >> "$LOG" 2>&1
echo "[S3a done] $(date)" | tee -a "$PROGRESS"

echo "[S3b train-dpo] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase train-dpo --model "$MODEL" --hf-model "$HF_MODEL" \
    --calib-tags tally --out-tag "v2_tally_only" \
    --adapter-dir "outputs/e6_dpo/$MODEL/adapter_v2_tally_only" \
    >> "$LOG" 2>&1
echo "[S3b done] $(date)" | tee -a "$PROGRESS"

for d in tally chartqa; do
  preds_var="${d^^}_PREDS_E5E"
  echo "[S3c sweep-adapter $d] $(date)" | tee -a "$PROGRESS"
  uv run python scripts/e6_dpo_lora.py \
      --phase sweep-adapter --model "$MODEL" --hf-model "$HF_MODEL" \
      --predictions-path "${!preds_var}" \
      --dataset-tag "$d" \
      --max-sweep-sids "$N_SWEEP" \
      --adapter-dir "outputs/e6_dpo/$MODEL/adapter_v2_tally_only" \
      --calib-tags tally \
      --config "configs/experiment_e5e_${d}qa_full.yaml" \
      >> "$LOG" 2>&1 || echo "[S3c $d FAILED]" | tee -a "$PROGRESS"
  echo "[S3c $d done] $(date)" | tee -a "$PROGRESS"
done
echo "[S3 ALL DONE] $(date)" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [E6 master v2] FINISHED $(date) ===" | tee -a "$PROGRESS"
