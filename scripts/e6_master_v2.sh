#!/usr/bin/env bash
# E6 Tally-only rerun — v3 master pipeline (LEACE full re-sweep with sorted sids)
# Bug fix: LEACE sid sampling was non-deterministic (set iteration depends on
# PYTHONHASHSEED). e6_leace.py now sorts eligible sids before [:max_sweep_sids].
# Old random-sample-A predictions backed up as predictions.bak_v1_*.jsonl;
# this run produces a clean deterministic-sample-B from scratch.
# Layer set: L 7, 16, 24, 30, 31 (early/mid/late-mid/peak/top-2-by-norm).
# - LEACE: full re-sweep all 5 layers from scratch
# - Subspace: full re-sweep with new layers (deterministic by dict iteration)
# - DPO: unchanged
set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

export CUDA_VISIBLE_DEVICES=1
MODEL=llava-next-interleaved-7b
HF_MODEL=llava-hf/llava-interleave-qwen-7b-hf
TAG=tally_e5e_n5k
N_SWEEP=500

LEACE_LAYERS_FULL="7,16,24,30,31"   # full re-sweep (random-sample-A backed up)
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
echo "=== S1 LEACE full re-sweep L=$LEACE_LAYERS_FULL ===" | tee -a "$PROGRESS"

echo "[S1'b sweep TallyQA] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$TALLY_PREDS_E5E" \
    --dataset-tag tally \
    --calib-tags "$TAG" --eraser-tag "$TAG" \
    --max-sweep-sids "$N_SWEEP" --out-tag "$TAG" \
    --layers "$LEACE_LAYERS_FULL" \
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
    --layers "$LEACE_LAYERS_FULL" \
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
echo "=== S3 DPO mix_synthetic (Tally+ChartQA+VQA, image-id split) ===" | tee -a "$PROGRESS"

DPO_TAG="v2_mix_synthetic"
SPLIT_MAP="outputs/e6_dpo/$MODEL/split_map_${DPO_TAG}.json"

echo "[S3a build-pairs] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase build-pairs --model "$MODEL" \
    --calib-tags tally,chartqa,vqa \
    --rejected-mode mix_synthetic \
    --train-frac 0.7 \
    --synth-ratios "tally:1,chartqa:5,vqa:5" \
    --out-tag "$DPO_TAG" \
    --max-pairs 50000 \
    >> "$LOG" 2>&1
echo "[S3a done] $(date)" | tee -a "$PROGRESS"

echo "[S3b train-dpo] $(date)" | tee -a "$PROGRESS"
uv run python scripts/e6_dpo_lora.py \
    --phase train-dpo --model "$MODEL" --hf-model "$HF_MODEL" \
    --calib-tags tally,chartqa,vqa --out-tag "$DPO_TAG" \
    --adapter-dir "outputs/e6_dpo/$MODEL/adapter_${DPO_TAG}" \
    >> "$LOG" 2>&1
echo "[S3b done] $(date)" | tee -a "$PROGRESS"

# S3c: sweep on eval-split sids only (no train/test leakage)
for d in tally chartqa; do
  preds_var="${d^^}_PREDS_E5E"
  echo "[S3c sweep-adapter $d eval-split-only] $(date)" | tee -a "$PROGRESS"
  uv run python scripts/e6_dpo_lora.py \
      --phase sweep-adapter --model "$MODEL" --hf-model "$HF_MODEL" \
      --predictions-path "${!preds_var}" \
      --dataset-tag "$d" \
      --max-sweep-sids "$N_SWEEP" \
      --adapter-dir "outputs/e6_dpo/$MODEL/adapter_${DPO_TAG}" \
      --split-map "$SPLIT_MAP" \
      --calib-tags tally,chartqa,vqa \
      --config "configs/experiment_e5e_${d}qa_full.yaml" \
      >> "$LOG" 2>&1 || echo "[S3c $d FAILED]" | tee -a "$PROGRESS"
  echo "[S3c $d done] $(date)" | tee -a "$PROGRESS"
done
echo "[S3 ALL DONE] $(date)" | tee -a "$PROGRESS"

echo "" | tee -a "$PROGRESS"
echo "=== [E6 master v2] FINISHED $(date) ===" | tee -a "$PROGRESS"
