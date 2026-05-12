#!/bin/bash
# LEACE re-calibration sweep — 5 datasets sequentially on OneVision Main, L=26, α∈{0.5,1.0,2.0}.
# Eraser tag: plotqa_infovqa_recal (h^b, h^b+D class definitions, PlotQA+InfoVQA pool).
set -e
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES=0

declare -a DATASETS=(
  "chartqa|outputs/experiment_e5e_chartqa_full/llava-onevision-qwen2-7b-ov/20260502-211028/predictions.jsonl|300"
  "mathvista|outputs/experiment_e5e_mathvista_full/llava-onevision-qwen2-7b-ov/20260502-212440/predictions.jsonl|300"
  "infographicvqa|outputs/experiment_e5b_5strat_infographicvqa_onevision/llava-onevision-qwen2-7b-ov/20260504-070829/predictions.jsonl|600"
  "plotqa|outputs/experiment_e5b_5strat_plotqa_onevision/llava-onevision-qwen2-7b-ov/20260504-075037/predictions.jsonl|2500"
  "tallyqa|outputs/experiment_e5e_tallyqa_full/llava-onevision-qwen2-7b-ov/20260502-083926/predictions.jsonl|2500"
)

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r DS_TAG PRED_PATH MAX_N <<< "$entry"
  echo "[$(date -u +%FT%TZ)] === ${DS_TAG} (max=${MAX_N}) ==="
  uv run python scripts/e6_leace.py \
    --phase sweep-leace \
    --model llava-onevision-qwen2-7b-ov \
    --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --predictions-path "${PRED_PATH}" \
    --dataset-tag "${DS_TAG}" \
    --eraser-tag plotqa_infovqa_recal \
    --out-tag recal \
    --layers 26 \
    --alphas 0.5,1.0,2.0 \
    --max-sweep-sids "${MAX_N}" \
    --max-new-tokens 8 \
    2>&1 | tee -a "outputs/_logs/leace_recal_sweep_${DS_TAG}.log"
  echo "[$(date -u +%FT%TZ)] === ${DS_TAG} done ==="
done
echo "[$(date -u +%FT%TZ)] ALL DATASETS DONE"
