#!/usr/bin/env bash
# H1 Stage 4b.1 — re-calibrate OneVision subspace on H1 raw-prompt
# PlotQA + InfoVQA predictions, then pool → SVD K=16.
#
# Inputs:  outputs/paper2/cross_model_cross_dataset/predictions/{plotqa,
#          infographicvqa}/llava-onevision-qwen2-7b-ov/predictions.jsonl
# Outputs: outputs/e6_steering/llava-onevision-qwen2-7b-ov/
#            calibration_plotqa_h1/{D_wrong,D_all}.pt
#            calibration_infographicvqa_h1/{D_wrong,D_all}.pt
#            _subspace/subspace_plotqa_infovqa_pooled_h1_K16.pt
#            _subspace/singular_values_plotqa_infovqa_pooled_h1.csv
#
# Parallelises the two calibration runs on GPUs 6 & 7 (typically idle
# during §5.1 attention extraction on GPUs 0-5). Runs ~30-40 min wall.
#
# Usage (from worktree root):
#   bash scripts/_run_stage4b_calibrate.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MAIN=$(git rev-parse --git-common-dir | xargs realpath | xargs dirname)
PY=.venv/bin/python
MODEL=llava-onevision-qwen2-7b-ov
HF_MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
PRED_ROOT=$MAIN/outputs/paper2/cross_model_cross_dataset/predictions
LOG_DIR=$MAIN/outputs/paper2/section_5_e6_steering/_logs
mkdir -p "$LOG_DIR"

echo "=== Stage 4b.1 · Calibrate OneVision subspace on H1 PlotQA + InfoVQA ==="
echo "  PlotQA  → GPU 6"
echo "  InfoVQA → GPU 7"
echo ""

# PlotQA calibration (heavier — 5000 wrong-base eligible)
(
  CUDA_VISIBLE_DEVICES=6 $PY scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$PRED_ROOT/plotqa/$MODEL/predictions.jsonl" \
    --dataset-tag plotqa_h1 \
    --e5c-run-dir outputs/paper2/cross_model_cross_dataset
) > "$LOG_DIR/calibrate_plotqa_h1.log" 2>&1 &
PID_PLOTQA=$!
echo "  spawned PlotQA calibrate, PID=$PID_PLOTQA"

# InfoVQA calibration (lighter — 1147 wrong-base)
(
  CUDA_VISIBLE_DEVICES=7 $PY scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" --hf-model "$HF_MODEL" \
    --predictions-path "$PRED_ROOT/infographicvqa/$MODEL/predictions.jsonl" \
    --dataset-tag infographicvqa_h1 \
    --e5c-run-dir outputs/paper2/cross_model_cross_dataset
) > "$LOG_DIR/calibrate_infographicvqa_h1.log" 2>&1 &
PID_INFOVQA=$!
echo "  spawned InfoVQA calibrate, PID=$PID_INFOVQA"

echo ""
echo "  waiting for both calibrations to complete …"
wait $PID_PLOTQA && echo "  PlotQA  done" || { echo "  PlotQA  FAILED"; exit 1; }
wait $PID_INFOVQA && echo "  InfoVQA done" || { echo "  InfoVQA FAILED"; exit 1; }

echo ""
echo "=== Stage 4b.2 · Pool + SVD → K=16 subspace ==="
$PY scripts/e6_compute_subspace.py \
  --model "$MODEL" \
  --tags plotqa_h1,infographicvqa_h1 \
  --scope plotqa_infovqa_pooled_h1 \
  --K-max 16

SUBSPACE_PATH=$MAIN/outputs/e6_steering/$MODEL/_subspace/subspace_plotqa_infovqa_pooled_h1_K16.pt
echo ""
if [ -f "$SUBSPACE_PATH" ]; then
  ls -la "$SUBSPACE_PATH"
  echo "✅  Stage 4b.1/4b.2 complete. Next: bash scripts/_run_stage4b_sweep.sh"
else
  echo "❌  Subspace tensor missing: $SUBSPACE_PATH"
  exit 1
fi
