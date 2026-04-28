#!/usr/bin/env bash
# Run E5b/E5c stratified cross-model expansion in sequence on GPU 1.
# Closes the §6.3 §3.1 cross-model expansion gap (qwen2.5-vl-7b and
# gemma3-27b on E5c VQAv2 + TallyQA) once the in-flight TallyQA gemma3-27b
# E5e job releases GPU 1.
#
# Configs (already pre-staged 2026-04-28):
#   configs/experiment_e5c_vqa.yaml    — VQAv2,  1000 base × 12 conditions
#   configs/experiment_e5c_tally.yaml  — TallyQA, 1000 base × 12 conditions
#
# Order:
#   1. qwen2.5-vl-7b on E5c VQAv2     (~2-4h on H200)
#   2. qwen2.5-vl-7b on E5c TallyQA   (~2-4h)
#   3. gemma3-27b-it on E5c VQAv2     (~5-8h, full n=1000 base)
#   4. gemma3-27b-it on E5c TallyQA   (~5-8h, full n=1000 base)
#
# Each invocation writes to outputs/experiment_e5c_{vqa,tally}/<model>/<ts>/.
# llava-next-interleaved-7b is already done; --models filter skips it.
#
# GPU: pinned to GPU 1. Do NOT launch while the gemma3-27b TallyQA E5e
# run on GPU 1 is still active — check `nvidia-smi --query-compute-apps`
# first.
#
# Resumability: run_experiment.py rewrites the timestamped output dir
# from scratch each invocation; if a stage fails, drop the partial dir
# and re-run the remaining stages by editing the order array below.
#
# Usage (pre-flight check + launch):
#   nvidia-smi --query-compute-apps=pid,process_name --format=csv
#   # confirm no vlm_anchoring python on GPU 1 — only then:
#   nohup bash scripts/run_e5b_e5c_cross_model_chain.sh \
#       > /tmp/e5c_cross_model_chain.log 2>&1 &
#   tail -f /tmp/e5c_cross_model_chain.log

set -uo pipefail

cd "$(dirname "$0")/.."

echo "[chain] starting E5c cross-model chain at $(date)"
echo "[chain] GPU 1 will be saturated for ~14-20h"

run_e5c() {
    local config="$1"
    local model="$2"
    echo "[chain] >>> launching ${config} × ${model}, $(date)"
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_experiment.py \
        --config "${config}" \
        --models "${model}" \
        || echo "[chain] WARN ${config} ${model} exited non-zero — continuing chain"
    echo "[chain] <<< ${config} × ${model} finished at $(date)"
}

# 7B model first (~2-4h each, fills paper §5 cross-model gap)
run_e5c configs/experiment_e5c_vqa.yaml   qwen2.5-vl-7b-instruct
run_e5c configs/experiment_e5c_tally.yaml qwen2.5-vl-7b-instruct

# 27B model second (heavier — only launch if there is overnight time left)
run_e5c configs/experiment_e5c_vqa.yaml   gemma3-27b-it
run_e5c configs/experiment_e5c_tally.yaml gemma3-27b-it

echo "[chain] all done at $(date)"
