#!/usr/bin/env bash
# Run E4 Phase 2 full validation in sequence on multiple models.
# Each model launches at the optimal strength chosen by Phase 1
# (per outputs/e4_mitigation/_summary/chosen_strength.json).
#
# Resumability: each model writes to outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl;
# re-running this script picks up from completed (sample_instance_id, condition, strength) keys.
#
# GPU: pinned to GPU 0 (per session-level user instruction).
#
# Usage:
#   bash scripts/run_e4_phase2_chain.sh > outputs/e4_mitigation/phase2_chain.log 2>&1 &

# Note: do NOT use `set -e` here — if one model errors out (OOM, kernel
# panic, anything), we still want the chain to attempt the remaining
# models. Resumability handles a re-launch after the session ends.
set -uo pipefail

cd "$(dirname "$0")/.."

# Read chosen strengths from Phase 1 JSON
S_LLAVA=$(uv run python -c "import json; d=json.load(open('outputs/e4_mitigation/_summary/chosen_strength.json')); print(d['llava-1.5-7b'])")
S_CONVLLAVA=$(uv run python -c "import json; d=json.load(open('outputs/e4_mitigation/_summary/chosen_strength.json')); print(d['convllava-7b'])")
S_INTERNVL3=$(uv run python -c "import json; d=json.load(open('outputs/e4_mitigation/_summary/chosen_strength.json')); print(d['internvl3-8b'])")

echo "[chain] starting Phase 2 chain at $(date)"
echo "[chain] strengths: llava=${S_LLAVA}, convllava=${S_CONVLLAVA}, internvl3=${S_INTERNVL3}"

# Skip any model whose chosen strength is "None" (no valid s* from Phase 1)
run_phase2() {
    local model="$1"
    local hf="$2"
    local strength="$3"
    if [[ "${strength}" == "None" || -z "${strength}" ]]; then
        echo "[chain] SKIP ${model}: no valid s* from Phase 1"
        return 0
    fi
    echo "[chain] >>> launching Phase 2 for ${model} at strength=${strength}, $(date)"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
        --model "${model}" --hf-model "${hf}" \
        --phase full --strength "${strength}" \
        || echo "[chain] WARN ${model} exited non-zero — continuing chain"
    echo "[chain] <<< ${model} Phase 2 finished at $(date)"
}

run_phase2 llava-1.5-7b llava-hf/llava-1.5-7b-hf "${S_LLAVA}"
run_phase2 convllava-7b ConvLLaVA/ConvLLaVA-sft-1536 "${S_CONVLLAVA}"
run_phase2 internvl3-8b OpenGVLab/InternVL3-8B-hf "${S_INTERNVL3}"

echo "[chain] all done at $(date)"
