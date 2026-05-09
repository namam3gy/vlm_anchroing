#!/usr/bin/env bash
# P0-1 chain resume: skip B1/B2 (already done), start from B3a.
# Use after run_p0_1_chain.sh aborts mid-chain on a fixable bug.

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing/.claude/worktrees/phase5+p0-1-gamma-beta-bridge

PLOTQA_PRED=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct/20260510-005948/predictions.jsonl
INFOVQA_PRED=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_infovqa/qwen3-vl-8b-instruct/20260510-023910/predictions.jsonl

E6_LOGS=outputs/e6_steering/qwen3-vl-8b-instruct
mkdir -p "$E6_LOGS"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === P0-1 resume from B3a ==="
echo "[$(ts)]    PlotQA preds: $PLOTQA_PRED"
echo "[$(ts)]    InfoVQA preds: $INFOVQA_PRED"

if [ ! -f "$PLOTQA_PRED" ]; then
    echo "[$(ts)] FATAL: PlotQA preds missing"
    exit 1
fi
if [ ! -f "$INFOVQA_PRED" ]; then
    echo "[$(ts)] FATAL: InfoVQA preds missing"
    exit 1
fi

# Compute run_dir parents (e5c-run-dir = parent of predictions.jsonl)
PLOTQA_RUN_DIR=$(dirname "$PLOTQA_PRED")
INFOVQA_RUN_DIR=$(dirname "$INFOVQA_PRED")

# ---------------------------------------------------------------------------
# B3a: calibrate-subspace plotqa  (single-GPU direct invocation)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B3a: calibrate-subspace plotqa ==="
uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model qwen3-vl-8b-instruct \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --e5c-run-dir "$PLOTQA_RUN_DIR" \
    --predictions-path "$PLOTQA_PRED" \
    --config configs/p0_1_calibration_qwen3vl_plotqa.yaml \
    --dataset-tag plotqa \
    --max-calibrate-pairs 5000 \
    > "$E6_LOGS/_calibrate_plotqa.log" 2>&1
echo "[$(ts)] B3a done"

# ---------------------------------------------------------------------------
# B3b: calibrate-subspace infovqa  (single-GPU direct invocation)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B3b: calibrate-subspace infovqa ==="
uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model qwen3-vl-8b-instruct \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --e5c-run-dir "$INFOVQA_RUN_DIR" \
    --predictions-path "$INFOVQA_PRED" \
    --config configs/p0_1_calibration_qwen3vl_infovqa.yaml \
    --dataset-tag infovqa \
    --max-calibrate-pairs 5000 \
    > "$E6_LOGS/_calibrate_infovqa.log" 2>&1
echo "[$(ts)] B3b done"

# ---------------------------------------------------------------------------
# B4: SVD
# ---------------------------------------------------------------------------
echo "[$(ts)] === B4: SVD ==="
mkdir -p "$E6_LOGS/_subspace"
uv run python scripts/e6_compute_subspace.py \
    --model qwen3-vl-8b-instruct \
    --scope plotqa_infovqa_pooled \
    --tags plotqa,infovqa \
    --K-max 16 \
    > "$E6_LOGS/_subspace/_svd.log" 2>&1
echo "[$(ts)] B4 done"

SUBSPACE_PT=outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt
if [ ! -f "$SUBSPACE_PT" ]; then
    echo "[$(ts)] FATAL: $SUBSPACE_PT not found after SVD"
    exit 3
fi
echo "[$(ts)] V_K artifact: $SUBSPACE_PT"

# ---------------------------------------------------------------------------
# C2-smoke
# ---------------------------------------------------------------------------
echo "[$(ts)] === C2-smoke: bridge inference n=5 (Instruct) ==="
SMOKE_TAG="smoke_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --max-samples 5 \
    --output-tag "$SMOKE_TAG" \
    > outputs/gamma_beta_bridge/_smoke.log 2>&1
SMOKE_JSONL=outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${SMOKE_TAG}/amplitude_per_trace.jsonl
if [ ! -f "$SMOKE_JSONL" ]; then
    echo "[$(ts)] FATAL: smoke jsonl missing at $SMOKE_JSONL"
    exit 4
fi
N_SMOKE=$(wc -l < "$SMOKE_JSONL")
echo "[$(ts)] C2-smoke done — $N_SMOKE records"
if [ "$N_SMOKE" -lt 8 ]; then
    echo "[$(ts)] FATAL: expected ≥8 smoke records (5 sids × 2 cond), got $N_SMOKE"
    exit 5
fi

# ---------------------------------------------------------------------------
# C3: full Instruct bridge
# ---------------------------------------------------------------------------
echo "[$(ts)] === C3: Instruct bridge inference (full) ==="
INSTRUCT_TAG="instruct_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --output-tag "$INSTRUCT_TAG" \
    > outputs/gamma_beta_bridge/_instruct.log 2>&1
echo "[$(ts)] C3 done"

# ---------------------------------------------------------------------------
# C4: full Thinking bridge (long)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C4: Thinking bridge inference (full, long) ==="
THINKING_TAG="thinking_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-thinking \
    --output-tag "$THINKING_TAG" \
    > outputs/gamma_beta_bridge/_thinking.log 2>&1
echo "[$(ts)] C4 done"

echo "[$(ts)] === P0-1 chain resume COMPLETE ==="
echo "  V_K subspace: $SUBSPACE_PT"
echo "  Instruct amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${INSTRUCT_TAG}/amplitude_per_trace.jsonl"
echo "  Thinking amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/${THINKING_TAG}/amplitude_per_trace.jsonl"
