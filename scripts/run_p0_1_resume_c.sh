#!/usr/bin/env bash
# P0-1 Phase C resume: V_K is ready, run bridge inference (Instruct + Thinking).

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing/.claude/worktrees/phase5+p0-1-gamma-beta-bridge

mkdir -p outputs/gamma_beta_bridge
ts() { date +'%Y-%m-%d %H:%M:%S'; }

SUBSPACE_PT=outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt
if [ ! -f "$SUBSPACE_PT" ]; then
    echo "[$(ts)] FATAL: $SUBSPACE_PT not found"
    exit 1
fi
echo "[$(ts)] === P0-1 resume Phase C === V_K: $SUBSPACE_PT"

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
# C4: full Thinking bridge (long, ~12 h)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C4: Thinking bridge inference (full, long) ==="
THINKING_TAG="thinking_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-thinking \
    --output-tag "$THINKING_TAG" \
    > outputs/gamma_beta_bridge/_thinking.log 2>&1
echo "[$(ts)] C4 done"

echo "[$(ts)] === Phase C COMPLETE ==="
echo "  Instruct amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${INSTRUCT_TAG}/amplitude_per_trace.jsonl"
echo "  Thinking amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/${THINKING_TAG}/amplitude_per_trace.jsonl"
