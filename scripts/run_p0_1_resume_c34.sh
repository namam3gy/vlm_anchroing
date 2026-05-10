#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing/.claude/worktrees/phase5+p0-1-gamma-beta-bridge
mkdir -p outputs/gamma_beta_bridge
ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === Phase C3 + C4 resume ==="

INSTRUCT_TAG="instruct_$(date +%Y%m%d-%H%M%S)"
echo "[$(ts)] === C3: Instruct bridge inference (full) ==="
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --output-tag "$INSTRUCT_TAG" \
    > outputs/gamma_beta_bridge/_instruct.log 2>&1
echo "[$(ts)] C3 done"

THINKING_TAG="thinking_$(date +%Y%m%d-%H%M%S)"
echo "[$(ts)] === C4: Thinking bridge inference (full, long) ==="
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-thinking \
    --output-tag "$THINKING_TAG" \
    > outputs/gamma_beta_bridge/_thinking.log 2>&1
echo "[$(ts)] C4 done"

echo "[$(ts)] === Phase C3+C4 COMPLETE ==="
echo "  Instruct: outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${INSTRUCT_TAG}/amplitude_per_trace.jsonl"
echo "  Thinking: outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/${THINKING_TAG}/amplitude_per_trace.jsonl"
