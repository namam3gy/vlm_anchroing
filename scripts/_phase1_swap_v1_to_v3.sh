#!/usr/bin/env bash
# Phase 1 P0 v3 swap-and-launch:
# 1. Poll v1 log for Stage 2 boundary
# 2. SIGTERM v1 + children
# 3. Wait for GPUs to free
# 4. 5-sample smoke test for OneVision §7.1-7.3 AnyRes code path
# 5. If smoke passes, launch v3 orchestrator (detached)
#
# Designed to run via nohup, fully detached from interactive session.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/swap.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

note "==== swap watcher armed (PID $$) ===="

# ─── Step 1: poll v1 log for Stage 2 marker ──────────────────────────────────
note "polling logs/phase1/post_baseline_parallel.log for Stage 2 boundary..."
WAIT_LOG=logs/phase1/post_baseline_parallel.log
while ! grep -q "Stage 2: chart/math" "$WAIT_LOG" 2>/dev/null; do
  sleep 10
done
note "v1 reached Stage 2 boundary"

# ─── Step 2: SIGTERM v1 + children ───────────────────────────────────────────
note "killing v1 + child python procs"
pkill -TERM -f "_phase1_post_baseline_parallel.sh" 2>/dev/null || true
# v1 stage 2 children = chart_base / math_base run_experiment.py + e6 calibrate
pkill -TERM -f "run_experiment.py.*--config configs/experiment_e5e_chartqa_full" 2>/dev/null || true
pkill -TERM -f "run_experiment.py.*--config configs/experiment_e5e_mathvista_full" 2>/dev/null || true
pkill -TERM -f "e6_steering_vector.py.*calibrate-subspace" 2>/dev/null || true
sleep 5
# Force kill stragglers
pkill -KILL -f "_phase1_post_baseline_parallel.sh" 2>/dev/null || true
pkill -KILL -f "run_experiment.py.*--config configs/experiment_e5e_chartqa_full" 2>/dev/null || true
pkill -KILL -f "run_experiment.py.*--config configs/experiment_e5e_mathvista_full" 2>/dev/null || true
pkill -KILL -f "e6_steering_vector.py.*calibrate-subspace" 2>/dev/null || true

# ─── Step 3: wait for GPUs 0/1/2 to free ─────────────────────────────────────
note "waiting for GPU 0/1/2 to free up (memory.used summed across 0/1/2 < 3 GB)"
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -3 | awk '{s+=$1} END {print s}')
  if [ -z "$used" ]; then used=99999; fi
  if [ "$used" -lt 3000 ]; then
    note "GPUs free (sum memory.used = $used MiB)"
    break
  fi
  sleep 5
done

# ─── Step 4: smoke test on GPU 0 (5 samples, OneVision E1 AnyRes path) ───────
note "smoke test: 5-sample OneVision E1 extraction on GPU 0"
SMOKE_TS=$(date +%Y%m%d-%H%M%S)
SMOKE_LOG="$LOG_DIR/smoke_e1_$SMOKE_TS.log"

CUDA_VISIBLE_DEVICES=0 uv run python scripts/extract_attention_mass.py \
    --model llava-onevision-qwen2-7b-ov \
    --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --config configs/experiment_e7_plotqa_full.yaml \
    --susceptibility-csv docs/insights/_data/susceptibility_plotqa_onevision.csv \
    --top-decile-n 3 --bottom-decile-n 2 \
    --bbox-file inputs/irrelevant_number_bboxes.json \
    --max-new-tokens 8 \
    > "$SMOKE_LOG" 2>&1
SMOKE_RC=$?
note "smoke exit code: $SMOKE_RC"

# Locate smoke output dir (newest under llava-onevision-qwen2-7b-ov)
SMOKE_DIR=$(ls -td outputs/attention_analysis/llava-onevision-qwen2-7b-ov/*/ 2>/dev/null | head -1)
SMOKE_PRED="${SMOKE_DIR%/}/per_step_attention.jsonl"

if [ ! -f "$SMOKE_PRED" ]; then
  note "SMOKE FAIL: $SMOKE_PRED missing — see $SMOKE_LOG"
  exit 1
fi
N=$(wc -l <"$SMOKE_PRED" 2>/dev/null || echo 0)
HAS_DIGIT=$(grep -c '"image_anchor_digit":' "$SMOKE_PRED" || true)
note "smoke: $N records, $HAS_DIGIT with image_anchor_digit"

if [ "$N" -lt 5 ] || [ "$HAS_DIGIT" -lt 1 ]; then
  note "SMOKE FAIL — record count or digit field insufficient"
  exit 1
fi

# Sanity: first record's image_anchor_digit values are non-trivial
uv run python -c "
import json, sys
recs = [json.loads(l) for l in open('$SMOKE_PRED').read().splitlines() if l.strip()]
ok = False
for r in recs:
    for ps in r.get('per_step', []):
        d = ps.get('image_anchor_digit')
        if d and any(v > 0 for v in d):
            ok = True; break
    if ok: break
sys.exit(0 if ok else 1)
" 2>>"$LOG"
if [ $? -ne 0 ]; then
  note "SMOKE FAIL — image_anchor_digit values all zero (bbox routing broken?)"
  exit 1
fi

note "SMOKE PASS — Phase A AnyRes code verified on real OneVision"
note "removing smoke output dir: $SMOKE_DIR"
rm -rf "$SMOKE_DIR"

# ─── Step 5: launch v3 orchestrator (foreground here; the parent nohup'd) ────
note "launching v3 orchestrator"
bash scripts/_phase1_post_baseline_parallel_v3.sh \
    > logs/phase1/post_baseline_parallel_v3.full.log 2>&1
V3_RC=$?
note "==== v3 orchestrator exit code: $V3_RC ===="
exit $V3_RC
