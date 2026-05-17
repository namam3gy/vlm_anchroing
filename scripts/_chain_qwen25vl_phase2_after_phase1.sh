#!/usr/bin/env bash
# Chain script: wait for Phase 1 InfoVQA + aggregator to finish, then launch
# Phase 2 (Stage-4 chosen-cell × 5-dataset sweep).
#
# Phase 1 completion markers (in order):
#   1. outputs/e6_steering/qwen2.5-vl-7b-instruct/pilot_grid_infographicvqa_n250/predictions.jsonl
#      created by Phase 1 driver after sweep merge.
#   2. docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv
#      created by analyze_e6_pilot_cells.py aggregator.
#
# Polling interval: 120s.
# Timeout: 6h (10800s polls × 120s = 6h max wait).
#
# Usage:
#   nohup bash scripts/_chain_qwen25vl_phase2_after_phase1.sh > outputs/e6_steering/qwen2.5-vl-7b-instruct/_chain.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=outputs/e6_steering/qwen2.5-vl-7b-instruct/_chain.log
ts() { date +'%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(ts)] $*"; }

INFOVQA_PRED=outputs/e6_steering/qwen2.5-vl-7b-instruct/pilot_grid_infographicvqa_n250/predictions.jsonl
CELL_CSV=docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv

note "==== chain script start — waiting for Phase 1 InfoVQA pilot + aggregator ===="
note "  watching: $INFOVQA_PRED  (Phase 1 sweep merge marker)"
note "  watching: $CELL_CSV       (aggregator marker)"

MAX_ITERS=180   # 180 × 120s = 6h
ITER=0
while [ $ITER -lt $MAX_ITERS ]; do
    ITER=$((ITER+1))
    if [ -f "$INFOVQA_PRED" ] && [ -f "$CELL_CSV" ]; then
        note "Phase 1 done after $ITER polls."
        break
    fi
    if [ $((ITER % 5)) -eq 0 ]; then
        note "still waiting (poll $ITER/$MAX_ITERS)…"
        if [ -f "$INFOVQA_PRED" ]; then
            note "  InfoVQA pilot exists; awaiting aggregator CSV."
        fi
    fi
    sleep 120
done

if [ $ITER -ge $MAX_ITERS ]; then
    note "FATAL: timeout waiting for Phase 1 completion markers"
    exit 2
fi

# Source resmgr (in case daemon was restarted between Phase 1 + Phase 2)
if [ -f /mnt/ddn/prod-runs/thyun.park/init.sh ]; then
    source /mnt/ddn/prod-runs/thyun.park/init.sh > /dev/null 2>&1 || true
fi

note "==== launching Phase 2 Stage-4 driver ===="
bash scripts/run_e6_cross_arch_qwen25vl_phase2.sh
RC=$?
note "==== Phase 2 exit code: $RC ===="
exit $RC
