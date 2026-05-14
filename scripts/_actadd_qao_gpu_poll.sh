#!/usr/bin/env bash
# Poll GPU every 30 min; when idle for 2 consecutive checks (= ~30 min of
# real idle), launch the ActAdd + QAO recalibration runner. Exits after launch
# (or when MAX_POLLS reached without idle).
#
# Idle: memory.used < 5000 MiB AND utilization.gpu < 5%.
# Two-consecutive-check guard: avoids race where the other GPU process happens
# to be between batches.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/_actadd_qao_gpu_poll.sh \
#       > logs/_actadd_qao_recal/poll.log 2>&1 &
#
# Env knobs:
#   POLL_INTERVAL  seconds between checks (default 1800 = 30 min)
#   IDLE_MEM_MB    memory threshold (default 5000)
#   IDLE_UTIL_PCT  utilization threshold (default 5)
#   MAX_POLLS      max checks before giving up (default 96 = 48 h)
#   GPU_INDEX      which GPU to inspect (default ${CUDA_VISIBLE_DEVICES:-0})
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/_actadd_qao_recal
mkdir -p "$LOG_DIR"
POLL_LOG="$LOG_DIR/poll.log"

POLL_INTERVAL="${POLL_INTERVAL:-1800}"
IDLE_MEM_MB="${IDLE_MEM_MB:-5000}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-5}"
MAX_POLLS="${MAX_POLLS:-96}"
GPU_INDEX="${GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-0}}"

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$POLL_LOG"; }

note "==== GPU idle poller started ===="
note "interval=${POLL_INTERVAL}s  thresholds: mem<${IDLE_MEM_MB}MiB util<${IDLE_UTIL_PCT}%  gpu_index=$GPU_INDEX  max_polls=$MAX_POLLS"

prev_idle=0
poll_count=0
while [ "$poll_count" -lt "$MAX_POLLS" ]; do
  poll_count=$((poll_count + 1))

  # Query target GPU only
  raw=$(nvidia-smi --id="$GPU_INDEX" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "ERR")
  if [ "$raw" = "ERR" ] || [ -z "$raw" ]; then
    note "poll #$poll_count: nvidia-smi failed; retrying after interval"
    prev_idle=0
    sleep "$POLL_INTERVAL"
    continue
  fi
  used=$(echo "$raw" | awk -F',' '{gsub(/ /,""); print $1}')
  util=$(echo "$raw" | awk -F',' '{gsub(/ /,""); print $2}')

  if [ "$used" -lt "$IDLE_MEM_MB" ] && [ "$util" -lt "$IDLE_UTIL_PCT" ]; then
    if [ "$prev_idle" -eq 1 ]; then
      note "poll #$poll_count: GPU idle (used=${used}MiB util=${util}%) — second consecutive idle. LAUNCHING runner."
      # Hand off — runner inherits stdout/stderr → run.log
      CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
          bash scripts/_actadd_qao_recalibration_runner.sh \
          >> "$LOG_DIR/run.log" 2>&1 &
      runner_pid=$!
      note "runner launched in background pid=$runner_pid; polling exits."
      exit 0
    fi
    note "poll #$poll_count: GPU idle (used=${used}MiB util=${util}%) — first check, waiting for confirm"
    prev_idle=1
  else
    if [ "$prev_idle" -eq 1 ]; then
      note "poll #$poll_count: GPU back to busy (used=${used}MiB util=${util}%) — resetting idle counter"
    else
      note "poll #$poll_count: GPU busy (used=${used}MiB util=${util}%)"
    fi
    prev_idle=0
  fi
  sleep "$POLL_INTERVAL"
done

note "==== reached MAX_POLLS=$MAX_POLLS without 2-consecutive-idle; giving up ===="
exit 1
