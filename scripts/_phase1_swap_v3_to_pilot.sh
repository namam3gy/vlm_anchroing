#!/usr/bin/env bash
# Watch v3 log for Stage 4A complete (= Stage 4B start), SIGTERM v3 before
# its narrow Stage 4B (single L=27 K=4 α=1.0 cell × 4 datasets) starts,
# then launch the §7.4.5 pilot grid (27 cells × PlotQA 250 + InfoVQA 250).
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/swap_v3_to_pilot.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

note "==== swap watcher armed (PID $$); polling for Stage 4B boundary ===="

# Wait for "Stage 4B" log marker.
WAIT_LOG=logs/phase1/post_baseline_parallel_v3.log
while ! grep -q "Stage 4B: parallel non-tally sweeps" "$WAIT_LOG" 2>/dev/null; do
  sleep 10
done
note "v3 reached Stage 4B boundary (Stage 4A tally sweep complete)"

# SIGTERM v3 main + child sweep procs (Stage 4B may have just spawned its
# 3 background subshells; kill them too).
note "killing v3 + Stage 4B child python procs"
pkill -TERM -f "_phase1_post_baseline_parallel_v3.sh" 2>/dev/null || true
pkill -TERM -f "e6_steering_vector.py.*sweep-subspace" 2>/dev/null || true
pkill -TERM -f "run_sweep_subspace_sharded.py" 2>/dev/null || true
sleep 5
pkill -KILL -f "_phase1_post_baseline_parallel_v3.sh" 2>/dev/null || true
pkill -KILL -f "e6_steering_vector.py.*sweep-subspace" 2>/dev/null || true
pkill -KILL -f "run_sweep_subspace_sharded.py" 2>/dev/null || true

# Wait for GPUs to free
note "waiting for GPU 0/1/2 to free up"
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -3 | awk '{s+=$1} END {print s}')
  used=${used:-99999}
  if [ "$used" -lt 3000 ]; then
    note "GPUs free (sum memory.used = $used MiB)"
    break
  fi
  sleep 5
done

# Launch pilot grid.
note "==== launching _phase1_pilot_grid.sh ===="
bash scripts/_phase1_pilot_grid.sh
PILOT_RC=$?
note "==== pilot grid exit code: $PILOT_RC ===="

# After pilot, the user inspects pilot_grid_cell_selection.csv and
# manually triggers Stage 4-final + Stage 5. Done here.
exit $PILOT_RC
