#!/usr/bin/env bash
# Phase 1 baseline polling watcher — emits one event per status change /
# milestone / hourly heartbeat. Detects: alive (with progress), COMPLETE,
# ZOMBIE (process gone w/o completion marker), STALLED (log not updating
# > 15 min while process supposedly alive).
#
# Each poll runs every POLL_INTERVAL seconds. Heartbeat keeps a baseline
# notification cadence even when nothing else changes.
set -u
cd "$(dirname "$0")/.."

POLL_INTERVAL=${POLL_INTERVAL:-600}     # 10 min default
HEARTBEAT_EVERY=${HEARTBEAT_EVERY:-6}   # 6 ticks × 10 min = 1 h heartbeat
STALL_THRESHOLD=${STALL_THRESHOLD:-900} # 15 min log silence = stalled
LOG=logs/phase1/baseline.log
TARGET_RE="bash scripts/_phase1_baseline.sh"   # pgrep pattern for the baseline driver

prev_marker=""
prev_status=""
tick=0

while true; do
  tick=$((tick + 1))

  # Process alive check (use exact arg match to avoid self-match)
  if pgrep -fa "$TARGET_RE" 2>/dev/null | grep -v "_phase1_watcher.sh" | grep -q "$TARGET_RE"; then
    proc=alive
  else
    if [ -f "$LOG" ] && grep -q "Phase 1 P0 baseline done" "$LOG"; then
      proc=COMPLETE
    else
      proc=ZOMBIE
    fi
  fi

  # Log freshness
  if [ -f "$LOG" ]; then
    log_age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    last_marker=$(grep -E "^==== " "$LOG" 2>/dev/null | tail -1 | head -c 80)
    last_iter=$(grep -oE "[0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1)
  else
    log_age=-1
    last_marker="(log missing)"
    last_iter=""
  fi

  # Stall override
  status="$proc"
  if [ "$proc" = "alive" ] && [ "$log_age" -gt "$STALL_THRESHOLD" ]; then
    status=STALLED
  fi

  emit=0
  [ "$status" != "$prev_status" ] && emit=1
  [ "$last_marker" != "$prev_marker" ] && emit=1
  [ $((tick % HEARTBEAT_EVERY)) -eq 0 ] && emit=1
  case "$status" in COMPLETE|ZOMBIE|STALLED) emit=1 ;; esac

  if [ "$emit" = "1" ]; then
    printf "[%s] %s | log_age=%ss | iter=%s | %s\n" \
      "$(date +%H:%M)" "$status" "$log_age" "$last_iter" "$last_marker"
  fi

  case "$status" in
    COMPLETE|ZOMBIE) exit 0 ;;
  esac

  prev_status="$status"
  prev_marker="$last_marker"
  sleep "$POLL_INTERVAL"
done
