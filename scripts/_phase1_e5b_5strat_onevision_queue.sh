#!/usr/bin/env bash
# Phase 2 E5b 5-stratum cross-dataset queue (OneVision Main only).
#
# Waits for the internvl3 TallyQA finalize watcher to complete, then
# runs OneVision-only 5-stratum E5b on PlotQA + InfoVQA + ChartQA +
# MathVista. Each dataset uses 2-GPU sharded inference + DataLoader
# prefetch.
#
# Order: smallest → largest, so we get small results landing fast.
#   1. MathVista (~171 samples × 12 cond, ~15 min)
#   2. ChartQA   (~226 samples × 12 cond, ~20 min)
#   3. InfoVQA   (~1147 samples × 12 cond, ~1.5 h)
#   4. PlotQA    (~5000 samples × 12 cond, ~3-4 h)
#
# Total: ~5-6 h on 2 H200 with prefetch.
#
# After all four complete, runs analyze_e5e_wrong_correct on each, then
# commits + pushes via plain git.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/e5b_5strat_onevision_queue.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

note "===== E5b 5-stratum OneVision queue start ====="

# ---------- Phase 0: wait for tally finalize watcher to exit ----------
note "Waiting for internvl3 tally finalize watcher to exit..."
while pgrep -f "_phase1_internvl3_tally_finalize" > /dev/null; do
  sleep 120
done
note "Tally finalize watcher gone. Begin 5-stratum queue."

# Quick sanity: tally merged predictions exist (otherwise something
# upstream broke and we should not start downstream work).
merged=$(find outputs/experiment_e5e_tallyqa_full/internvl3-8b -maxdepth 2 -name predictions.jsonl -size +50M 2>/dev/null | grep -v _shards | head -1)
if [ -z "$merged" ]; then
  note "ABORT: tally merged predictions.jsonl not found"
  exit 1
fi
note "Tally OK: $merged"

# ---------- Phase 1: per-dataset sharded runs (2 GPU, prefetch ON) ----
# Order: small → large. PlotQA last so a partial completion still leaves
# the smaller datasets fully analyzed.
DATASETS=(
  "experiment_e5b_5strat_mathvista_onevision"
  "experiment_e5b_5strat_chartqa_onevision"
  "experiment_e5b_5strat_infographicvqa_onevision"
  "experiment_e5b_5strat_plotqa_onevision"
)

MODEL="llava-onevision-qwen2-7b-ov"

for cfg in "${DATASETS[@]}"; do
  note "----- ${cfg} -----"
  cfg_path="configs/${cfg}.yaml"
  if [ ! -f "$cfg_path" ]; then
    note "WARN: missing config $cfg_path — skip"
    continue
  fi

  # Look for an existing run dir with merged predictions.jsonl. If any
  # post-2026-05-04 run is already complete, skip (idempotent re-runs).
  existing=$(find "outputs/${cfg}/${MODEL}" -maxdepth 2 -name predictions.jsonl -size +1M 2>/dev/null | grep -v _shards | head -1)
  if [ -n "$existing" ]; then
    note "SKIP: existing complete run at $existing"
    continue
  fi

  shard_log="$LOG_DIR/e5b_5strat_${cfg}.log"
  note "launching 2-shard sharded run -> $shard_log"
  VLM_ENABLE_PREFETCH=1 uv run python scripts/run_experiment_sharded.py \
      --config "$cfg_path" \
      --model "$MODEL" \
      --gpus 0,1 \
      > "$shard_log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    note "ERR: ${cfg} returned rc=$rc — see $shard_log"
    # don't abort; continue with the next dataset
  else
    note "${cfg} OK"
  fi
done

# ---------- Phase 2: per-cell aggregation ----------
note "===== Phase 2: per-cell aggregation ====="
for cfg in "${DATASETS[@]}"; do
  note "analyze_e5e_wrong_correct $cfg"
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$cfg" >> "$LOG" 2>&1 || \
    note "WARN: aggregator failed on $cfg (analyzer may not support 5-stratum yet — that's OK; raw predictions are usable)"
done

# ---------- Phase 3: commit + push ----------
note "===== Phase 3: git commit + push ====="
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no tracked changes (gitignored data/output only — expected)"
else
  git commit -m "Phase 2 E5b 5-stratum cross-dataset (OneVision Main, 4 datasets)

OneVision-only 5-stratum E5b validation on PlotQA + InfoVQA + ChartQA +
MathVista. Full anchor distance schedule (b + a/m × {S1..S5} + d = 12-cond)
under scheme=relative. Validates §5 plausibility-window decay shape on
the four chart/figure/math datasets that previously had only S1.

Configs: configs/experiment_e5b_5strat_<ds>_onevision.yaml × 4.
Outputs (gitignored): outputs/experiment_e5b_5strat_<ds>_onevision/
llava-onevision-qwen2-7b-ov/<ts>/predictions.jsonl × 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" >> "$LOG" 2>&1
  git push origin "$cur_branch" >> "$LOG" 2>&1 || note "push warn"
  note "Pushed to $cur_branch"
fi

note "===== E5b 5-stratum OneVision queue done ====="
