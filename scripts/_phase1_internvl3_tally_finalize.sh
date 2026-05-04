#!/usr/bin/env bash
# Final completion watcher for internvl3-8b TallyQA baseline rerun (2-shard,
# prefetch ON). Polls for the merged predictions.jsonl, then runs CPU
# re-aggregations, commits, and pushes.
#
# Avoids the bash set -euo pipefail + grep-empty edge case that bit the
# previous chain.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/internvl3_tally_finalize.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

note "Watching for internvl3 TallyQA merged predictions.jsonl..."
while true; do
  # Look for merged predictions.jsonl (NOT shard sub-files), > 50MB,
  # in any internvl3-8b run dir created today.
  merged=$(find outputs/experiment_e5e_tallyqa_full/internvl3-8b -maxdepth 2 -name predictions.jsonl -size +50M 2>/dev/null | grep -v _shards | head -1)
  if [ -n "$merged" ]; then
    note "Merged predictions found: $merged"
    break
  fi
  # If the launcher process is gone and still no merged file, abort.
  if ! pgrep -f "run_experiment_sharded.*internvl3-8b" > /dev/null; then
    note "WARN: launcher gone but no merged file — aborting"
    exit 1
  fi
  sleep 60
done

# Wait a beat for summary.json to land
sleep 10

note "Re-aggregating per_cell × 5 datasets"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done

note "Building 6-model × 5-dataset summary"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

note "Committing + pushing"
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no tracked changes (gitignored data only — expected)"
else
  git commit -m "internvl3-8b TallyQA baseline rerun: complete 6-model × 5-dataset matrix

Final missing cell from the 6-model main panel: internvl3-8b on TallyQA
(38245 samples, 4-cond stratified [[0,5]]). Re-ran with 2-shard sharding
(GPU 0+1, fresh post-restart) + DataLoader prefetch. Re-aggregated
per_cell × 5 datasets and rebuilt main_panel_5dataset_summary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" >> "$LOG" 2>&1
  git push origin "$cur_branch" >> "$LOG" 2>&1 || note "push warn"
  note "Pushed to $cur_branch"
fi

note "internvl3 tally finalization done."
