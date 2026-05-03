#!/usr/bin/env bash
# Speed up the tail of internvl3-8b recovery: after tally finishes (the
# bottleneck), kill the original recovery script (which would run plotqa /
# infovqa / chartqa / mathvista on GPU 1/2/3) and re-launch those 4 datasets
# on GPU 0/1/2/3 (4-shard sharding) for ~25% faster per dataset.
#
# Trade-off: Priority 5 (also using all 4 GPUs) sees +28% contention on
# GPU 0 for the duration (~40min), but internvl3's per-dataset speedup
# (~33%) net-saves ~9min wall and frees recovery to commit + push sooner.
#
# Polls original recovery log for "tallyqa internvl3-8b sharded baseline"
# completion, then takes over.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/recover_internvl3_fast_tail.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# Poll until tallyqa baseline (the slow one) finishes. The recovery script
# emits a per-dataset start line and only writes predictions.jsonl > 1M
# when the merge succeeds. Watch for both signals to be safe.
note "Watching for tally completion..."
while true; do
  # Tally done if predictions.jsonl > 100M exists in any internvl3-8b run dir
  # under experiment_e5e_tallyqa_full (4-cond × 38K samples ≈ 100MB+)
  done_pred=$(find outputs/experiment_e5e_tallyqa_full/internvl3-8b -name predictions.jsonl -size +50M 2>/dev/null | head -1)
  if [ -n "$done_pred" ]; then
    note "Tally predictions.jsonl found: $done_pred"
    break
  fi
  sleep 60
done

# Wait a beat to ensure the original recovery has moved on / finished tally
# fully (merge + summary).
sleep 30

# Kill the original recovery script so it doesn't continue with 3-GPU
# sharding for the remaining datasets. It's mid for-loop after tally.
RECOVERY_PID=$(pgrep -f "_phase1_recover_internvl3_phaseG" || true)
if [ -n "$RECOVERY_PID" ]; then
  note "Killing original recovery PID $RECOVERY_PID"
  kill "$RECOVERY_PID" 2>/dev/null || true
  sleep 5
  # Also kill any in-flight child processes
  pkill -f "run_experiment.*internvl3-8b" 2>/dev/null || true
  pkill -f "run_experiment_sharded.*internvl3-8b" 2>/dev/null || true
fi

# Now run the 4 remaining datasets sharded on all 4 GPUs.
SHARDED_GPUS="0,1,2,3"

declare -A DS_CFG=(
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
)

for ds in plotqa infographicvqa chartqa mathvista; do
  cfg="${DS_CFG[$ds]}"
  exp="$(basename "$cfg" .yaml)"
  exp_dir="outputs/$exp/internvl3-8b"
  # Re-check skip predicate: if a prior recent baseline exists from the
  # original recovery (e.g. it managed to finish plotqa before we killed it)
  if [ -d "$exp_dir" ] && find "$exp_dir" -name predictions.jsonl -size +1M -newer "$LOG" 2>/dev/null | grep -q .; then
    note "[$ds] already has recent predictions — skip"
    continue
  fi
  note "[$ds] internvl3-8b sharded baseline (GPUs=$SHARDED_GPUS, 4-shard)"
  uv run python scripts/run_experiment_sharded.py \
      --config "$cfg" --model internvl3-8b --gpus "$SHARDED_GPUS" \
      > "$LOG_DIR/recover_internvl3_${ds}_fasttail.log" 2>&1 \
    || note "WARN: $ds failed"
done

note "Re-running per_cell + 5-dataset summary with internvl3-8b included"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

note "Committing internvl3-8b recovery and pushing"
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no changes to commit"
else
  git commit -m "Phase G recovery: internvl3-8b × 5 datasets baselines

Original Phase G run failed for internvl3-8b because configs used
OpenGVLab/InternVL3-8B (trust_remote_code variant — incompatible with
AutoModelForImageTextToText). Configs patched to OpenGVLab/InternVL3-8B-hf.
Recovery ran tallyqa on GPU 1/2/3 (3-shard), then this fast-tail variant
ran the 4 smaller datasets on GPU 0/1/2/3 (4-shard) for ~25% per-dataset
speedup. Also re-aggregated per_cell + 5-dataset summary." >> "$LOG" 2>&1 \
    || note "git commit warn"
  git push origin "$cur_branch" >> "$LOG" 2>&1 \
    || note "git push warn"
  note "Pushed to $cur_branch"
fi

note "internvl3-8b fast-tail recovery + summary done."
