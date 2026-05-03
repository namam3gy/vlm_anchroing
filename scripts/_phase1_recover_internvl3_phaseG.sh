#!/usr/bin/env bash
# Recovers Phase G internvl3-8b baselines that failed because configs used
# OpenGVLab/InternVL3-8B (trust_remote_code variant — incompatible with
# AutoModelForImageTextToText). Configs now patched to OpenGVLab/InternVL3-8B-hf.
#
# Polls for the master queue's Phase G to clear (Phase H qwen2.5-vl-7b
# extraction line in master_queue.log signals Phase G done) then re-runs
# internvl3-8b sharded baseline on 5 datasets.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/recover_internvl3_phaseG.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# Wait for Phase G to be past internvl3 (which is the FIRST model). Specifically
# wait until master queue has moved into qwen2.5-vl-32b-instruct (the second
# model) AND that has finished. Easiest signal: wait for "Phase H" line in the
# master queue log.
note "Waiting for master queue to reach Phase H (Phase G complete)..."
while ! grep -q "Phase H:" "$LOG_DIR/post_pilot_master_queue.log" 2>/dev/null; do
  sleep 60
done
note "Master queue past Phase G — running internvl3-8b sharded recovery."

# Use 4 GPUs (sharded). Master queue Phase H runs sequentially on GPU 0 only,
# so we'll pin recovery to GPUs 1/2/3 to coexist (3-way shard).
SHARDED_GPUS="1,2,3"

declare -A DS_CFG=(
  [tallyqa]="configs/experiment_e5e_tallyqa_full.yaml"
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
)

for ds in tallyqa plotqa infographicvqa chartqa mathvista; do
  cfg="${DS_CFG[$ds]}"
  exp="$(basename "$cfg" .yaml)"
  exp_dir="outputs/$exp/internvl3-8b"
  if [ -d "$exp_dir" ] && find "$exp_dir" -name predictions.jsonl -size +1M -mmin -1440 2>/dev/null | grep -q .; then
    note "[$ds] already has recent predictions — skip"
    continue
  fi
  note "[$ds] internvl3-8b sharded baseline (GPUs=$SHARDED_GPUS)"
  uv run python scripts/run_experiment_sharded.py \
      --config "$cfg" --model internvl3-8b --gpus "$SHARDED_GPUS" \
      > "$LOG_DIR/recover_internvl3_${ds}.log" 2>&1
done

note "Re-running per_cell + 5-dataset summary with internvl3-8b included"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

# Option (a): commit + push the internvl3-8b additions on whatever branch is
# currently checked out (master after Phase J merge, or the feature branch if
# recovery races ahead). We commit to whatever HEAD points at and push the
# same branch — Phase J's merge picks it up if we're still on the feature
# branch, otherwise this lands directly on master.
note "Committing internvl3-8b recovery and pushing"
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no changes to commit (recovery left no diff?)"
else
  git commit -m "Phase G recovery: internvl3-8b × 5 datasets baselines

Original Phase G run failed for internvl3-8b because configs used
OpenGVLab/InternVL3-8B (trust_remote_code variant — incompatible with
AutoModelForImageTextToText). Configs patched to OpenGVLab/InternVL3-8B-hf
and re-ran sharded × 5 datasets on GPUs 1/2/3 (concurrent with Phase H
qwen2.5-vl-7b on GPU 0). 5-dataset summary refreshed." >> "$LOG" 2>&1 \
    || note "git commit warn"
  git push origin "$cur_branch" >> "$LOG" 2>&1 \
    || note "git push warn (push manually if needed)"
  note "Pushed to $cur_branch"
fi

note "internvl3-8b Phase G recovery + summary + commit + push done."
