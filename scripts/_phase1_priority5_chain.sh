#!/usr/bin/env bash
# Priority 5 expansion: 7 models × 5 datasets × 6-cond (b / a-S1..S5 /
# m-S1..S5 / d) baseline matrix.
#
# Triggered after master queue Phase J completes (i.e., once the
# branch is merged → master + pushed). Polls for "Phase J:" line in the
# master queue log.
#
# Models: llava-onevision-7b-ov, llava-next-interleaved-7b, internvl3-8b,
#         qwen2.5-vl-7b, qwen2.5-vl-32b, gemma3-4b-it, gemma3-27b-it
# Datasets: tallyqa, chartqa, mathvista, plotqa, infographicvqa
# 6-cond = stratified anchor with 5 distance strata (default
# ANCHOR_DISTANCE_STRATA in src/vlm_anchor/data.py) instead of single
# [[0, 5]] stratum.
#
# Compute estimate (rough, with DataLoader prefetch enabled):
#   7 models × 5 ds × ~1.5h avg = ~50h sharded across 4 GPUs
#   ≈ 12-15h wall-clock with prefetch (25% gain).
#
# Outputs:
#   outputs/experiment_p5_<dataset>_full/<model>/<run>/predictions.jsonl
#   docs/insights/_data/priority5_7model_5dataset_summary.csv
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/priority5_chain.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# Wait until master queue Phase J commit lands. Phase J = "branch merge +
# push to master" — emits "Phase J:" log line just before checkout master.
note "Waiting for master queue Phase J to start..."
while ! grep -q "Phase J:" "$LOG_DIR/post_pilot_master_queue.log" 2>/dev/null; do
  sleep 60
done
# Also wait for J to fully complete (no master queue process running).
while pgrep -f "_phase1_post_pilot_master_queue" > /dev/null; do
  sleep 30
done
note "Master queue done. Starting Priority 5 expansion."

# --- 1) Create 6-cond config YAMLs from existing 4-cond configs ---
# Source configs (single-stratum 4-cond):
declare -A SRC_CFG=(
  [tallyqa]="configs/experiment_e5e_tallyqa_full.yaml"
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
)
declare -A NEW_CFG
for ds in tallyqa chartqa mathvista plotqa infographicvqa; do
  src="${SRC_CFG[$ds]}"
  dst="configs/experiment_p5_${ds}_full.yaml"
  NEW_CFG[$ds]="$dst"
  if [ ! -f "$dst" ]; then
    # Strip anchor_distance_strata (forces default 5-stratum). Also bump
    # output_root namespace so we don't collide with 4-cond runs.
    awk '
      /^  anchor_distance_strata:/ { next }
      /^output_root:/ {
        sub(/outputs/, "outputs/p5", $0); print; next
      }
      { print }
    ' "$src" > "$dst"
    note "Created $dst (6-cond)"
  fi
done

# --- 2) Determine GPU set ---
GPUS_NOW="${GPU_LIST:-0,1,2,3}"
note "Using GPUs: $GPUS_NOW"

# --- 3) Run baselines: 7 models × 5 datasets ---
# Order: smallest models first to fail fast on issues.
MODELS=(
  gemma3-4b-it
  llava-next-interleaved-7b
  llava-onevision-qwen2-7b-ov
  internvl3-8b
  qwen2.5-vl-7b-instruct
  gemma3-27b-it
  qwen2.5-vl-32b-instruct
)
DATASETS=(chartqa mathvista infographicvqa plotqa tallyqa)

cell_done() {
  local exp_dir="$1"
  [ -d "$exp_dir" ] || return 1
  for ts in "$exp_dir"/*; do
    [ -d "$ts" ] || continue
    f="$ts/predictions.jsonl"
    if [ -f "$f" ] && [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -gt $((1024 * 1024)) ]; then
      return 0
    fi
  done
  return 1
}

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    cfg="${NEW_CFG[$ds]}"
    exp_name="experiment_p5_${ds}_full"
    exp_dir="outputs/p5/$exp_name/$model"
    if cell_done "$exp_dir"; then
      note "[$model/$ds] already done — skip"
      continue
    fi
    note "[$model/$ds] sharded baseline (6-cond, GPUs=$GPUS_NOW, prefetch ON)"
    VLM_ENABLE_PREFETCH=1 uv run python scripts/run_experiment_sharded.py \
        --config "$cfg" --model "$model" --gpus "$GPUS_NOW" \
        > "$LOG_DIR/p5_${model}_${ds}.log" 2>&1 \
      || note "WARN: $model/$ds failed (see $LOG_DIR/p5_${model}_${ds}.log)"
  done
  # Commit per-model so partial progress is captured.
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "Priority 5 partial: $model × 5 datasets (6-cond)" >> "$LOG" 2>&1 \
      || note "git commit warn ($model)"
  fi
done

# --- 4) Build summary CSV ---
note "Building 7-model × 5-dataset 6-cond summary"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1 || true

# --- 5) Commit + push ---
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Priority 5 final: 7-model × 5-dataset × 6-cond summary" >> "$LOG" 2>&1 \
    || note "git commit warn (final)"
fi
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git push origin "$cur_branch" >> "$LOG" 2>&1 \
  || note "git push warn (push manually if needed)"

note "Priority 5 chain complete."
