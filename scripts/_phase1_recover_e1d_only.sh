#!/usr/bin/env bash
# Lightweight recovery: just finish Phase E E1d ChartQA + MathVista, then
# CPU re-aggregations + commit + push. SKIPS internvl3 TallyQA rerun
# (deferred per user request 2026-05-04 ~02:05).
#
# - ChartQA E1d already running (PID 368960). Polls for exit.
# - MathVista E1d after ChartQA (4-shard, all GPUs).
# - analyze_causal_ablation + per_cell + 6-model summary rebuild.
# - Commit + push.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/recover_e1d_only.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

PEAK=27
HF_OV="llava-hf/llava-onevision-qwen2-7b-ov-hf"

# --- 1) Wait for ChartQA E1d ---
note "Waiting for ChartQA E1d to finish..."
while pgrep -f "run_causal_ablation_sharded.*chartqa" > /dev/null; do
  sleep 30
done
chart_pred=$(find outputs/causal_ablation/llava-onevision-qwen2-7b-ov -maxdepth 2 -name predictions.jsonl -newer "$LOG_DIR/recover_phaseE_chartqa.log" -size +100k 2>/dev/null | grep -v _shards | head -1)
if [ -n "$chart_pred" ]; then
  note "ChartQA E1d done: $chart_pred"
else
  note "WARN: ChartQA E1d output not found or too small"
fi

# --- 2) MathVista E1d ---
note "Starting MathVista E1d (4-shard)"
uv run python scripts/run_causal_ablation_sharded.py \
    --model llava-onevision-qwen2-7b-ov --hf-model "$HF_OV" \
    --peak-layer "$PEAK" \
    --config configs/experiment_e5e_mathvista_full.yaml \
    --susceptibility-csv docs/insights/_data/susceptibility_mathvista_onevision.csv \
    --top-decile-n 100 --bottom-decile-n 100 \
    --max-new-tokens 8 \
    --gpus 0,1,2,3 \
    > "$LOG_DIR/recover_phaseE_mathvista.log" 2>&1 \
  || note "WARN: MathVista E1d non-zero exit"
note "MathVista E1d done"

# --- 3) CPU re-aggregations ---
note "Re-running analyze_causal_ablation"
uv run python scripts/analyze_causal_ablation.py >> "$LOG" 2>&1 || note "analyze warn"

note "Re-running per_cell × 5 datasets + 6-model summary"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

# --- 4) Commit + push ---
note "Committing + pushing"
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no tracked changes to commit (gitignored data only — expected)"
else
  git commit -m "Phase E recovery: ChartQA + MathVista E1d (proper susceptibility)

Original Phase E master queue used PlotQA susceptibility CSV for both
ChartQA and MathVista (commit 7a27750), producing empty outputs because
the qid mappings didn't match. Built per-dataset susceptibility CSVs from
existing OneVision baselines, then re-ran E1d sharded (4-shard) on all
GPUs. internvl3-8b TallyQA rerun deferred." >> "$LOG" 2>&1 || note "commit warn"
  git push origin "$cur_branch" >> "$LOG" 2>&1 || note "push warn"
  note "Pushed to $cur_branch"
fi

note "E1d-only recovery done."
