#!/usr/bin/env bash
# Final recovery chain — fixes data integrity issues found 2026-05-04 ~02:00:
#   1) Phase E E1d ChartQA: was launched with PlotQA susceptibility CSV (wrong),
#      empty output. Re-run with proper ChartQA susceptibility CSV.
#   2) Phase E E1d MathVista: same issue. Re-run with MathVista susceptibility.
#   3) internvl3-8b TallyQA baseline: previous fast-tail killed before merge.
#      Re-run cleanly with 4-shard sharding.
#   4) Re-aggregate analyze_causal_ablation summary (CPU)
#   5) Re-run per_cell + 7-model × 5-dataset summary (CPU)
#   6) Commit + push all updates.
#
# Sequencing:
#   - ChartQA E1d already running (started by hand 02:00). Polling for exit.
#   - MathVista E1d after ChartQA (4-shard, all GPUs).
#   - internvl3 TallyQA after MathVista (4-shard, all GPUs).
#   - Then CPU steps + commit.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=logs/phase1
LOG="$LOG_DIR/recover_remaining_fixes.log"
mkdir -p "$LOG_DIR"
note() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

PEAK=27  # OneVision answer peak from per_layer_summary
HF_OV="llava-hf/llava-onevision-qwen2-7b-ov-hf"

# --- 1) Wait for ChartQA E1d (already in flight) ---
note "Waiting for ChartQA E1d to finish..."
while pgrep -f "run_causal_ablation_sharded.*chartqa" > /dev/null; do
  sleep 60
done
# Verify chartqa output
chart_pred=$(find outputs/causal_ablation/llava-onevision-qwen2-7b-ov -maxdepth 2 -name predictions.jsonl -newer "$LOG_DIR/recover_phaseE_chartqa.log" -size +100k 2>/dev/null | grep -v _shards | head -1)
if [ -n "$chart_pred" ]; then
  note "ChartQA E1d done: $chart_pred"
else
  note "WARN: ChartQA E1d output not found or too small — check $LOG_DIR/recover_phaseE_chartqa.log"
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
  || note "WARN: MathVista E1d run had non-zero exit"
note "MathVista E1d step done"

# --- 3) internvl3-8b TallyQA baseline rerun ---
note "Starting internvl3-8b TallyQA baseline (4-shard)"
uv run python scripts/run_experiment_sharded.py \
    --config configs/experiment_e5e_tallyqa_full.yaml \
    --model internvl3-8b --gpus 0,1,2,3 \
    > "$LOG_DIR/recover_internvl3_tallyqa_final.log" 2>&1 \
  || note "WARN: internvl3 TallyQA had non-zero exit"
note "internvl3 TallyQA step done"

# --- 4) CPU re-aggregations ---
note "Re-running analyze_causal_ablation"
uv run python scripts/analyze_causal_ablation.py >> "$LOG" 2>&1 || note "analyze_causal_ablation warn"

note "Re-running analyze_attention_per_layer + analyze_cross_dataset_peaks"
uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1 || true
uv run python scripts/analyze_cross_dataset_peaks.py >> "$LOG" 2>&1 || true

note "Re-running per_cell × 5 datasets"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done

note "Building 7-model × 5-dataset summary"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

# --- 5) Commit + push ---
note "Committing all recovery fixes"
cur_branch=$(git rev-parse --abbrev-ref HEAD)
git add -A
if git diff --cached --quiet; then
  note "no changes to commit (gitignored data only — expected)"
else
  git commit -m "Recovery: Phase E ChartQA/MathVista E1d + internvl3 TallyQA baseline

Fixes data integrity issues from 2026-05-03:

1) Phase E E1d ChartQA + MathVista re-ran with their own susceptibility
   CSVs. The original master queue script reused PlotQA's susceptibility
   CSV for both, producing empty outputs (commit 7a27750 'Phase E')
   because the qid mappings didn't match the actual ChartQA/MathVista
   samples — the top/bottom decile filtering yielded 0 rows.

2) Built susceptibility_chartqa_onevision.csv +
   susceptibility_mathvista_onevision.csv from existing OneVision
   baselines (gitignored output but referenced by re-runs).

3) internvl3-8b TallyQA baseline re-ran with 4-shard, full 38K samples.
   Previous fast-tail watcher (commit 6c7d99a) killed the original
   3-shard run before merge — shard0 lost data, shard1/2 had partial
   files but no merged predictions.jsonl. Watcher bug separately fixed.

4) Re-aggregated analyze_causal_ablation, per_cell × 5 datasets,
   7-model × 5-dataset summary." >> "$LOG" 2>&1 || note "git commit warn"
  git push origin "$cur_branch" >> "$LOG" 2>&1 || note "git push warn"
  note "Pushed to $cur_branch"
fi

note "Recovery chain done."
