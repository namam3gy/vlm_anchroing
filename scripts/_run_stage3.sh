#!/usr/bin/env bash
# H1 Stage 3 driver — promote last shard run + regen canonical CSVs + §4 figures.
#
# Runs the full §4 main-panel chain end-to-end:
#   1. Promote outputs/paper2/_shard_runs/.../qwen2.5-vl-32b-instruct/ predictions
#      to outputs/paper2/cross_model_cross_dataset/predictions/plotqa/...
#      + touch _done.marker  → 30/30 cells complete.
#   2. Execute paper_cross_model_cross_dataset.ipynb headless → emits canonical
#      CSVs at outputs/paper2/.../summary/main_panel_per_cell.csv +
#      docs/insights/_data/main_panel_5dataset_per_cell.csv.
#   3. Execute paper_section_4_figures.ipynb headless → §4 Figure 1/2/4 PDF+PNG
#      at outputs/paper2/cross_model_cross_dataset/section_4_figures/.
#
# Usage (from worktree root):
#   bash scripts/_run_stage3.sh
#
# Pre-conditions:
#   - 8-shard run for qwen2.5-vl-32b/plotqa has completed and merged into
#     outputs/paper2/_shard_runs/experiment_e7_plotqa_full/qwen2.5-vl-32b-instruct/<ts>/
#   - All other 29 cells already have _done.marker.

set -euo pipefail
cd "$(dirname "$0")/.."   # worktree root

PY=.venv/bin/python

echo "=== Stage 3.1 · Promote qwen2.5-vl-32b/plotqa shard output ==="
$PY scripts/_h1_promote_shard.py \
  --dataset plotqa \
  --model qwen2.5-vl-32b-instruct

N=$(find outputs/paper2/cross_model_cross_dataset/predictions -name "_done.marker" | wc -l)
echo "  total markers now: $N"
if [ "$N" != "30" ]; then
  echo "ERROR: expected 30 markers, got $N. Stopping before notebook execution." >&2
  exit 1
fi

echo ""
echo "=== Stage 3.2 · paper_cross_model_cross_dataset.ipynb (canonical CSV emit) ==="
$PY -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/paper_cross_model_cross_dataset.ipynb

echo ""
echo "=== Stage 3.3 · paper_section_4_figures.ipynb (Figure 1/2/4) ==="
$PY -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/paper_section_4_figures.ipynb

echo ""
echo "=== Stage 3 outputs ==="
echo "Canonical CSVs:"
ls -la docs/insights/_data/main_panel_5dataset_per_cell.csv \
       outputs/paper2/cross_model_cross_dataset/summary/main_panel_per_cell.csv 2>/dev/null
echo ""
echo "Figures (PDF + PNG):"
ls -la outputs/paper2/cross_model_cross_dataset/section_4_figures/ 2>/dev/null

echo ""
echo "✅  Stage 3 complete. Next: Stage 3b (L1 6-bin CSV) or Stage 4 (mechanism)."
