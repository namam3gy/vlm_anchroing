#!/usr/bin/env bash
# H1 Stage 3b driver — L1 6-bin confidence quartile CSV regen (§4.4 figure).
#
# Chain:
#   1. recompute_answer_span_confidence.py --root <H1 predictions root>
#      → adds `answer_span_*` proxy fields (cross_entropy, geo_mean_prob, …)
#        to each cell's predictions.jsonl in place (.bak backup written).
#   2. analyze_confidence_anchoring.py --paper2-root <root> --n-bins 6
#      --out-suffix _6bin
#      → emits `docs/insights/_data/L1_confidence_quartile_long_6bin.csv`
#        (+ `_pair_records.csv`, `_proxy_monotonicity_6bin.csv`,
#         `_proxy_comparison_6bin.csv`).
#   3. cp L1_confidence_quartile_long_6bin.csv  →
#        outputs/paper2/cross_model_cross_dataset/summary/  (consumed by
#        notebooks/paper_section_4_figures.ipynb Figure 4).
#
# Usage (from worktree root, after Stage 3 main has landed):
#   bash scripts/_run_stage3b.sh
#
# Pre-conditions:
#   - 30/30 cells complete (_done.marker present) under
#     outputs/paper2/cross_model_cross_dataset/predictions/.

set -euo pipefail
cd "$(dirname "$0")/.."   # worktree root

PY=.venv/bin/python
H1_PRED_ROOT=outputs/paper2/cross_model_cross_dataset/predictions
H1_SUMMARY_DIR=outputs/paper2/cross_model_cross_dataset/summary
CANON_DIR=docs/insights/_data

echo "=== Stage 3b.1 · Recompute answer_span_* fields on H1 predictions ==="
$PY scripts/recompute_answer_span_confidence.py --root "$H1_PRED_ROOT"

echo ""
echo "=== Stage 3b.2 · 6-bin L1 confidence-quartile analysis (H1 layout) ==="
$PY scripts/analyze_confidence_anchoring.py \
  --paper2-root "$H1_PRED_ROOT" \
  --n-bins 6 \
  --out-suffix _6bin \
  --primary-proxy cross_entropy \
  --print-summary

echo ""
echo "=== Stage 3b.3 · Mirror canonical CSV into outputs/paper2/ summary ==="
mkdir -p "$H1_SUMMARY_DIR"
cp -v "$CANON_DIR/L1_confidence_quartile_long_6bin.csv" \
      "$H1_SUMMARY_DIR/L1_confidence_quartile_long_6bin.csv"

echo ""
echo "=== Stage 3b.4 · Refresh §4 figures (Figure 4 picks up new L1 CSV) ==="
$PY scripts/_exec_notebook.py notebooks/paper_section_4_figures.ipynb

echo ""
echo "=== Stage 3b outputs ==="
ls -la "$CANON_DIR/L1_confidence_quartile_long_6bin.csv" \
       "$H1_SUMMARY_DIR/L1_confidence_quartile_long_6bin.csv" 2>/dev/null
echo ""
echo "Figures (PDF + PNG):"
ls -la outputs/paper2/cross_model_cross_dataset/section_4_figures/ 2>/dev/null

echo ""
echo "✅  Stage 3b complete. §4 Figure 4 (confidence binning) refreshed end-to-end."
