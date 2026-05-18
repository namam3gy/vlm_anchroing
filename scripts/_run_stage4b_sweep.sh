#!/usr/bin/env bash
# H1 Stage 4b.3 — §5.2 K-subspace sweep on H1 subspace + H1 predictions.
#
# Chain:
#   1. seed canonical subspace (notebook step) — symlinks H1 subspace
#   2. sweep_pilot (45-cell pilot grid, PlotQA n=250)
#   3. sweep_5dataset_layer (5 datasets × layer sweep)
#   4. aggregate (pilot grid CSV + layer sweep CSV + figures)
#
# Uses the notebook orchestration in
# paper_section_5_2_subspace_sweep.ipynb. The notebook reads GPUS from
# the VLM_ANCHOR_GPUS env var. If §5.1 attention extraction is still
# running on some GPUs, this script auto-detects free GPUs (memory <
# 1 GB) and passes only those. Re-running this script after more GPUs
# free up will skip already-done sweep cells via marker resume.
#
# Usage (from worktree root):
#   bash scripts/_run_stage4b_sweep.sh                 # autodetect free GPUs
#   VLM_ANCHOR_GPUS=0,1,2,3,4,5,6,7 bash scripts/_run_stage4b_sweep.sh
#                                                     # force GPU list

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MAIN=$(git rev-parse --git-common-dir | xargs realpath | xargs dirname)

if [ -z "${VLM_ANCHOR_GPUS:-}" ]; then
  # Free = memory < 1024 MiB
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
              | awk -F', ' '$2 < 1024 {print $1}' | paste -sd, -)
  if [ -z "$FREE_GPUS" ]; then
    echo "ERROR: no free GPUs detected. Set VLM_ANCHOR_GPUS to force." >&2
    exit 1
  fi
  export VLM_ANCHOR_GPUS=$FREE_GPUS
fi
echo "Using GPUs: $VLM_ANCHOR_GPUS"

# Pre-check: H1 subspace tensor must exist (Stage 4b.1+2 must have run)
SUBSPACE_PATH=$MAIN/outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/subspace_plotqa_infovqa_pooled_h1_K16.pt
if [ ! -f "$SUBSPACE_PATH" ]; then
  echo "ERROR: H1 subspace not found at $SUBSPACE_PATH" >&2
  echo "Run 'bash scripts/_run_stage4b_calibrate.sh' first." >&2
  exit 1
fi
echo "H1 subspace: $SUBSPACE_PATH"

echo ""
echo "=== Stage 4b.3 · run §5.2 sweep notebook (seed + pilot + 5d layer sweep + aggregate) ==="
$PY scripts/_exec_notebook.py notebooks/paper_section_5_2_subspace_sweep.ipynb

echo ""
echo "=== Stage 4b outputs ==="
ls -la $MAIN/outputs/paper2/section_5_e6_steering/_data/ 2>/dev/null || true
ls -la $MAIN/outputs/paper2/section_5_figures/paper_5_2*.pdf 2>/dev/null || true
echo ""
echo "✅  Stage 4b sweep complete. Next: Stage 4c (§6 mitigation) or Stage 5 paper updates."
