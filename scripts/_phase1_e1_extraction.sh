#!/usr/bin/env bash
# Phase 1 P0 v3 §7.1-7.3 mechanism panel extraction — OneVision Main.
#
# Layout:
#  Stage E1 (parallel 3 GPUs):
#    GPU 0: E1 attention-mass + E1-patch on PlotQA   (~1.5h)
#    GPU 1: E1 attention-mass + E1-patch on TallyQA  (~1.5h)
#    GPU 2: E1 attention-mass + E1-patch on InfoVQA  (~1.5h)
#  Stage E1d (single GPU after E1 done):
#    GPU 0: E1d causal ablation on PlotQA            (~1h)
#  Stage analysis (CPU after extraction done):
#    analyze_attention_per_layer
#    analyze_attention_patch
#    analyze_causal_ablation
#    => report empirical peak layer for OneVision Main
#
# Idempotent: each cell skipped if outputs/attention_analysis/<model>/<run>/
# per_step_attention.jsonl exists with > 100 records.
#
# Pre-req:
#  - susceptibility_<dataset>_onevision.csv must exist for each dataset
#    (build via scripts/build_dataset_susceptibility.py)
#  - inputs/irrelevant_number_bboxes.json (digit bbox file) — already exists
#  - outputs/experiment_e7_plotqa_full / e7_infographicvqa_full /
#    e5e_tallyqa_full predictions for OneVision main
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=llava-onevision-qwen2-7b-ov
HF=llava-hf/llava-onevision-qwen2-7b-ov-hf
LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/e1_extraction.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

cell_done_e1() {
  local model_dir="outputs/attention_analysis/$MODEL"
  [ -d "$model_dir" ] || return 1
  for ts in "$model_dir"/*; do
    [ -d "$ts" ] || continue
    local f="$ts/per_step_attention.jsonl"
    if [ -f "$f" ]; then
      local n; n=$(wc -l <"$f" 2>/dev/null || echo 0)
      [ "$n" -gt 100 ] && return 0
    fi
  done
  return 1
}

cell_done_e1d() {
  local model_dir="outputs/causal_ablation/$MODEL"
  [ -d "$model_dir" ] || return 1
  for ts in "$model_dir"/*; do
    [ -d "$ts" ] || continue
    local f="$ts/predictions.jsonl"
    if [ -f "$f" ]; then
      local n; n=$(wc -l <"$f" 2>/dev/null || echo 0)
      [ "$n" -gt 100 ] && return 0
    fi
  done
  return 1
}

# Per-dataset E1 launcher (gpu, dataset_tag, susceptibility_csv, config).
# OneVision multi-image inputs are forced into base-resolution mode
# (730 tokens/image, no AnyRes high-res) by passing images as a nested
# list — see EagerAttentionRunner._prepare_inputs. seq_len drops from
# ~5400 → ~1460 → stored attentions fit on a single H200.
e1_one() {
  local gpu="$1" tag="$2" susc_csv="$3" cfg="$4" log_tag="$5"
  note "[GPU$gpu] E1 $tag"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/extract_attention_mass.py \
      --model "$MODEL" --hf-model "$HF" \
      --config "$cfg" \
      --susceptibility-csv "$susc_csv" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      >> "$LOG_DIR/$log_tag.log" 2>&1
}

note "==== Phase 1 §7.1-7.3 extraction start (Main = $MODEL) ===="

# Quick existence check on susceptibility CSVs (build them via the helper if missing).
for dataset_csv in \
    docs/insights/_data/susceptibility_plotqa_onevision.csv \
    docs/insights/_data/susceptibility_tallyqa_onevision.csv \
    docs/insights/_data/susceptibility_infovqa_onevision.csv; do
  if [ ! -f "$dataset_csv" ]; then
    note "ERR: missing $dataset_csv — build via build_dataset_susceptibility.py first"
    exit 1
  fi
done

note "---- Stage E1: parallel attention extraction across 3 GPUs ----"

(
  e1_one 0 plotqa \
    docs/insights/_data/susceptibility_plotqa_onevision.csv \
    configs/experiment_e7_plotqa_full.yaml e1_plotqa
) &
PG0=$!
(
  e1_one 1 tallyqa \
    docs/insights/_data/susceptibility_tallyqa_onevision.csv \
    configs/experiment_e5e_tallyqa_full.yaml e1_tallyqa
) &
PG1=$!
(
  e1_one 2 infovqa \
    docs/insights/_data/susceptibility_infovqa_onevision.csv \
    configs/experiment_e7_infographicvqa_full.yaml e1_infovqa
) &
PG2=$!

wait $PG0 $PG1 $PG2
note "Stage E1 complete"

note "---- Stage E1b: per-layer analysis + peak layer for OneVision ----"
uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1
note "  patch concentration (E1-patch)"
uv run python scripts/analyze_attention_patch.py --print-summary >> "$LOG" 2>&1

# Extract OneVision peak layer (overall, answer-step) from peak_layer_summary.csv.
PEAK_LAYER="$(uv run python -c "
import pandas as pd, sys
p = 'outputs/attention_analysis/_per_layer/peak_layer_summary.csv'
df = pd.read_csv(p)
mask = (df['model'] == '$MODEL') & (df['stratum'] == 'all') & (df['step'] == 'answer')
sub = df[mask]
if len(sub) == 0:
    sys.stderr.write(f'no peak row for $MODEL\\n'); sys.exit(1)
print(int(sub.iloc[0]['peak_layer']))
" 2>&1)"
if ! [[ "$PEAK_LAYER" =~ ^[0-9]+$ ]]; then
  note "ERR: failed to extract peak layer: $PEAK_LAYER"
  exit 1
fi
note "OneVision peak layer (answer-step, overall): L=$PEAK_LAYER"

note "---- Stage E1d: causal ablation on PlotQA at L=$PEAK_LAYER (sharded GPU 0/1/2) ----"
if cell_done_e1d; then
  note "skip E1d (already complete)"
else
  uv run python scripts/run_causal_ablation_sharded.py \
      --model "$MODEL" --hf-model "$HF" \
      --peak-layer "$PEAK_LAYER" \
      --config configs/experiment_e7_plotqa_full.yaml \
      --susceptibility-csv docs/insights/_data/susceptibility_plotqa_onevision.csv \
      --top-decile-n 100 --bottom-decile-n 100 \
      --max-new-tokens 8 \
      --gpus 0,1,2 \
      >> "$LOG_DIR/e1d_plotqa.log" 2>&1
fi

note "---- Stage analysis: causal aggregate (CPU) ----"
uv run python scripts/analyze_causal_ablation.py >> "$LOG" 2>&1

note "==== Phase 1 §7.1-7.3 extraction done ===="
note "Peak layer (OneVision, answer-step, overall): L=$PEAK_LAYER"
note "Inspect outputs/attention_analysis/_per_layer/peak_layer_summary.csv for full breakdown."
note "Inspect outputs/causal_ablation/_summary/per_model_per_mode.csv for E1d df reductions."
