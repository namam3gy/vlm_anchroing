#!/usr/bin/env bash
# Phase 1 P0 v3 post-pilot master queue.
#
# Triggered by _phase1_pilot_grid.sh after cell-selection CSV is written.
# Runs the entire post-pilot work queue end-to-end, committing after each
# major phase, and finally merging branch → master + pushing.
#
# Phases:
#   B  Stage 4-final (chosen cell × 5 datasets) — sharded sweep_tally + parallel non-tally
#   C  Stage 5 CPU finalization (recompute_answer_span_confidence × 5, per_cell × 5,
#      analyze_confidence_anchoring, build_e5e_e7_5dataset_summary)
#   D  Phase 1.5 cross-dataset §7.1-7.3:
#        - 5-panel mechanism models × {PlotQA, TallyQA, InfoVQA} E1 extraction
#        - OneVision × VQAv2 E1 extraction
#        - analyze_attention_per_layer + analyze_attention_patch rerun
#   E  E1d causal ablation extension: OneVision × {Tally, Info, Chart, Math}
#   G  New-model baseline runs (Priority 5, b/a-S1/m-S1/d 4-cond per OneVision setup):
#        internvl3-8b, qwen2.5-vl-32b-instruct, gemma3-4b-it × 5 datasets
#   H  qwen2.5-vl-7b §7.1-7.3 extension on 5 main datasets (Priority 2b)
#   I  Final 7-model × 5-dataset summary aggregation
#   J  branch merge → master + push (per memory feedback_auto_branch_merge_push)
#
# All phases idempotent: skip if outputs already exist. Commits track progress.
set -uo pipefail
cd "$(dirname "$0")/.."

MAIN_MODEL=llava-onevision-qwen2-7b-ov
HF_MAIN=llava-hf/llava-onevision-qwen2-7b-ov-hf
TAG=plotqa_infovqa_pooled_n5k

# Comma-separated GPU IDs available for sharded work. Auto-detect at start
# of each phase: prefer all 4 (0,1,2,3) when GPU 3 is free (other tenant
# released), fall back to 3 (0,1,2). Override via env: GPU_LIST=...
GPU_LIST_DEFAULT="0,1,2,3"
GPU_LIST_FALLBACK="0,1,2"
detect_gpus() {
  # Returns comma-separated GPU ids that are currently idle (memory < 2 GB).
  local override="${GPU_LIST:-}"
  if [ -n "$override" ]; then
    echo "$override"
    return
  fi
  local idle=""
  while read -r idx mem; do
    if [ "${mem%% *}" -lt 2048 ]; then
      idle="${idle}${idle:+,}${idx}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  # Need >= 3 GPUs for our sharded drivers; fall back when fewer
  local count=$(echo "$idle" | tr ',' '\n' | wc -l)
  if [ "$count" -ge 4 ]; then
    echo "0,1,2,3"
  elif [ "$count" -ge 3 ]; then
    echo "0,1,2"
  else
    echo "$idle"
  fi
}

LOG_DIR=logs/phase1
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/post_pilot_master_queue.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
phase() { note "================== $* =================="; }
gcommit() {
  local msg="$1"
  git add -A
  if git diff --cached --quiet; then
    note "no changes to commit ($msg)"
    return 0
  fi
  git commit -m "$msg" >> "$LOG" 2>&1 || note "git commit warn: $msg"
}

cell_done() {
  # Returns 0 (true) if any timestamp dir under <model_dir> has
  # predictions.jsonl + summary.json (or just predictions.jsonl when sweep)
  # over the size threshold.
  local model_dir="$1"
  local threshold_kb="${2:-1000}"
  [ -d "$model_dir" ] || return 1
  for ts in "$model_dir"/*; do
    [ -d "$ts" ] || continue
    local f="$ts/predictions.jsonl"
    if [ -f "$f" ]; then
      local sz; sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
      [ "$sz" -gt "$((threshold_kb * 1024))" ] && return 0
    fi
  done
  return 1
}

latest_run() {
  local exp="$1" model="$2"
  local model_dir="outputs/$exp/$model"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

note "==================================================================="
note "Phase 1 P0 v3 POST-PILOT MASTER QUEUE start"
note "==================================================================="

# -----------------------------------------------------------------------------
# Phase A: extract chosen cell from pilot CSV
# -----------------------------------------------------------------------------
phase "Phase A: read chosen cell from pilot grid CSV"
PILOT_POOLED_CSV=docs/insights/_data/pilot_grid_cell_selection_pooled.csv
if [ ! -f "$PILOT_POOLED_CSV" ]; then
  note "ERR: $PILOT_POOLED_CSV missing — pilot grid not finished?"
  exit 1
fi

# Pull (layer, K, alpha) of the best surviving cell (lowest mean dDF, not rejected).
read -r CHOSEN_L CHOSEN_K CHOSEN_ALPHA <<<"$(uv run python -c "
import pandas as pd
df = pd.read_csv('$PILOT_POOLED_CSV')
surviving = df[~df['any_rejected']]
if surviving.empty:
    raise SystemExit('all cells rejected — manual intervention required')
best = surviving.sort_values('mean_dDF').iloc[0]
print(int(best['layer']), int(best['K']), float(best['alpha']))
")"
note "Chosen cell: L=$CHOSEN_L K=$CHOSEN_K α=$CHOSEN_ALPHA"

# -----------------------------------------------------------------------------
# Phase B: Stage 4-final — chosen cell × 5 datasets
# -----------------------------------------------------------------------------
GPUS_NOW="$(detect_gpus)"
phase "Phase B: Stage 4-final (sweep × 5 datasets at chosen cell, GPUs=$GPUS_NOW)"

declare -A DS_CFG=(
  [tallyqa]="configs/experiment_e5e_tallyqa_full.yaml"
  [chartqa]="configs/experiment_e5e_chartqa_full.yaml"
  [mathvista]="configs/experiment_e5e_mathvista_full.yaml"
  [plotqa]="configs/experiment_e7_plotqa_full.yaml"
  [infographicvqa]="configs/experiment_e7_infographicvqa_full.yaml"
)
declare -A DS_EXP=(
  [tallyqa]="experiment_e5e_tallyqa_full"
  [chartqa]="experiment_e5e_chartqa_full"
  [mathvista]="experiment_e5e_mathvista_full"
  [plotqa]="experiment_e7_plotqa_full"
  [infographicvqa]="experiment_e7_infographicvqa_full"
)

SUBSPACE_PT="outputs/e6_steering/$MAIN_MODEL/_subspace/subspace_${TAG}_K16.pt"
[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt missing"; exit 1; }

# Stage 4A: sharded sweep_tally
SWEEP_TALLY_DIR="outputs/e6_steering/$MAIN_MODEL/sweep_subspace_tallyqa_${TAG}_chosen"
TALLY_TS="$(latest_run experiment_e5e_tallyqa_full "$MAIN_MODEL")"
TALLY_PRED="outputs/experiment_e5e_tallyqa_full/$MAIN_MODEL/$TALLY_TS/predictions.jsonl"
if [ -f "$SWEEP_TALLY_DIR/predictions.jsonl" ]; then
  note "skip sweep_tally (chosen cell, already done)"
else
  note "Stage 4A: sharded sweep_tally @ L=$CHOSEN_L K=$CHOSEN_K α=$CHOSEN_ALPHA"
  uv run python scripts/run_sweep_subspace_sharded.py \
      --config "${DS_CFG[tallyqa]}" \
      --model "$MAIN_MODEL" --hf-model "$HF_MAIN" \
      --predictions-path "$TALLY_PRED" \
      --dataset-tag "tallyqa_chosen" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "${TAG}_chosen" \
      --sweep-layers "$CHOSEN_L" --sweep-ks "$CHOSEN_K" --sweep-alphas "$CHOSEN_ALPHA" \
      --max-samples 5000 \
      --gpus "$GPUS_NOW" >> "$LOG" 2>&1
  # Move from auto-generated dir naming
  AUTO="outputs/e6_steering/$MAIN_MODEL/sweep_subspace_tallyqa_chosen_${TAG}_chosen"
  if [ -d "$AUTO" ]; then
    mkdir -p "$SWEEP_TALLY_DIR"
    mv "$AUTO"/* "$SWEEP_TALLY_DIR/" 2>/dev/null || true
    rmdir "$AUTO" 2>/dev/null || true
  fi
fi

# Stage 4B: parallel non-tally sweeps
sweep_one_single() {
  local ds="$1" cfg="$2" exp="$3" gpu="$4"
  local sweep_dir="outputs/e6_steering/$MAIN_MODEL/sweep_subspace_${ds}_${TAG}_chosen"
  if [ -f "$sweep_dir/predictions.jsonl" ]; then
    note "[GPU$gpu] skip $ds chosen-cell sweep (done)"
    return 0
  fi
  local ts; ts="$(latest_run "$exp" "$MAIN_MODEL")"
  [ -n "$ts" ] || { note "[GPU$gpu] WARN no $exp/$MAIN_MODEL run"; return 0; }
  local preds="outputs/$exp/$MAIN_MODEL/$ts/predictions.jsonl"
  note "[GPU$gpu] sweep $ds @ chosen cell"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model "$MAIN_MODEL" --hf-model "$HF_MAIN" \
      --e5c-run-dir "$(dirname "$preds")" \
      --predictions-path "$preds" \
      --dataset-tag "${ds}_chosen" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "${TAG}_chosen" \
      --sweep-layers "$CHOSEN_L" --sweep-ks "$CHOSEN_K" --sweep-alphas "$CHOSEN_ALPHA" \
      --max-samples 5000 \
      --config "$cfg" >> "$LOG_DIR/phaseB_${ds}_chosen.log" 2>&1
  # Move from auto dir name
  local auto="outputs/e6_steering/$MAIN_MODEL/sweep_subspace_${ds}_chosen_${TAG}_chosen"
  if [ -d "$auto" ]; then
    mkdir -p "$sweep_dir"
    mv "$auto"/* "$sweep_dir/" 2>/dev/null || true
    rmdir "$auto" 2>/dev/null || true
  fi
}

(sweep_one_single plotqa "${DS_CFG[plotqa]}" "${DS_EXP[plotqa]}" 0) &
SP0=$!
(
  sweep_one_single infographicvqa "${DS_CFG[infographicvqa]}" "${DS_EXP[infographicvqa]}" 1
  sweep_one_single chartqa        "${DS_CFG[chartqa]}"        "${DS_EXP[chartqa]}" 1
) &
SP1=$!
(sweep_one_single mathvista "${DS_CFG[mathvista]}" "${DS_EXP[mathvista]}" 2) &
SP2=$!
wait $SP0 $SP1 $SP2

note "Phase B Stage 4c: analyze_e6_subspace per dataset"
for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  sweep_dir="outputs/e6_steering/$MAIN_MODEL/sweep_subspace_${ds}_${TAG}_chosen"
  [ -f "$sweep_dir/predictions.jsonl" ] || continue
  uv run python scripts/analyze_e6_subspace.py --sweep-dir "$sweep_dir" >> "$LOG" 2>&1
done

gcommit "Phase B: Stage 4-final at pilot-chosen cell L=$CHOSEN_L K=$CHOSEN_K α=$CHOSEN_ALPHA × 5 datasets"

# -----------------------------------------------------------------------------
# Phase C: Stage 5 CPU finalization
# -----------------------------------------------------------------------------
phase "Phase C: Stage 5 CPU finalization"

note "  recompute_answer_span_confidence × 5 (parallel)"
PIDS=()
for exp in experiment_e7_plotqa_full experiment_e7_infographicvqa_full \
           experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full; do
  uv run python scripts/recompute_answer_span_confidence.py \
      --root "outputs/$exp" >> "$LOG_DIR/phaseC_recompute_${exp}.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"

note "  per_cell.csv refresh × 5"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done

note "  analyze_confidence_anchoring"
uv run python scripts/analyze_confidence_anchoring.py --print-summary \
    --primary-proxy cross_entropy >> "$LOG" 2>&1

note "  build_e5e_e7_5dataset_summary"
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

gcommit "Phase C: Stage 5 CPU finalization (recompute_confidence × 5, per_cell × 5, anchoring, summary)"

# -----------------------------------------------------------------------------
# Phase D: Phase 1.5 cross-dataset §7.1-7.3
# -----------------------------------------------------------------------------
phase "Phase D: cross-dataset §7.1-7.3 (Phase 1.5)"

# Use OneVision's susceptibility CSVs for the 3 new datasets — analyzes the
# SAME questions across panel models. For VQAv2 use existing 7-model CSV.
PANEL_MODELS_HF=(
  "gemma4-e4b OpenGVLab/Mowgli-7B-AlignAnything"   # placeholder; resolved below
)
# Actual list — name and HF id pairs.
declare -A PANEL_HF=(
  [gemma4-e4b]="google/gemma-4-e4b-it"
  [llava-1.5-7b]="liuhaotian/llava-v1.5-7b"
  [convllava-7b]="ConvLLaVA/ConvLLaVA-Stage5-7B-LoRA"
  [fastvlm-7b]="apple/FastVLM-7B"
  [llava-next-interleaved-7b]="llava-hf/llava-interleave-qwen-7b-hf"
)

run_e1_one() {
  local model="$1" hf="$2" tag="$3" susc="$4" cfg="$5" gpu="$6"
  local ts; ts="$(latest_run "${tag}" "$model")"
  if cell_done "outputs/attention_analysis/$model" 100; then
    note "[$model/$tag] E1 already exists — skip"
    return 0
  fi
  note "[GPU$gpu] E1 $model on $tag"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/extract_attention_mass.py \
      --model "$model" --hf-model "$hf" \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      >> "$LOG_DIR/phaseD_e1_${model}_${tag}.log" 2>&1
}

# 5 panel × 3 new datasets = 15 cells
for model in gemma4-e4b llava-1.5-7b convllava-7b fastvlm-7b llava-next-interleaved-7b; do
  hf="${PANEL_HF[$model]}"
  for ds_tag in plotqa tallyqa infovqa; do
    case "$ds_tag" in
      plotqa)        cfg="configs/experiment_e7_plotqa_full.yaml"
                     susc="docs/insights/_data/susceptibility_plotqa_onevision.csv"
                     exp="experiment_e7_plotqa_full" ;;
      tallyqa)       cfg="configs/experiment_e5e_tallyqa_full.yaml"
                     susc="docs/insights/_data/susceptibility_tallyqa_onevision.csv"
                     exp="experiment_e5e_tallyqa_full" ;;
      infovqa)       cfg="configs/experiment_e7_infographicvqa_full.yaml"
                     susc="docs/insights/_data/susceptibility_infovqa_onevision.csv"
                     exp="experiment_e7_infographicvqa_full" ;;
    esac
    # Sequential per (model, ds) on GPU 0; 16 cells × ~5min sharded = ~80min
    # but we run one at a time on GPU 0 to keep simple (~5min × 15 = ~75min sharded).
    # For speed, run 3 datasets in parallel on GPU 0/1/2 per model.
    :  # placeholder; actual launch loop below
  done
done

# Actual parallel launch: per panel model, run its 3 datasets in parallel on GPU 0/1/2
for model in gemma4-e4b llava-1.5-7b convllava-7b fastvlm-7b llava-next-interleaved-7b; do
  hf="${PANEL_HF[$model]}"
  if cell_done "outputs/attention_analysis/$model" 100; then
    note "$model already has E1 outputs (perhaps from VQAv2 or earlier) — re-discovery picks up"
  fi
  (run_e1_one "$model" "$hf" "experiment_e7_plotqa_full" \
      docs/insights/_data/susceptibility_plotqa_onevision.csv \
      configs/experiment_e7_plotqa_full.yaml 0) &
  P0=$!
  (run_e1_one "$model" "$hf" "experiment_e5e_tallyqa_full" \
      docs/insights/_data/susceptibility_tallyqa_onevision.csv \
      configs/experiment_e5e_tallyqa_full.yaml 1) &
  P1=$!
  (run_e1_one "$model" "$hf" "experiment_e7_infographicvqa_full" \
      docs/insights/_data/susceptibility_infovqa_onevision.csv \
      configs/experiment_e7_infographicvqa_full.yaml 2) &
  P2=$!
  wait $P0 $P1 $P2
  note "$model 3-dataset E1 done"
done

# OneVision × VQAv2 (1 cell)
note "OneVision × VQAv2 E1 extraction"
if ! cell_done "outputs/attention_analysis/$MAIN_MODEL/__vqav2_marker__"; then
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/extract_attention_mass.py \
      --model "$MAIN_MODEL" --hf-model "$HF_MAIN" \
      --config configs/experiment.yaml \
      --susceptibility-csv docs/insights/_data/susceptibility_strata.csv \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      >> "$LOG_DIR/phaseD_e1_onevision_vqav2.log" 2>&1
fi

note "Phase D analyze_attention_per_layer + analyze_attention_patch (multi-dataset combine)"
uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1
uv run python scripts/analyze_attention_patch.py --print-summary >> "$LOG" 2>&1

gcommit "Phase D: cross-dataset §7.1-7.3 (5 panel × 3 new ds + OneVision × VQAv2)"

# -----------------------------------------------------------------------------
# Phase E: E1d causal extension on OneVision × {Tally, Info, Chart, Math}
# -----------------------------------------------------------------------------
GPUS_NOW="$(detect_gpus)"
phase "Phase E: E1d causal extension (GPUs=$GPUS_NOW)"

# Get OneVision peak layer from analyze output
PEAK="$(uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/attention_analysis/_per_layer/peak_layer_summary.csv')
sub = df[(df['model']=='$MAIN_MODEL') & (df['stratum']=='all') & (df['step']=='answer')]
print(int(sub.iloc[0]['peak_layer']) if len(sub) else 27)
")"
note "OneVision peak L for E1d extension: $PEAK"

run_e1d_one() {
  local ds="$1" cfg="$2" susc="$3"
  if cell_done "outputs/causal_ablation/$MAIN_MODEL"; then
    # Coarse check; can't easily distinguish per-dataset. Run anyway if no marker
    :
  fi
  note "E1d sharded $ds at L=$PEAK"
  uv run python scripts/run_causal_ablation_sharded.py \
      --model "$MAIN_MODEL" --hf-model "$HF_MAIN" \
      --peak-layer "$PEAK" \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --max-new-tokens 8 \
      --gpus "$GPUS_NOW" \
      >> "$LOG_DIR/phaseE_e1d_${ds}.log" 2>&1
}

run_e1d_one tallyqa configs/experiment_e5e_tallyqa_full.yaml \
  docs/insights/_data/susceptibility_tallyqa_onevision.csv
run_e1d_one infographicvqa configs/experiment_e7_infographicvqa_full.yaml \
  docs/insights/_data/susceptibility_infovqa_onevision.csv
run_e1d_one chartqa configs/experiment_e5e_chartqa_full.yaml \
  docs/insights/_data/susceptibility_plotqa_onevision.csv  # use plotqa CSV as proxy
run_e1d_one mathvista configs/experiment_e5e_mathvista_full.yaml \
  docs/insights/_data/susceptibility_plotqa_onevision.csv

uv run python scripts/analyze_causal_ablation.py >> "$LOG" 2>&1

gcommit "Phase E: E1d causal extension on OneVision × {Tally, Info, Chart, Math}"

# -----------------------------------------------------------------------------
# Phase G: New-model baselines × 5 datasets (Priority 5)
# -----------------------------------------------------------------------------
GPUS_NOW="$(detect_gpus)"
phase "Phase G: New-model baselines (internvl3, qwen2.5-vl-32b, gemma3-4b) × 5 ds (GPUs=$GPUS_NOW)"

NEW_MODELS=(internvl3-8b qwen2.5-vl-32b-instruct gemma3-4b-it)
for nm in "${NEW_MODELS[@]}"; do
  for ds_tag in tallyqa chartqa mathvista plotqa infographicvqa; do
    cfg="${DS_CFG[$ds_tag]}"
    exp="${DS_EXP[$ds_tag]}"
    if cell_done "outputs/$exp/$nm"; then
      note "[$nm/$ds_tag] baseline already done — skip"
      continue
    fi
    note "[$nm/$ds_tag] sharded baseline run"
    uv run python scripts/run_experiment_sharded.py \
        --config "$cfg" --model "$nm" --gpus "$GPUS_NOW" \
        >> "$LOG_DIR/phaseG_${nm}_${ds_tag}.log" 2>&1
  done
  gcommit "Phase G partial: $nm × 5 datasets baselines"
done

# Refresh summary tables now with new models
note "Refresh per_cell + 5-dataset summary with new models"
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp" >> "$LOG" 2>&1
done
uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

gcommit "Phase G full: 7-model 5-dataset baselines complete; summary refreshed"

# -----------------------------------------------------------------------------
# Phase H: qwen2.5-vl-7b §7.1-7.3 extension on 5 main datasets (Priority 2b)
# -----------------------------------------------------------------------------
phase "Phase H: qwen2.5-vl-7b-instruct §7.1-7.3 extension"

QWEN_HF="Qwen/Qwen2.5-VL-7B-Instruct"
for ds_tag in plotqa tallyqa infovqa chartqa mathvista; do
  case "$ds_tag" in
    plotqa)    cfg="configs/experiment_e7_plotqa_full.yaml"
               susc="docs/insights/_data/susceptibility_plotqa_onevision.csv" ;;
    tallyqa)   cfg="configs/experiment_e5e_tallyqa_full.yaml"
               susc="docs/insights/_data/susceptibility_tallyqa_onevision.csv" ;;
    infovqa)   cfg="configs/experiment_e7_infographicvqa_full.yaml"
               susc="docs/insights/_data/susceptibility_infovqa_onevision.csv" ;;
    chartqa)   cfg="configs/experiment_e5e_chartqa_full.yaml"
               susc="docs/insights/_data/susceptibility_plotqa_onevision.csv" ;;  # reuse
    mathvista) cfg="configs/experiment_e5e_mathvista_full.yaml"
               susc="docs/insights/_data/susceptibility_plotqa_onevision.csv" ;;  # reuse
  esac
  note "qwen2.5-vl × $ds_tag E1"
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/extract_attention_mass.py \
      --model qwen2.5-vl-7b-instruct \
      --hf-model "$QWEN_HF" \
      --config "$cfg" \
      --susceptibility-csv "$susc" \
      --top-decile-n 100 --bottom-decile-n 100 \
      --bbox-file inputs/irrelevant_number_bboxes.json \
      --max-new-tokens 8 \
      >> "$LOG_DIR/phaseH_qwen25vl_${ds_tag}.log" 2>&1
done

uv run python scripts/analyze_attention_per_layer.py >> "$LOG" 2>&1
uv run python scripts/analyze_attention_patch.py --print-summary >> "$LOG" 2>&1

gcommit "Phase H: qwen2.5-vl-7b §7.1-7.3 on 5 main datasets"

# -----------------------------------------------------------------------------
# Phase I: Final 7-model × 5-dataset summary aggregation
# -----------------------------------------------------------------------------
phase "Phase I: Final summary aggregation"

uv run python scripts/build_e5e_e7_5dataset_summary.py --print >> "$LOG" 2>&1

# Save a comprehensive snapshot under docs/insights/_data/
uv run python -c "
import json
from pathlib import Path

DATASETS = ['experiment_e5e_tallyqa_full','experiment_e5e_chartqa_full',
            'experiment_e5e_mathvista_full','experiment_e7_plotqa_full',
            'experiment_e7_infographicvqa_full']
MODELS = ['llava-onevision-qwen2-7b-ov','llava-next-interleaved-7b',
          'qwen2.5-vl-7b-instruct','gemma3-27b-it',
          'internvl3-8b','qwen2.5-vl-32b-instruct','gemma3-4b-it']
def latest_run(exp,m):
    d=Path('outputs')/exp/m
    if not d.exists(): return None
    rs=[]
    for ts in d.iterdir():
        if not ts.is_dir(): continue
        s=ts/'summary.json'; p=ts/'predictions.jsonl'
        if s.exists() and p.exists(): rs.append((ts.stat().st_mtime,ts))
    return max(rs)[1] if rs else None

import csv
out_path=Path('docs/insights/_data/phase1_p0_v3_7model_5dataset_summary.csv')
out_path.parent.mkdir(parents=True,exist_ok=True)
with out_path.open('w',newline='') as fh:
    w=csv.writer(fh)
    w.writerow(['dataset','model','acc_b','adopt_a','df_a','em_a',
                'adopt_m','df_m','em_m','acc_d'])
    for ds in DATASETS:
        for m in MODELS:
            run=latest_run(ds,m)
            if run is None:
                w.writerow([ds,m,'','','','','','','',''])
                continue
            try:
                s=json.loads((run/'summary.json').read_text())
            except Exception:
                w.writerow([ds,m,'ERR','','','','','','',''])
                continue
            b=next((k for k in s if k=='target_only'),None)
            a=next((k for k in s if k.startswith('target_plus_irrelevant_number') and 'masked' not in k),None)
            mm=next((k for k in s if 'masked' in k),None)
            d=next((k for k in s if 'neutral' in k),None)
            row=[ds,m]
            for arm,key,met in [(b,'b','accuracy_exact'),
                                (a,'a','anchor_adoption_rate'),
                                (a,'a','anchor_direction_follow_rate'),
                                (a,'a','accuracy_exact'),
                                (mm,'m','anchor_adoption_rate'),
                                (mm,'m','anchor_direction_follow_rate'),
                                (mm,'m','accuracy_exact'),
                                (d,'d','accuracy_exact')]:
                row.append(s.get(arm,{}).get(met,'') if arm else '')
            w.writerow(row)
print(f'wrote {out_path}')
" >> "$LOG" 2>&1

gcommit "Phase I: 7-model × 5-dataset comprehensive summary"

# -----------------------------------------------------------------------------
# Phase J: Branch merge → master + push (per memory feedback_auto_branch_merge_push)
# -----------------------------------------------------------------------------
phase "Phase J: branch merge → master + push"

CUR_BRANCH="$(git branch --show-current)"
note "current branch: $CUR_BRANCH"
if [ "$CUR_BRANCH" = "master" ]; then
  note "already on master; just push"
  git push origin master >> "$LOG" 2>&1 || note "push warn (no remote?)"
else
  git checkout master >> "$LOG" 2>&1
  git merge --no-ff "$CUR_BRANCH" -m "merge $CUR_BRANCH: Phase 1 P0 v3 + extensions" >> "$LOG" 2>&1
  git push origin master >> "$LOG" 2>&1 || note "push warn (no remote?)"
  git checkout "$CUR_BRANCH" >> "$LOG" 2>&1
fi

note "==================================================================="
note "Phase 1 P0 v3 POST-PILOT MASTER QUEUE complete"
note "==================================================================="
