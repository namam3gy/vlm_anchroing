#!/usr/bin/env bash
# E6 cross-arch on Qwen2.5-VL — calibration filter ablation.
#
# Tests whether the wrong-base calibration filter is the root cause of weak
# cross-arch mitigation (vs prompt-format hypothesis). Recalibrates with
# anchor-positive filter (adopt(a) OR df(a)) on existing E7 predictions,
# re-runs SVD + peak-pick + Stage-4 chosen cell × 5 datasets at L=26 K=8 α=1.0,
# then compares Δdf vs current Phase 2 result.
#
# Pre-requisites (must exist):
#   outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/<ts>/predictions.jsonl
#   outputs/experiment_e7_infographicvqa_full/qwen2.5-vl-7b-instruct/<ts>/predictions.jsonl
#   (and 3 more E5e baselines for Stage-4 eval; see Phase 2 driver)
#
# Output dirs (suffix `_anchor_positive` to keep distinct from existing):
#   calibration_plotqa_anchor_positive/
#   calibration_infovqa_anchor_positive/
#   calibration_plotqa_infovqa_pooled_anchor_positive/v.pt + v_meta.json
#   _subspace/subspace_plotqa_infovqa_pooled_anchor_positive_K16.pt
#   _subspace/peak_layers_plotqa_infovqa_pooled_anchor_positive.json
#   sweep_subspace_<ds>_plotqa_infovqa_pooled_anchor_positive_chosen/
#   docs/insights/_data/stage4_final_per_dataset{_ci,}_qwen_anchor_positive.{csv,md}
#
# Budget: ~30 min calibrate × 2 (parallel GPU) + ~5 min SVD + Stage-4 chosen cell
#         × 5 datasets ~10-12h sharded 3-GPU. Total ~12-13h wall.
#
# Usage:
#   bash scripts/run_e6_filter_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Force HF cache-only — Qwen2.5-VL weights/processor/tokenizer all cached
# locally already; this bypasses transformers' is_base_mistral() HF API
# probe (which 429-rate-limits when too many parallel loads happen).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL=qwen2.5-vl-7b-instruct
HF=Qwen/Qwen2.5-VL-7B-Instruct
SCOPE=plotqa_infovqa_pooled_anchor_positive

PLOTQA_RUN_DIR=outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/20260502-022631
INFOVQA_RUN_DIR=outputs/experiment_e7_infographicvqa_full/qwen2.5-vl-7b-instruct/20260502-071849

E6_LOGS=outputs/e6_steering/$MODEL
mkdir -p "$E6_LOGS"
LOG="$E6_LOGS/_filter_ablation.log"
ts() { date +'%Y-%m-%d %H:%M:%S'; }
note() { printf "[%s] %s\n" "$(ts)" "$*" | tee -a "$LOG"; }

note "==== Filter ablation start ===="
note "filter=anchor-positive (adopt(a) OR df(a))"

# ---------------------------------------------------------------------------
# Step 1: calibrate-subspace × 2 datasets SEQUENTIAL (HF tokenizer load
# race-condition observed when two processes start within the same second).
# Pool sizes are small (PlotQA 278 / InfoVQA 34 anchor-positive sids), each
# calibrate takes ~5-10 min, so sequential is acceptable (~20 min total).
# ---------------------------------------------------------------------------
PLOTQA_GPU="${PLOTQA_GPU:-0}"
INFOVQA_GPU="${INFOVQA_GPU:-0}"

note "Step 1a — calibrate-subspace anchor-positive plotqa on GPU $PLOTQA_GPU"
CUDA_VISIBLE_DEVICES="$PLOTQA_GPU" uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" --hf-model "$HF" \
    --e5c-run-dir "$PLOTQA_RUN_DIR" \
    --predictions-path "$PLOTQA_RUN_DIR/predictions.jsonl" \
    --config configs/experiment_e7_plotqa_full.yaml \
    --dataset-tag plotqa_anchor_positive \
    --calibration-filter anchor-positive \
    --max-calibrate-pairs 350 \
    > "$E6_LOGS/_calibrate_plotqa_anchor_positive.log" 2>&1
PLOTQA_RC=$?
note "Step 1a done — plotqa rc=$PLOTQA_RC"
[ $PLOTQA_RC -eq 0 ] || { note "FATAL: plotqa calibrate failed (rc=$PLOTQA_RC)"; exit 2; }

note "Step 1b — calibrate-subspace anchor-positive infovqa on GPU $INFOVQA_GPU"
CUDA_VISIBLE_DEVICES="$INFOVQA_GPU" uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" --hf-model "$HF" \
    --e5c-run-dir "$INFOVQA_RUN_DIR" \
    --predictions-path "$INFOVQA_RUN_DIR/predictions.jsonl" \
    --config configs/experiment_e7_infographicvqa_full.yaml \
    --dataset-tag infovqa_anchor_positive \
    --calibration-filter anchor-positive \
    --max-calibrate-pairs 350 \
    > "$E6_LOGS/_calibrate_infovqa_anchor_positive.log" 2>&1
INFOVQA_RC=$?
note "Step 1b done — infovqa rc=$INFOVQA_RC"
[ $INFOVQA_RC -eq 0 ] || { note "FATAL: infovqa calibrate failed (rc=$INFOVQA_RC)"; exit 2; }

# ---------------------------------------------------------------------------
# Step 2: pool per-source D → pooled v.pt + v_meta.json
# ---------------------------------------------------------------------------
note "Step 2 — pool D matrices → calibration_${SCOPE}/v.pt"
POOLED_DIR="$E6_LOGS/calibration_${SCOPE}"
uv run python -c "
import json, torch
from pathlib import Path
base = Path('$E6_LOGS')
sources = ['plotqa_anchor_positive', 'infovqa_anchor_positive']
Dw_list, Da_list, nw, na = [], [], {}, {}
for tag in sources:
    Dw = torch.load(base / f'calibration_{tag}' / 'D_wrong.pt', weights_only=True)
    Da = torch.load(base / f'calibration_{tag}' / 'D_all.pt', weights_only=True)
    Dw_list.append(Dw); Da_list.append(Da)
    nw[tag] = int(Dw.shape[0]); na[tag] = int(Da.shape[0])
Dw = torch.cat(Dw_list, dim=0); Da = torch.cat(Da_list, dim=0)
v = torch.stack([Dw.float().mean(0), Da.float().mean(0)], dim=0)
out = Path('$POOLED_DIR'); out.mkdir(parents=True, exist_ok=True)
torch.save(v, out / 'v.pt')
(out / 'v_meta.json').write_text(json.dumps({
    'model': '$MODEL', 'hf_model': '$HF',
    'dataset_tag': '$SCOPE',
    'calibration_filter': 'anchor-positive',
    'n_wrong': int(Dw.shape[0]), 'n_all': int(Da.shape[0]),
    'n_wrong_per_source': nw, 'n_all_per_source': na,
    'source_tags': sources,
    'n_layers': int(v.shape[1]), 'd_model': int(v.shape[2]),
    'D_wrong_shape': list(Dw.shape), 'D_all_shape': list(Da.shape),
    '_note': 'D_wrong semantically = anchor-positive subset (adopt OR df).',
}, indent=2))
print(f'pooled v.pt saved at {out}/v.pt shape={tuple(v.shape)} n_priority={Dw.shape[0]} n_all={Da.shape[0]}')
" 2>&1 | tail -2
note "Step 2 done"

# ---------------------------------------------------------------------------
# Step 3: SVD pool
# ---------------------------------------------------------------------------
note "Step 3 — SVD compute"
mkdir -p "$E6_LOGS/_subspace"
uv run python scripts/e6_compute_subspace.py \
    --model "$MODEL" \
    --scope "$SCOPE" \
    --tags plotqa_anchor_positive,infovqa_anchor_positive \
    --K-max 16 \
    > "$E6_LOGS/_subspace/_svd_anchor_positive.log" 2>&1
note "Step 3 done"

SUBSPACE_PT="$E6_LOGS/_subspace/subspace_${SCOPE}_K16.pt"
[ -f "$SUBSPACE_PT" ] || { note "FATAL: $SUBSPACE_PT missing"; exit 3; }

# ---------------------------------------------------------------------------
# Step 4: peak-pick (sanity — verify L=26 stays dominant or shifts)
# ---------------------------------------------------------------------------
PEAK_JSON="$E6_LOGS/_subspace/peak_layers_${SCOPE}.json"
note "Step 4 — peak-layer pick"
uv run python scripts/e6_pick_peak_layers.py \
    --model "$MODEL" --tag "$SCOPE" --top-k 5 \
    --out "$PEAK_JSON" 2>&1 | tail -15

# ---------------------------------------------------------------------------
# Step 5: Stage-4 chosen cell L=26 K=8 α=1.0 × 5 datasets
# ---------------------------------------------------------------------------
note "Step 5 — Stage-4 chosen cell sweep × 5 datasets"
LAYERS="26"
KS="8"
ALPHAS="1.0"
GPUS="${ABLATION_GPUS:-0,1,2}"
MAX_SAMPLES="${ABLATION_MAX_SAMPLES:-500}"

latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r tsdir; do
    f="$model_dir/$tsdir/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$tsdir"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

run_stage4() {
  local ds="$1" cfg="$2" exp="$3"
  local tsdir; tsdir="$(latest_run "$exp")"
  [ -n "$tsdir" ] || { note "ERR: no $MODEL run for $exp"; return 1; }
  local preds="outputs/$exp/$MODEL/$tsdir/predictions.jsonl"

  local stage4_dir="$E6_LOGS/sweep_subspace_${ds}_${SCOPE}_chosen"
  if [ -f "$stage4_dir/predictions.jsonl" ]; then
    note "skip Stage-4 $ds (exists)"
    return 0
  fi

  note "Stage-4 $ds (cell L=26 K=8 α=1.0, $MAX_SAMPLES samples, GPUs=$GPUS)"
  uv run python scripts/run_sweep_subspace_sharded.py \
      --config "$cfg" \
      --model "$MODEL" --hf-model "$HF" \
      --predictions-path "$preds" \
      --dataset-tag "$ds" \
      --subspace-path "$SUBSPACE_PT" \
      --subspace-scope "$SCOPE" \
      --sweep-layers "$LAYERS" --sweep-ks "$KS" --sweep-alphas "$ALPHAS" \
      --max-samples "$MAX_SAMPLES" \
      --gpus "$GPUS" >> "$LOG" 2>&1

  local auto_dir="$E6_LOGS/sweep_subspace_${ds}_${SCOPE}"
  if [ -d "$auto_dir" ] && [ ! -d "$stage4_dir" ]; then
    mv "$auto_dir" "$stage4_dir"
  fi
  note "$ds Stage-4 done -> $stage4_dir"
}

run_stage4 mathvista       configs/experiment_e5e_mathvista_full.yaml         experiment_e5e_mathvista_full
run_stage4 chartqa         configs/experiment_e5e_chartqa_full.yaml           experiment_e5e_chartqa_full
run_stage4 infographicvqa  configs/experiment_e7_infographicvqa_full.yaml     experiment_e7_infographicvqa_full
run_stage4 plotqa          configs/experiment_e7_plotqa_full.yaml             experiment_e7_plotqa_full
run_stage4 tallyqa         configs/experiment_e5e_tallyqa_full.yaml           experiment_e5e_tallyqa_full

# ---------------------------------------------------------------------------
# Step 6: aggregate via env-var override
# ---------------------------------------------------------------------------
note "Step 6 — aggregator + paired-bootstrap CI"
E6_STAGE4_MODEL="$MODEL" \
E6_STAGE4_SCOPE="$SCOPE" \
E6_STAGE4_OUTPUT_SUFFIX="_qwen_anchor_positive" \
uv run python scripts/build_e6_stage4_summary.py >> "$LOG" 2>&1

E6_STAGE4_MODEL="$MODEL" \
E6_STAGE4_SCOPE="$SCOPE" \
E6_STAGE4_OUTPUT_SUFFIX="_qwen_anchor_positive" \
uv run python scripts/build_e6_stage4_bootstrap_ci.py >> "$LOG" 2>&1

note "==== Filter ablation COMPLETE ===="
note "  Summary: docs/insights/_data/stage4_final_per_dataset_qwen_anchor_positive.{csv,md}"
note "  Paired CI: docs/insights/_data/stage4_final_per_dataset_ci_qwen_anchor_positive.{csv,md}"
note "  Compare vs: docs/insights/_data/stage4_final_per_dataset_qwen.{csv,md} (wrong-base baseline)"
