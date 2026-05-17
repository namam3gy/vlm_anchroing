#!/usr/bin/env bash
# E6 cross-arch on Qwen2.5-VL-7B-Instruct — Phase 0 driver.
#
# Mirrors scripts/run_p0_1_resume_b3.sh (P0-1 γ-β bridge calibration chain
# for Qwen3-VL) — same calibrate-subspace → SVD pool → peak-pick recipe.
# Reads from existing E7 main-panel predictions (b/a-S1/m-S1/d) for
# Qwen2.5-VL-7B-Instruct; no new prediction inference required.
#
# Phase 0 = identify L bin center for the 27-cell pilot grid by picking
# top-K layers ranked by ||v_wrong[L]|| (per-layer (a − m) residual diff
# norm) on PlotQA + InfoVQA pooled wrong-base + 4-cond eligible.
#
# Output:
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_plotqa/
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_infovqa/
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_plotqa_infovqa_pooled/
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/peak_layers_plotqa_infovqa_pooled.json
#
# Budget: ~4 H200-hour per source pass (OneVision was ~4 min/source on H200
# with similar n) + ~5 min SVD + ~1 min peak pick. Total ≤ 1 H200-hour
# wall-clock if both sources run sequentially on one GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_e6_cross_arch_qwen25vl_phase0.sh

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=qwen2.5-vl-7b-instruct
HF_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

PLOTQA_RUN_DIR=outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/20260502-022631
INFOVQA_RUN_DIR=outputs/experiment_e7_infographicvqa_full/qwen2.5-vl-7b-instruct/20260502-071849

PLOTQA_PRED="$PLOTQA_RUN_DIR/predictions.jsonl"
INFOVQA_PRED="$INFOVQA_RUN_DIR/predictions.jsonl"

E6_LOGS=outputs/e6_steering/qwen2.5-vl-7b-instruct
mkdir -p "$E6_LOGS"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === E6 cross-arch Phase 0 — Qwen2.5-VL-7B-Instruct ==="
echo "[$(ts)]    PlotQA preds:  $PLOTQA_PRED"
echo "[$(ts)]    InfoVQA preds: $INFOVQA_PRED"

if [ ! -f "$PLOTQA_PRED" ]; then
    echo "[$(ts)] FATAL: PlotQA preds missing — $PLOTQA_PRED"
    exit 1
fi
if [ ! -f "$INFOVQA_PRED" ]; then
    echo "[$(ts)] FATAL: InfoVQA preds missing — $INFOVQA_PRED"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 0a-{1,2}: calibrate-subspace plotqa + infovqa in parallel on
# separate GPUs (5-GPU host; PLOTQA_GPU + INFOVQA_GPU configurable).
# ---------------------------------------------------------------------------
PLOTQA_GPU="${PLOTQA_GPU:-1}"
INFOVQA_GPU="${INFOVQA_GPU:-2}"

echo "[$(ts)] === Step 0a (parallel): plotqa on GPU $PLOTQA_GPU + infovqa on GPU $INFOVQA_GPU ==="

(CUDA_VISIBLE_DEVICES="$PLOTQA_GPU" uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" \
    --hf-model "$HF_MODEL" \
    --e5c-run-dir "$PLOTQA_RUN_DIR" \
    --predictions-path "$PLOTQA_PRED" \
    --config configs/experiment_e7_plotqa_full.yaml \
    --dataset-tag plotqa \
    --max-calibrate-pairs 5000 \
    > "$E6_LOGS/_calibrate_plotqa.log" 2>&1) &
PLOTQA_PID=$!
echo "[$(ts)]    plotqa PID $PLOTQA_PID on GPU $PLOTQA_GPU"

(CUDA_VISIBLE_DEVICES="$INFOVQA_GPU" uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model "$MODEL" \
    --hf-model "$HF_MODEL" \
    --e5c-run-dir "$INFOVQA_RUN_DIR" \
    --predictions-path "$INFOVQA_PRED" \
    --config configs/experiment_e7_infographicvqa_full.yaml \
    --dataset-tag infovqa \
    --max-calibrate-pairs 5000 \
    > "$E6_LOGS/_calibrate_infovqa.log" 2>&1) &
INFOVQA_PID=$!
echo "[$(ts)]    infovqa PID $INFOVQA_PID on GPU $INFOVQA_GPU"

wait $PLOTQA_PID
PLOTQA_RC=$?
wait $INFOVQA_PID
INFOVQA_RC=$?
echo "[$(ts)] Step 0a done — plotqa rc=$PLOTQA_RC, infovqa rc=$INFOVQA_RC"
if [ $PLOTQA_RC -ne 0 ] || [ $INFOVQA_RC -ne 0 ]; then
    echo "[$(ts)] FATAL: at least one calibrate-subspace failed — see logs"
    exit 2
fi

# ---------------------------------------------------------------------------
# Step 0b': pool per-source D matrices → canonical pooled v.pt + v_meta.json
# (e6_compute_subspace.py only writes the SVD subspace_*.pt; e6_pick_peak_layers.py
# requires a calibration_<tag>/v.pt entry, which we synthesise here to match the
# OneVision pooled directory schema.)
# ---------------------------------------------------------------------------
POOLED_DIR="$E6_LOGS/calibration_plotqa_infovqa_pooled"
echo "[$(ts)] === Step 0b': pool per-source D → $POOLED_DIR/v.pt ==="
uv run python -c "
import json, torch
from pathlib import Path
base = Path('$E6_LOGS')
sources = ['plotqa', 'infovqa']
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
    'model': '$MODEL', 'hf_model': '$HF_MODEL',
    'dataset_tag': 'plotqa_infovqa_pooled',
    'n_wrong': int(Dw.shape[0]), 'n_all': int(Da.shape[0]),
    'n_wrong_per_source': nw, 'n_all_per_source': na,
    'source_tags': sources,
    'n_layers': int(v.shape[1]), 'd_model': int(v.shape[2]),
    'D_wrong_shape': list(Dw.shape), 'D_all_shape': list(Da.shape),
    '_note': 'Pooled by concatenating per-source D matrices and re-meaning.',
}, indent=2))
print(f'pooled v.pt saved at {out}/v.pt shape={tuple(v.shape)} n_wrong={Dw.shape[0]} n_all={Da.shape[0]}')
" 2>&1 | tail -2
echo "[$(ts)] Step 0b' done"

# ---------------------------------------------------------------------------
# Step 0b: SVD pool (plotqa, infovqa)
# ---------------------------------------------------------------------------
echo "[$(ts)] === Step 0b: SVD pool ==="
mkdir -p "$E6_LOGS/_subspace"
uv run python scripts/e6_compute_subspace.py \
    --model "$MODEL" \
    --scope plotqa_infovqa_pooled \
    --tags plotqa,infovqa \
    --K-max 16 \
    > "$E6_LOGS/_subspace/_svd.log" 2>&1
echo "[$(ts)] Step 0b done — log $E6_LOGS/_subspace/_svd.log"

SUBSPACE_PT="$E6_LOGS/_subspace/subspace_plotqa_infovqa_pooled_K16.pt"
if [ ! -f "$SUBSPACE_PT" ]; then
    echo "[$(ts)] FATAL: $SUBSPACE_PT not found after SVD"
    exit 3
fi
echo "[$(ts)] V_K artifact: $SUBSPACE_PT"

# ---------------------------------------------------------------------------
# Step 0c: peak-layer pick (top-5)
# ---------------------------------------------------------------------------
echo "[$(ts)] === Step 0c: peak-layer pick ==="
PEAK_JSON="$E6_LOGS/_subspace/peak_layers_plotqa_infovqa_pooled.json"
uv run python scripts/e6_pick_peak_layers.py \
    --model "$MODEL" \
    --tag plotqa_infovqa_pooled \
    --top-k 5 \
    --out "$PEAK_JSON"

echo "[$(ts)] Step 0c done — peak layers at $PEAK_JSON:"
cat "$PEAK_JSON" || true

# ---------------------------------------------------------------------------
# Norm-profile sanity print
# ---------------------------------------------------------------------------
NORM_CSV="$E6_LOGS/calibration_plotqa_infovqa_pooled/norms_per_layer.csv"
if [ -f "$NORM_CSV" ]; then
    echo "[$(ts)] === norms_per_layer.csv head ==="
    head -5 "$NORM_CSV"
    echo "[$(ts)] === top-10 layers by ||v_wrong[L]|| ==="
    awk -F, 'NR>1 {print $0}' "$NORM_CSV" | sort -t, -k2,2 -rn | head -10
fi

echo "[$(ts)] === Phase 0 COMPLETE ==="
echo "  V_K subspace:  $SUBSPACE_PT"
echo "  Peak layers:   $PEAK_JSON"
echo "  Norm profile:  $NORM_CSV"
