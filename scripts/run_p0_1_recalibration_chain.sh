#!/usr/bin/env bash
# P0-1 Phase B'/C' re-calibration chain.
# 1. TallyQA inference (Qwen3-VL-Instruct) — adds 3rd calibration dataset
# 2. TallyQA calibrate-subspace — extends D matrix
# 3. Pooled SVD (TallyQA + PlotQA + InfoVQA) — produces new V_K[L=*]
# 4. Bridge inference re-run (Instruct + Thinking) with K=16 coefficients,
#    layers L∈{14, 20, 25, 29, 30, 33, 34}
#
# Cost estimate: ~10-12 H200-hour
# Run: nohup bash scripts/run_p0_1_recalibration_chain.sh > outputs/p0_1_recal.log 2>&1 &

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing/.claude/worktrees/phase5+p0-1-gamma-beta-bridge

LOGS=outputs/p0_1_calibration_qwen3vl/_logs
E6_LOGS=outputs/e6_steering/qwen3-vl-8b-instruct
mkdir -p "$LOGS" "$E6_LOGS" outputs/gamma_beta_bridge

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === P0-1 Phase B'/C' re-calibration chain ==="

# Existing calibration paths (PlotQA + InfoVQA preds + D matrices already computed)
PLOTQA_PRED=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct/20260510-005948/predictions.jsonl
INFOVQA_PRED=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_infovqa/qwen3-vl-8b-instruct/20260510-023910/predictions.jsonl

# Verify existing artifacts still on disk
for p in "$PLOTQA_PRED" "$INFOVQA_PRED"; do
    if [ ! -f "$p" ]; then
        echo "[$(ts)] FATAL: $p missing — need Phase B PlotQA/InfoVQA preds"
        exit 1
    fi
done
if [ ! -f outputs/e6_steering/qwen3-vl-8b-instruct/calibration_plotqa/D_wrong.pt ]; then
    echo "[$(ts)] FATAL: PlotQA D_wrong.pt missing — need Phase B'"
    exit 1
fi
if [ ! -f outputs/e6_steering/qwen3-vl-8b-instruct/calibration_infovqa/D_wrong.pt ]; then
    echo "[$(ts)] FATAL: InfoVQA D_wrong.pt missing"
    exit 1
fi
echo "[$(ts)] Existing PlotQA + InfoVQA artifacts verified"

# ---------------------------------------------------------------------------
# B'1: TallyQA Qwen3-VL-Instruct inference (~3h on n=5000)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B'1: TallyQA inference ==="
uv run python scripts/run_experiment.py \
    --config configs/p0_1_calibration_qwen3vl_tallyqa.yaml \
    --max-samples 5000 \
    > "$LOGS/tallyqa.log" 2>&1
echo "[$(ts)] B'1 done"

TALLYQA_BASE=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_tallyqa/qwen3-vl-8b-instruct
TALLYQA_RUN_DIR=""
for d in $(ls -dt "$TALLYQA_BASE"/*/ 2>/dev/null); do
    if [ -f "${d}predictions.jsonl" ] && [ -f "${d}summary.json" ]; then
        n=$(wc -l < "${d}predictions.jsonl")
        if [ "$n" -ge 1000 ]; then
            TALLYQA_RUN_DIR="$d"
            break
        fi
    fi
done
if [ -z "$TALLYQA_RUN_DIR" ]; then
    echo "[$(ts)] FATAL: TallyQA full-run dir not found under $TALLYQA_BASE"
    exit 2
fi
TALLYQA_PRED="${TALLYQA_RUN_DIR}predictions.jsonl"
echo "[$(ts)] TallyQA preds: $TALLYQA_PRED"

# ---------------------------------------------------------------------------
# B'2: TallyQA calibrate-subspace (~1.5h)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B'2: TallyQA calibrate-subspace ==="
uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model qwen3-vl-8b-instruct \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --e5c-run-dir "$TALLYQA_RUN_DIR" \
    --predictions-path "$TALLYQA_PRED" \
    --config configs/p0_1_calibration_qwen3vl_tallyqa.yaml \
    --dataset-tag tallyqa \
    --max-calibrate-pairs 5000 \
    > "$E6_LOGS/_calibrate_tallyqa.log" 2>&1
echo "[$(ts)] B'2 done"

# ---------------------------------------------------------------------------
# B'3: Pooled SVD across 3 datasets (TallyQA + PlotQA + InfoVQA)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B'3: SVD pooled (tally + plotqa + infovqa) ==="
mkdir -p "$E6_LOGS/_subspace"
uv run python scripts/e6_compute_subspace.py \
    --model qwen3-vl-8b-instruct \
    --scope tally_plotqa_infovqa_pooled \
    --tags tallyqa,plotqa,infovqa \
    --K-max 16 \
    > "$E6_LOGS/_subspace/_svd_3pool.log" 2>&1
echo "[$(ts)] B'3 done"

NEW_SUBSPACE=outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_tally_plotqa_infovqa_pooled_K16.pt
if [ ! -f "$NEW_SUBSPACE" ]; then
    echo "[$(ts)] FATAL: $NEW_SUBSPACE not found after SVD"
    exit 3
fi
echo "[$(ts)] New V_K artifact: $NEW_SUBSPACE"

# ---------------------------------------------------------------------------
# C'-smoke: smoke run new bridge script (n=5)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C'-smoke: bridge inference n=5 ==="
SMOKE_TAG="smoke_recal_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --max-samples 5 \
    --output-tag "$SMOKE_TAG" \
    > outputs/gamma_beta_bridge/_smoke_recal.log 2>&1
SMOKE_JSONL=outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${SMOKE_TAG}/amplitude_per_trace.jsonl
if [ ! -f "$SMOKE_JSONL" ]; then
    echo "[$(ts)] FATAL: smoke jsonl missing at $SMOKE_JSONL"
    exit 4
fi
N_SMOKE=$(wc -l < "$SMOKE_JSONL")
echo "[$(ts)] C'-smoke done — $N_SMOKE records"
if [ "$N_SMOKE" -lt 8 ]; then
    echo "[$(ts)] FATAL: expected >=8 smoke records, got $N_SMOKE"
    exit 5
fi

# ---------------------------------------------------------------------------
# C'3: full Instruct bridge re-run (~3h with 7 layers + K=16 coefs)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C'3: Instruct bridge re-run ==="
INSTRUCT_TAG="instruct_recal_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --output-tag "$INSTRUCT_TAG" \
    > outputs/gamma_beta_bridge/_instruct_recal.log 2>&1
echo "[$(ts)] C'3 done"

# ---------------------------------------------------------------------------
# C'4: full Thinking bridge re-run (~3-6h)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C'4: Thinking bridge re-run ==="
THINKING_TAG="thinking_recal_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-thinking \
    --output-tag "$THINKING_TAG" \
    > outputs/gamma_beta_bridge/_thinking_recal.log 2>&1
echo "[$(ts)] C'4 done"

echo "[$(ts)] === Phase B'/C' COMPLETE ==="
echo "  V_K subspace: $NEW_SUBSPACE"
echo "  Instruct: outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${INSTRUCT_TAG}/amplitude_per_trace.jsonl"
echo "  Thinking: outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/${THINKING_TAG}/amplitude_per_trace.jsonl"
