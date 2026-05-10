#!/usr/bin/env bash
# P0-1 phase B + C chain runner.
#
# Steps (sequential, abort on first failure):
#   wait_B1: wait until B1 PlotQA inference (already running) writes summary.json
#   B2:      Qwen3-VL-Instruct InfoVQA calibration inference
#   B3a:     calibrate-subspace plotqa  (consumes B1 preds)
#   B3b:     calibrate-subspace infovqa (consumes B2 preds)
#   B4:      SVD for V_K[L=*] K=8 from PlotQA + InfoVQA pooled D_wrong
#   C2-smoke: GPU smoke run of bridge inference script (n=5)
#   C3:      full Instruct bridge inference
#   C4:      full Thinking bridge inference (long)
#
# IMPORTANT: run_experiment.py appends config_path.stem as a subdir under
# output_root, so actual paths look like
#   outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct/<ts>/
# The chain handles this layout for both PlotQA and InfoVQA configs.
#
# Run: nohup bash scripts/run_p0_1_chain.sh > outputs/p0_1_chain.log 2>&1 &

set -euo pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing/.claude/worktrees/phase5+p0-1-gamma-beta-bridge

LOGS=outputs/p0_1_calibration_qwen3vl/_logs
E6_LOGS=outputs/e6_steering/qwen3-vl-8b-instruct
PLOTQA_BASE=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct
INFOVQA_BASE=outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_infovqa/qwen3-vl-8b-instruct
mkdir -p "$LOGS" "$E6_LOGS"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === P0-1 chain started ==="

# ---------------------------------------------------------------------------
# wait_B1 — block until B1 PlotQA summary.json exists with full-run size
# (smoke runs would have ~5-20 records; we filter on predictions.jsonl line count
# >= 1000 to be sure we're picking the n=5000 full run, not a stray smoke.)
# ---------------------------------------------------------------------------
PLOTQA_RUN_DIR=""
echo "[$(ts)] === wait_B1: polling for $PLOTQA_BASE/<ts>/summary.json (full run only) ==="
while true; do
    for d in $(ls -dt "$PLOTQA_BASE"/*/ 2>/dev/null || true); do
        if [ -f "${d}summary.json" ] && [ -f "${d}predictions.jsonl" ]; then
            n=$(wc -l < "${d}predictions.jsonl")
            if [ "$n" -ge 1000 ]; then
                PLOTQA_RUN_DIR="$d"
                break 2
            fi
        fi
    done
    sleep 60
done

PLOTQA_PRED="${PLOTQA_RUN_DIR}predictions.jsonl"
echo "[$(ts)] B1 detected — PlotQA predictions: $PLOTQA_PRED"

# ---------------------------------------------------------------------------
# B2: InfoVQA Qwen3-VL-Instruct inference
# ---------------------------------------------------------------------------
echo "[$(ts)] === B2: InfoVQA inference ==="
uv run python scripts/run_experiment.py \
    --config configs/p0_1_calibration_qwen3vl_infovqa.yaml \
    --max-samples 5000 \
    > "$LOGS/infovqa.log" 2>&1
echo "[$(ts)] B2 done"

# Identify InfoVQA run dir (most recent under InfoVQA base)
INFOVQA_RUN_DIR=""
for d in $(ls -dt "$INFOVQA_BASE"/*/ 2>/dev/null || true); do
    if [ -f "${d}predictions.jsonl" ] && [ -f "${d}summary.json" ]; then
        n=$(wc -l < "${d}predictions.jsonl")
        if [ "$n" -ge 100 ]; then
            INFOVQA_RUN_DIR="$d"
            break
        fi
    fi
done
if [ -z "$INFOVQA_RUN_DIR" ]; then
    echo "[$(ts)] FATAL: cannot find InfoVQA run dir under $INFOVQA_BASE"
    ls -la "$INFOVQA_BASE"/ 2>&1 || true
    exit 2
fi
INFOVQA_PRED="${INFOVQA_RUN_DIR}predictions.jsonl"
echo "[$(ts)] B2 InfoVQA predictions: $INFOVQA_PRED"

# ---------------------------------------------------------------------------
# B3a: calibrate-subspace plotqa  (via sharded driver for arg/path handling)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B3a: calibrate-subspace plotqa ==="
uv run python scripts/run_calibrate_subspace_sharded.py \
    --config configs/p0_1_calibration_qwen3vl_plotqa.yaml \
    --model qwen3-vl-8b-instruct \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --predictions-path "$PLOTQA_PRED" \
    --dataset-tag plotqa \
    --max-calibrate-pairs 5000 \
    --gpus 0 \
    > "$E6_LOGS/_calibrate_plotqa.log" 2>&1
echo "[$(ts)] B3a done"

# ---------------------------------------------------------------------------
# B3b: calibrate-subspace infovqa  (via sharded driver)
# ---------------------------------------------------------------------------
echo "[$(ts)] === B3b: calibrate-subspace infovqa ==="
uv run python scripts/run_calibrate_subspace_sharded.py \
    --config configs/p0_1_calibration_qwen3vl_infovqa.yaml \
    --model qwen3-vl-8b-instruct \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --predictions-path "$INFOVQA_PRED" \
    --dataset-tag infovqa \
    --max-calibrate-pairs 5000 \
    --gpus 0 \
    > "$E6_LOGS/_calibrate_infovqa.log" 2>&1
echo "[$(ts)] B3b done"

# ---------------------------------------------------------------------------
# B4: SVD
# ---------------------------------------------------------------------------
echo "[$(ts)] === B4: SVD ==="
mkdir -p "$E6_LOGS/_subspace"
uv run python scripts/e6_compute_subspace.py \
    --model qwen3-vl-8b-instruct \
    --scope plotqa_infovqa_pooled \
    --tags plotqa,infovqa \
    --K-max 16 \
    > "$E6_LOGS/_subspace/_svd.log" 2>&1
echo "[$(ts)] B4 done"

SUBSPACE_PT=outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt
if [ ! -f "$SUBSPACE_PT" ]; then
    echo "[$(ts)] FATAL: $SUBSPACE_PT not found after SVD"
    exit 3
fi
echo "[$(ts)] V_K artifact: $SUBSPACE_PT"

# ---------------------------------------------------------------------------
# C2-smoke: GPU smoke for bridge inference (n=5)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C2-smoke: bridge inference n=5 (Instruct) ==="
SMOKE_TAG="smoke_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --max-samples 5 \
    --output-tag "$SMOKE_TAG" \
    > outputs/gamma_beta_bridge/_smoke.log 2>&1
SMOKE_JSONL=outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${SMOKE_TAG}/amplitude_per_trace.jsonl
if [ ! -f "$SMOKE_JSONL" ]; then
    echo "[$(ts)] FATAL: smoke jsonl missing at $SMOKE_JSONL"
    exit 4
fi
N_SMOKE=$(wc -l < "$SMOKE_JSONL")
echo "[$(ts)] C2-smoke done — $N_SMOKE records"
if [ "$N_SMOKE" -lt 8 ]; then
    echo "[$(ts)] FATAL: expected ≥8 smoke records (5 sids × 2 cond), got $N_SMOKE"
    exit 5
fi

# ---------------------------------------------------------------------------
# C3: full Instruct bridge inference
# ---------------------------------------------------------------------------
echo "[$(ts)] === C3: Instruct bridge inference (full) ==="
INSTRUCT_TAG="instruct_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-instruct \
    --output-tag "$INSTRUCT_TAG" \
    > outputs/gamma_beta_bridge/_instruct.log 2>&1
echo "[$(ts)] C3 done"

# ---------------------------------------------------------------------------
# C4: full Thinking bridge inference (long, ~12 h)
# ---------------------------------------------------------------------------
echo "[$(ts)] === C4: Thinking bridge inference (full, long) ==="
THINKING_TAG="thinking_$(date +%Y%m%d-%H%M%S)"
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml \
    --models qwen3-vl-8b-thinking \
    --output-tag "$THINKING_TAG" \
    > outputs/gamma_beta_bridge/_thinking.log 2>&1
echo "[$(ts)] C4 done"

echo "[$(ts)] === P0-1 chain COMPLETE ==="
echo "Artifacts:"
echo "  V_K subspace: $SUBSPACE_PT"
echo "  Instruct amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-instruct/${INSTRUCT_TAG}/amplitude_per_trace.jsonl"
echo "  Thinking amplitudes: outputs/gamma_beta_bridge/qwen3-vl-8b-thinking/${THINKING_TAG}/amplitude_per_trace.jsonl"
