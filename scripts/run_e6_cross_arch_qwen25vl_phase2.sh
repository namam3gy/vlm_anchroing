#!/usr/bin/env bash
# E6 cross-arch on Qwen2.5-VL-7B-Instruct — Phase 2 Stage-4 driver.
#
# Mirrors OneVision's Stage-4 chosen-cell × 5-dataset sweep + paired-bootstrap
# CI aggregation. Pre-decision per user 2026-05-18: chosen cell = L=26 K=8 α=1.0
# (identical to OneVision; recipe-portable). Phase 1 partial aggregator
# confirmed at 03:37 KST — Phase 1 full completion will not flip ranking
# (1.78pp margin over rank-2).
#
# Per dataset: run sweep-subspace at chosen cell on the existing E7/E5e
# baseline predictions (b/a/m/d), capping samples at the full eligible pool.
# Each shard handles a sid round-robin slice. 3-GPU sharded per current pod.
#
# Output structure (mirrors OneVision):
#   outputs/e6_steering/qwen2.5-vl-7b-instruct/
#     sweep_subspace_<ds>_plotqa_infovqa_pooled_chosen/predictions.jsonl
#
# Aggregator outputs:
#   docs/insights/_data/stage4_final_per_dataset_qwen.csv      (point estimates)
#   docs/insights/_data/stage4_final_per_dataset_qwen.md       (markdown table)
#   docs/insights/_data/stage4_final_per_dataset_ci_qwen.csv   (95% + Bonferroni-20 CI)
#   docs/insights/_data/stage4_final_per_dataset_ci_qwen.md
#   docs/insights/_data/stage4_final_bootstrap_draws_qwen.npz  (raw draws)
#
# Budget estimate: TallyQA (largest, ~38k eligible) dominates; full sweep
# at 1 cell × 3 shards × ~1.3 fwd/s ≈ 8h wall for TallyQA. Smaller datasets
# ~30-60min each. Total: ~10-12h wall.
#
# Usage:
#   bash scripts/run_e6_cross_arch_qwen25vl_phase2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=qwen2.5-vl-7b-instruct
HF=Qwen/Qwen2.5-VL-7B-Instruct
SCOPE=plotqa_infovqa_pooled
SUBSPACE_PT="outputs/e6_steering/$MODEL/_subspace/subspace_${SCOPE}_K16.pt"

# Chosen cell (pre-decision 2026-05-18, identical to OneVision).
LAYERS="26"
KS="8"
ALPHAS="1.0"
GPUS="${PHASE2_GPUS:-0,1,2}"
MAX_SAMPLES="${PHASE2_MAX_SAMPLES:-5000}"

LOG_DIR=outputs/e6_steering/$MODEL
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/_phase2_stage4.log"
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

[ -f "$SUBSPACE_PT" ] || { note "ERR: subspace .pt missing at $SUBSPACE_PT (run Phase 0)"; exit 1; }

latest_run() {
  local exp="$1"
  local model_dir="outputs/$exp/$MODEL"
  [ -d "$model_dir" ] || { echo ""; return; }
  ls -1 "$model_dir" 2>/dev/null | while read -r ts; do
    f="$model_dir/$ts/predictions.jsonl"
    [ -f "$f" ] && printf "%d\t%s\n" "$(wc -l <"$f")" "$ts"
  done | sort -k1,1n -k2,2 | tail -1 | awk '{print $2}'
}

run_stage4() {
  local ds="$1" cfg="$2" exp="$3"
  local ts; ts="$(latest_run "$exp")"
  [ -n "$ts" ] || { note "ERR: no $MODEL run for $exp"; return 1; }
  local preds="outputs/$exp/$MODEL/$ts/predictions.jsonl"

  local stage4_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${SCOPE}_chosen"
  if [ -f "$stage4_dir/predictions.jsonl" ]; then
    note "skip Stage-4 $ds (predictions.jsonl already exists at $stage4_dir)"
    return 0
  fi

  note "==== Stage-4 $ds (chosen cell L=26 K=8 α=1.0, max-samples=$MAX_SAMPLES, GPUs=$GPUS) ===="
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

  # The driver writes to sweep_subspace_<ds>_<scope> by default; rename to
  # sweep_subspace_<ds>_<scope>_chosen to match aggregator's expected layout.
  local auto_dir="outputs/e6_steering/$MODEL/sweep_subspace_${ds}_${SCOPE}"
  if [ -d "$auto_dir" ] && [ ! -d "$stage4_dir" ]; then
    mv "$auto_dir" "$stage4_dir"
  fi
  note "$ds Stage-4 done → $stage4_dir"
}

note "==== Phase 2 Stage-4 chain start (chosen L=26 K=8 α=1.0) ===="

# Order: smaller datasets first for faster validation, TallyQA last.
run_stage4 mathvista       configs/experiment_e5e_mathvista_full.yaml         experiment_e5e_mathvista_full
run_stage4 chartqa         configs/experiment_e5e_chartqa_full.yaml           experiment_e5e_chartqa_full
run_stage4 infographicvqa  configs/experiment_e7_infographicvqa_full.yaml     experiment_e7_infographicvqa_full
run_stage4 plotqa          configs/experiment_e7_plotqa_full.yaml             experiment_e7_plotqa_full
run_stage4 tallyqa         configs/experiment_e5e_tallyqa_full.yaml           experiment_e5e_tallyqa_full

note "==== All 5 Stage-4 sweeps complete — running summary + CI aggregators ===="

E6_STAGE4_MODEL="$MODEL" \
E6_STAGE4_SCOPE="$SCOPE" \
E6_STAGE4_OUTPUT_SUFFIX="_qwen" \
uv run python scripts/build_e6_stage4_summary.py >> "$LOG" 2>&1

E6_STAGE4_MODEL="$MODEL" \
E6_STAGE4_SCOPE="$SCOPE" \
E6_STAGE4_OUTPUT_SUFFIX="_qwen" \
uv run python scripts/build_e6_stage4_bootstrap_ci.py >> "$LOG" 2>&1

note "==== Phase 2 COMPLETE ===="
note "  Summary: docs/insights/_data/stage4_final_per_dataset_qwen.{csv,md}"
note "  Paired CI: docs/insights/_data/stage4_final_per_dataset_ci_qwen.{csv,md}"
note "  Raw draws: docs/insights/_data/stage4_final_bootstrap_draws_qwen.npz"
