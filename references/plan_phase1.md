# Phase 1 runbook — paper consistency push (2026-05-02 restructure)

This is the operational runbook for next-session Phase 1 work. Reads in
sequence with `references/project.md §0.4` (paper architecture) and
`references/roadmap.md §7` (Phase-structured priority queue).

## 0. Context (read first)

Decision summary (2026-05-02):

- **Main model**: `llava-interleave-7b` — every section primary
- **Sub panel**: + `qwen2.5-vl-7b-instruct` + `gemma3-27b-it`
- **5-dataset main matrix**: TallyQA + ChartQA + MathVista + **PlotQA** + **InfographicVQA**, all at n=5000 target (or full numeric where smaller)
- **Dropped**: VQAv2 (legacy, multi-GT). Existing 7-model VQAv2 → appendix only.
- **TallyQA**: re-run at n=5000 stratified (was 38k full); legacy 38k archive at `outputs/_legacy_tallyqa_n38k/`
- **§7.4.5 mitigation**: recalibrate on PlotQA + InfoVQA pooled, full gt range (no [0,8] restriction); sweep cap raised 500 → 5000 wrong-base (statistical-power revision 2026-05-02)
- **§7 mechanism**: 5-model perfect-square panel only (incl Main); InternVL3 + Qwen2.5-VL → appendix

## 1. Phase 1 task list (P0 only — execute in order)

### 1.1 Run baseline E5e/E7 on the 3 datasets needing fresh runs, 3-model panel

Configs (post 2026-05-02 restructure):

- `configs/experiment_e7_plotqa_full.yaml` — 3 models, `max_samples: 5000`. PlotQA snapshot is fetch-time stratified (5 gt bins × 1000) via `scripts/fetch_plotqa_test.py --max-samples 5000 --stratified`.
- `configs/experiment_e7_infographicvqa_full.yaml` — 3 models, `max_samples: 1147` = full numeric subset (data-bound, cap not binding).
- `configs/experiment_e5e_tallyqa_full.yaml` — 3 models, `max_samples: 5000` + `samples_per_answer: 700` for runtime stratification across gt 0-8. Existing 38k full-set runs archived to `outputs/_legacy_tallyqa_n38k/` (paper-architecture restructure: all main matrix at n=5000 for cross-dataset comparability).

ChartQA (`experiment_e5e_chartqa_full`) and MathVista (`experiment_e5e_mathvista_full`) reuse existing E5e runs — both data-bound below 5000 cap (~705 / ~385 numeric subset respectively).

Execute via the sequential GPU-1 driver (paper-run constraint: GPU 0 reserved for shared use):

```bash
# 3-model × 3-dataset (PlotQA + InfoVQA + TallyQA) sequential on GPU 1.
bash scripts/_phase1_baseline.sh > logs/phase1/baseline.log 2>&1 &
```

Expected wall on GPU 1 sequential (per-inference: llava ~120ms, qwen ~180ms, gemma3-27b ~550ms; 4 conditions per sid):
- PlotQA n=5000: 5000 × 4 × 0.85s ≈ 4h 40min
- InfoVQA n=1147: 1147 × 4 × 0.85s ≈ 1h 5min
- TallyQA n=5000: 5000 × 4 × 0.85s ≈ 4h 40min
- **Total ≈ 10h 30min sequential** (model loading overhead negligible at ~30s × 9 cells)

Output:
- `outputs/experiment_e7_plotqa_full/<model>/<ts>/predictions.jsonl`
- `outputs/experiment_e7_infographicvqa_full/<model>/<ts>/predictions.jsonl`
- `outputs/experiment_e5e_tallyqa_full/<model>/<ts>/predictions.jsonl` (fresh n=5000; legacy 38k at `outputs/_legacy_tallyqa_n38k/`)

### 1.2 E6 Subspace recalibration on PlotQA + InfoVQA pooled

After §1.1 baselines land, recompute the subspace.

Calibration data source (replace Tally-only):

- Wrong-base sids from `outputs/experiment_e7_plotqa_full/llava-next-interleaved-7b/<ts>/predictions.jsonl`
- Wrong-base sids from `outputs/experiment_e7_infographicvqa_full/llava-next-interleaved-7b/<ts>/predictions.jsonl`
- Pool both (target N ≈ 5000 = ~2300 PlotQA wrong-base + ~960 InfoVQA wrong-base; if shy of 5000, accept lower; if over, sample without replacement)

Calibration pipeline (mirroring `scripts/e6_master_v2.sh` S0–S2 but with new data):

```bash
# S0: calibrate-subspace (D matrix forward passes on wrong-base a/m pairs)
# Note: e6_steering_vector.py --phase calibrate-subspace expects
# --predictions-path; pass PlotQA preds first, then call again with InfoVQA
# preds, then pool D matrices via e6_compute_subspace.py --scope custom-tag
# OR: extend calibrate-subspace to accept multiple --predictions-path

# Tag for the new calibration:
TAG=plotqa_infovqa_pooled_n5k

# S0a — PlotQA D matrix
CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model llava-next-interleaved-7b \
    --hf-model llava-hf/llava-interleave-qwen-7b-hf \
    --predictions-path outputs/experiment_e7_plotqa_full/llava-next-interleaved-7b/<TS>/predictions.jsonl \
    --dataset-tag plotqa \
    --calibration-tag $TAG \
    --max-calibrate-pairs 2500 \
    --config configs/experiment_e7_plotqa_full.yaml

# S0b — InfoVQA D matrix
CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model llava-next-interleaved-7b \
    --hf-model llava-hf/llava-interleave-qwen-7b-hf \
    --predictions-path outputs/experiment_e7_infographicvqa_full/llava-next-interleaved-7b/<TS>/predictions.jsonl \
    --dataset-tag infographicvqa \
    --calibration-tag $TAG \
    --max-calibrate-pairs 1100 \
    --config configs/experiment_e7_infographicvqa_full.yaml

# S1 — pool + SVD per layer
uv run python scripts/e6_compute_subspace.py \
    --model llava-next-interleaved-7b \
    --scope $TAG \
    --tags plotqa,infographicvqa \
    --K-max 16

# Output: outputs/e6_steering/llava-next-interleaved-7b/_subspace/subspace_${TAG}_K16.pt
```

> ⚠️ Verify `e6_steering_vector.py --phase calibrate-subspace` accepts the new
> `--config` (E5e-style, b/a/m/d 4-cond) without crashing on missing fields.
> The pilot validation already exercised the predictions schema; calibration
> is a separate code path.

### 1.3 E6 sweep at single cell L31_K04_α=1.0 across 5 datasets, full gt range

**Sweep cap**: `--max-samples 5000` wrong-base sids per dataset (raised from 500 in 2026-05-02 statistical-power revision; smaller datasets cap naturally below 5000 by their eligible-4cond wrong-base count).

```bash
SUBSPACE_PT=outputs/e6_steering/llava-next-interleaved-7b/_subspace/subspace_${TAG}_K16.pt

# For each of the 5 datasets, sweep at L=31, K=4, α=1.0 single point
# (matches headline cell from old setup; plus baseline)
for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  PREDS=outputs/experiment_e7_${ds}_full/llava-next-interleaved-7b/<TS>/predictions.jsonl  # or e5e for non-Phase1
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/e6_steering_vector.py \
      --phase sweep-subspace \
      --model llava-next-interleaved-7b \
      --hf-model llava-hf/llava-interleave-qwen-7b-hf \
      --e5c-run-dir $(dirname $PREDS) \
      --predictions-path $PREDS \
      --dataset-tag $ds \
      --subspace-path $SUBSPACE_PT \
      --subspace-scope $TAG \
      --sweep-layers 31 --sweep-ks 4 --sweep-alphas 1.0 \
      --config configs/experiment_e7_${ds}_full.yaml  # use existing e5e configs for tally/chartqa/mathvista
done

# Analyze each
for ds in plotqa infographicvqa tallyqa chartqa mathvista; do
  uv run python scripts/analyze_e6_subspace.py \
      --sweep-dir outputs/e6_steering/llava-next-interleaved-7b/sweep_subspace_${ds}_${TAG}
done
```

Existing E5e configs to reuse for Sub datasets:
- `configs/experiment_e5e_tallyqa_full.yaml`
- `configs/experiment_e5e_chartqa_full.yaml`
- `configs/experiment_e5e_mathvista_full.yaml`

### 1.4 Reaggregate §3 main panel + §5 cross-dataset table for the 5-dataset matrix

Drop VQAv2 cells from headline tables. Keep VQAv2 in supplementary appendix.

```bash
# §5 E5e per-cell CSV refresh — extend to 5 datasets
uv run python scripts/analyze_e5e_cross_dataset.py \
    --models llava-next-interleaved-7b qwen2.5-vl-7b-instruct gemma3-27b-it \
    --datasets plotqa infographicvqa tallyqa chartqa mathvista
# (script may need flag plumbing if non-existent)

# §6 confidence reaggregation
uv run python scripts/analyze_confidence_anchoring.py \
    --predictions-roots <new e7 + existing e5e dirs> ...
```

## 2. Phase 1 success criteria

- ✅ 3-model × 5-dataset baseline matrix complete (15 cells)
- ✅ E6 PlotQA+InfoVQA-pooled-calibrated subspace computed and stored at `subspace_plotqa_infovqa_pooled_n5k_K16.pt`
- ✅ Sweep at L31_K04_α=1.0 on 5 datasets at full gt range — selection rule (Δdf ≤ −5% rel + Δem ≥ −2pp) holds on at least 4/5 datasets
- ✅ §3.3 / §5 / §6 paper-section _data CSVs reflect 5-dataset matrix
- ⚠️ If E6 fails on TallyQA full-range (small-gt confound): **plan B** = pool with TallyQA wrong-base too (becomes 3-source: count + chart + info). Document the iteration.

## 3. Phase 2 + Phase 3 (after Phase 1)

See `references/roadmap.md §7` Phase 2 (E5b/c digit-pixel breadth, ~20h) and
Phase 3 (E1-patch + Main, E4 + Main, ~10–14h). Decision: only kick off Phase 2/3
once Phase 1 succeeds and we have signal that the new architecture holds.

## 4. Decision log

- **2026-05-02 — Architecture decisions** (this restructure): see `references/roadmap.md §10` 2026-05-02 entry for full rationale.
- **2026-05-01 — Pilot validation**: pilots on llava (n=200) confirmed PlotQA + InfoVQA produce strong anchor pull (df 0.327 / 0.190); existing Tally-cal subspace transferred at gt∈[1,8].
- **2026-05-01 — Median distance metric**: switched from mean to median due to PlotQA outlier inflation (single 9,999,993 case made S3 mean 59,707 vs median 75).
