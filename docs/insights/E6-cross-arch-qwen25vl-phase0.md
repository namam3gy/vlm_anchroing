# E6 cross-arch on Qwen2.5-VL-7B-Instruct — Phase 0 evidence

**Date:** 2026-05-17. **Branch:** `worktree-e6-cross-arch-qwen25vl`.
Plan: [`docs/experiments/E6-cross-arch-design.md`](../experiments/E6-cross-arch-design.md).
Driver: [`scripts/run_e6_cross_arch_qwen25vl_phase0.sh`](../../scripts/run_e6_cross_arch_qwen25vl_phase0.sh).

## Phase 0 goal

Identify the L bin center for the 27-cell pilot grid on
Qwen2.5-VL-7B-Instruct by picking top-K layers ranked by
`‖v_wrong[L]‖` (per-layer (a − m) residual diff norm) on PlotQA +
InfoVQA pooled wrong-base + 4-cond eligible — exact OneVision §6.1
recipe.

## Inputs (existing predictions, no new inference)

| Source | Run dir | n_eligible_4cond | n_wrong-base |
|---|---|---:|---:|
| PlotQA | `outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/20260502-022631/` | 4,707 | 926 |
| InfoVQA | `outputs/experiment_e7_infographicvqa_full/qwen2.5-vl-7b-instruct/20260502-071849/` | 1,076 | 222 |
| **Pooled** | — | **5,783** | **1,148** |

**Pooled n_wrong (1,148) is below OneVision's 2,757** — Qwen2.5-VL is
more anchor-resistant than OneVision (outline §D.1 df 0.146 vs 0.178
on broad cohort, ~⅓ × wrong-base on the (a − m) eligibility filter).
Sample-size implication for Phase 2 statistical headlines noted in
plan §"Calibration pool".

## Recipe (parallel GPU 1 + GPU 2, ~30 min wall)

1. `e6_steering_vector.py --phase calibrate-subspace` on PlotQA → `D_wrong.pt (926, 28, 3584)` + `D_all.pt (4707, 28, 3584)` + per-source `v.pt`.
2. Same on InfoVQA (parallel GPU) → `D_wrong.pt (222, 28, 3584)` + `D_all.pt (1076, 28, 3584)`.
3. Pool per-source D matrices (concat + re-mean) → `calibration_plotqa_infovqa_pooled/v.pt` of shape `(2, 28, 3584)`.
4. `e6_compute_subspace.py --scope plotqa_infovqa_pooled --K-max 16` → `_subspace/subspace_plotqa_infovqa_pooled_K16.pt` + `singular_values_plotqa_infovqa_pooled.csv`.
5. `e6_pick_peak_layers.py --tag plotqa_infovqa_pooled --top-k 5` → `_subspace/peak_layers_plotqa_infovqa_pooled.json`.

## Result — per-layer ‖v_wrong[L]‖ profile (top 8)

| Layer L | ‖v_wrong[L]‖ | Rank |
|---:|---:|---:|
| **L26** | **8.5241** | 1 ★ |
| L25 | 7.0119 | 2 ★ |
| L24 | 6.1897 | 3 ★ |
| L27 | 5.8102 | 4 ★ |
| L23 | 5.2318 | 5 ★ |
| L22 | 4.5886 | 6 |
| L21 | 3.9074 | 7 |
| L20 | 3.4556 | 8 |

Full ramp: norm grows monotonically from L0 ≈ 0.1 up to L26 = 8.52,
sharp drop at L27 = 5.81. Late-residual integration site at L26
(depth-norm 93 %).

## Headline finding

**Qwen2.5-VL-7B-Instruct L\*_qwen = 26 — exactly matches OneVision's
L\* = 26.** Both models use the same Qwen2.5-7B-derived 28-layer LM
backbone; this Phase 0 result suggests **the integration site
depth is LM-backbone-determined and encoder-independent** — SigLIP
(OneVision) vs Qwen2-ViT-NaViT (Qwen2.5-VL) produce different
upstream visual representations, but the late-residual integration
site in the LM is the same.

This is a **secondary cross-arch finding** that informs outline §5.3
routing-and-integration framework — the integration site is determined
in the LM residual downstream of the encoder; encoder swap does not
shift the peak L. (Magnitude transfer + 4-clause free-lunch outcome
still TBD pending Phase 1-3.)

## 27-cell pilot L bin

`L ∈ {25, 26, 27}` centered on L\*_qwen = 26 — identical to
OneVision's pilot bin. No deviation from the exact OneVision §6.1
recipe.

## Wall-time observation (Phase 1 budget revision needed)

PlotQA calibrate-subspace took **1,797 s** (30 min) on H200 for 5,000
pairs of forward passes (model is 7B, sdpa attention, dynamic-resolution
image encoder). OneVision's equivalent was ~4 min per memory note —
**~7.5× slower** on Qwen2.5-VL.

Likely driver: Qwen2-ViT NaViT-style dynamic-resolution encoder
produces variable-length visual token sequences (longer than
SigLIP's fixed 729 tokens) → longer attention sequence → higher
forward cost.

**Phase 1 27-cell pilot revised budget**: at ~0.3–0.4 s per forward
pass on Qwen2.5-VL, sweep_n200 stratified × 4 conds × 27 cells ≈
21,600 forwards ≈ 1.8–2.5 hours per cell single-GPU. Sharded across
5 GPUs (resmgr has full 5-GPU host available, 2026-05-17 status) ≈
2 hours wall — *feasible* but Phase 1+2+3 total revised from ~7
H200-day to ~10–12 H200-day.

## Outputs persisted

```
outputs/e6_steering/qwen2.5-vl-7b-instruct/
├── _calibrate_plotqa.log
├── _calibrate_infovqa.log
├── calibration_plotqa/
│   ├── D_wrong.pt (926, 28, 3584)
│   ├── D_all.pt (4707, 28, 3584)
│   ├── v.pt (2, 28, 3584)
│   └── v_meta.json
├── calibration_infovqa/
│   ├── D_wrong.pt (222, 28, 3584)
│   ├── D_all.pt (1076, 28, 3584)
│   ├── v.pt (2, 28, 3584)
│   └── v_meta.json
├── calibration_plotqa_infovqa_pooled/      ← schema match OneVision
│   ├── v.pt (2, 28, 3584)
│   └── v_meta.json
└── _subspace/
    ├── subspace_plotqa_infovqa_pooled_K16.pt (28, 16, 3584)
    ├── singular_values_plotqa_infovqa_pooled.csv
    └── peak_layers_plotqa_infovqa_pooled.json  ← Phase 0 deliverable
```

## Next — Phase 1 27-cell pilot

L bin = {25, 26, 27} × K ∈ {2, 4, 8} × α ∈ {0.5, 1.0, 2.0} =
27 cells; calibration_tag = `plotqa_infovqa_pooled`; sweep on
PlotQA + InfoVQA n=200 stratified (within-distribution at pilot
stage). Cell selection by Δem(a) ≥ −6pp deal-breaker + combined
|Δdf(a)| ranking.

Driver TBD: `scripts/run_e6_cross_arch_qwen25vl_phase1.sh` (will
mirror OneVision's pilot grid driver with sharding across 5 GPUs).
