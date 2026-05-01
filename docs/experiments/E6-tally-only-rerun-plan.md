# E6 Tally-only rerun — formal plan

**Created: 2026-04-30**
**Branch: `e6-tally-only-rerun`** (cut from `e6-methods-4c-4a-3` at commit `c08a271`)
**Model: llava-next-interleaved-7b**
**GPU: 1 (sequential)**

## 1. Motivation

After the multi-method search (Methods 0a/0b/1/2/4c/4a/3) declared all methods
❌ FAILED under the original two-sided em rule, three observations triggered
this rerun:

1. **LEACE 4c re-analysis under one-sided em rule** (`em_pp ≥ baseline − 2pp`,
   allow em gains as intended mitigation effect): cell L30_a2.0 passes both
   TallyQA n=100 (df −13.2 %, em **+5.88 pp**) and ChartQA n=99 (df **−38.1 %**,
   em invariant). **First non-noise cross-dataset overlap = 1 cell** in the
   entire multi-method search. Documented in `E6-steering-vector.md` Method 4c
   section.
2. **CogBias selection bias diagnostic** showed n=100 baselines were inflated
   (Tally 14.0 % apparent vs 12.85 % true at n=500). LEACE n=100 likely has
   the same issue; full-set re-validation needed.
3. **Calibration-source asymmetry**: `cos(v_tally, v_chartqa) = 0.47–0.62`
   while `cos(v_VQA, v_tally) = 0.98`. Pooled mean v_general (over 3 datasets)
   averaged a 1/3-outlier ChartQA direction into the calibration, blunting it
   on the two aligned datasets. Tally-only calibration may be sharper.
4. **Layer choice**: prior sweep grid `L ∈ {16, 22, 28, 30, 31}` was set from
   Phase 0 VQAv2-cal `‖v[L]‖` peak at L=30. Tally N=5000 calibration may
   peak elsewhere — choose layers from data, not hardcoded.

## 2. Hypothesis

H_rerun: Under (a) Tally-only calibration with N=5000 ≫ 1000, (b) one-sided
em rule, (c) data-driven peak-layer selection, **at least one of LEACE / Subspace
/ DPO** retains a cross-dataset passing cell at sweep n=500 (Tally + ChartQA).

Falsifier: All three methods fail cross-dataset selection rule (≥ 5 % rel df
reduction, em ≥ baseline − 2 pp on ≥ 2 datasets). Implication: §7.4.5
permanent fallback to "cross-dataset failure as the empirical contribution".

## 3. Pipeline

### Stage S0 — calibration extraction (~50 min, GPU 1)

Re-extract D_wrong + Q_wrong from TallyQA E5e at N=5000:
- N grows from 346 (E5c, capped at 1000 sids) → 5000 (from 11,209 available
  wrong-base in E5e Tally), comfortably above d_model = 4096 SNR floor.
- Saves to `outputs/e6_steering/<model>/calibration_tally_e5e_n5k/`.

| Substage | Script | Phase | Output | ETA |
|---|---|---|---|---|
| S0a | `e6_steering_vector.py` | calibrate-subspace | D_wrong.pt, v.pt, norms_per_layer.csv | ~30 min |
| S0b | `e6_query_adaptive_offset.py` | calibrate-qao | Q_wrong.pt, Q_all.pt | ~25 min |

### Stage S0.5 — peak-layer selection (~1 min, CPU)

Read `norms_per_layer.csv` from new Tally calibration. Pick top-K layers by
`‖v_wrong[L]‖` for downstream sweeps.

- Top 5 layers → LEACE (5 × 4 alpha = 20 cells)
- Top 5 layers → Subspace (5 × 4 K × 4 alpha = 80 cells)

Output: `outputs/e6_steering/<model>/_subspace/peak_layers_tally_e5e_n5k.json`

### Stage S1 — LEACE Method 4c Tally-only (~4 h, GPU 1)

| Substage | Description | ETA |
|---|---|---:|
| S1a | calibrate-leace `--calib-tags tally_e5e_n5k --eraser-tag tally_e5e_n5k` | ~3 min CPU |
| S1b | sweep TallyQA n=500 with auto-selected layers | ~95 min |
| S1c | sweep ChartQA n=500 with auto-selected layers | ~120 min |
| S1d | analyze both with `--em-rule one_sided` and `two_sided` | <1 min |

### Stage S2 — Subspace Method 1 Tally-only (~12 h, GPU 1)

| Substage | Description | ETA |
|---|---|---:|
| S2a | compute SVD `--scope tally_e5e_n5k --tags tally_e5e_n5k` | ~5 min CPU |
| S2b | sweep TallyQA n=500 (80 cells) | ~6 h |
| S2c | sweep ChartQA n=500 (80 cells) | ~5 h |
| S2d | analyze with `analyze_e6_subspace.py` (will need one-sided support added) | <1 min |

### Stage S3 — DPO Method 3 with case-by-case rejected (~3 h, GPU 1)

Pre-work: modify `e6_dpo_lora.py:_phase_build_pairs`:
- Skip if `pred == gt` (already correct, no signal)
- `rejected = anchor` if `pred == anchor` (current behaviour for these cases)
- `rejected = pred_a` if `pred ≠ anchor AND df_moved=True`
  (df_moved: `(pa − pb)·(anchor − pb) > 0 AND pa ≠ pb`)
- Skip otherwise (low-signal random-wrong cases)

Tally-only training data (or oversample ChartQA to 50/50) — TBD after build-pairs
inspection.

| Substage | Description | ETA |
|---|---|---:|
| S3a | re-build pairs with new rejected logic | ~5 min |
| S3b | DPO LoRA training (~5000 pairs, rank=256, α=256) | ~90 min |
| S3c | sweep TallyQA + ChartQA at adapter | ~60 min |
| S3d | analyze (one-sided em rule) | <1 min |

## 4. Selection rule (universal)

```
df_rel_change ≤ −5 %                          # at least 5 % rel df reduction
AND em_pp_change ≥ −2 pp                       # one-sided em rule (em gains OK)
AND apply on ≥ 2 of {Tally, ChartQA}           # cross-dataset robustness
```

VQAv2 gated — only run if cross-dataset passes (avoids gold standard before proof).

## 5. Total budget

| Stage | ETA |
|---|---:|
| S0 calibration | 50 min |
| S0.5 layer pick | 1 min |
| S1 LEACE | 4 h |
| S2 Subspace | 12 h |
| S3 DPO | 3 h |
| **Total** | **~20 h overnight** |

## 6. Documentation + git plan

- **Branch**: `e6-tally-only-rerun` (cut from `e6-methods-4c-4a-3`)
- **Commits**:
  1. ✓ This plan + tracker doc + script CLI flags (`--em-rule`, `--eraser-tag`,
     `--layers`, `--alphas`, `--max-calibrate-pairs`)
  2. After S0 done: peak-layer selection results + roadmap §10 update
  3. After S1 done: LEACE Tally-only result tables in `E6-steering-vector.md`
  4. After S2 done: Subspace Tally-only result tables
  5. After S3 done: DPO v2 result tables
  6. Final synthesis: cross-method comparison + verdict (✅ pass any /
     ❌ all fail) → roadmap §7.4.5 framing

## 7. Pre-existing artifacts preserved

- `calibration_tally_e5c_n346/` ← old E5c-based 346-sample tally calibration
- `leace_erasers_pooled_n1145/` ← old pooled erasers
- `sweep_leace_{tally,chartqa}_pooled/` ← old n=100 LEACE sweep results
- `sweep_subspace_tally_n500_pooled/` ← Method 1 n=500 Tally selection-bias check (completed earlier today)
- `sweep_subspace_*_pooled/` ← Method 1 n=100 results
- `sweep_cogbias_*_{pooled,fullset_pooled}/` ← Method 4a n=100 + full-set
- `sweep_dpo_*_pooled/` ← Method 3 v1 (97 % Tally training data, anchor-only rejected)

All preserved untouched; new artifacts go to `*_tally_e5e_n5k_pooled/` paths.

## 8. Status

See `E6-tally-only-rerun-tracker.md` for live status.

## 9. Outcomes catalog

| Method | Outcome | Cross-dataset overlap | em rule | Source |
|---|---|---:|---|---|
| 4c LEACE pooled n=1145 | ⚠ tentative ✅ | 1 (L30_a2.0) | one-sided | n=100 sweep |
| 4c LEACE Tally-only n=5k | ⏳ in flight | TBD | one-sided | this rerun |
| 1 Subspace pooled n=1145 | ❌ failed | 0 | two-sided | n=100 + n=500 Tally |
| 1 Subspace Tally-only n=5k | ⏳ queued | TBD | one-sided | this rerun |
| 4a CogBias pooled | ❌ full-set ChartQA −4.3 % below threshold | 0 | both rules | full-set |
| 3 DPO v1 (97 % Tally) | ❌ ChartQA parse failures, em −3.7 pp | 0 | one-sided + invariance | full sweep |
| 3 DPO v2 (case-by-case rejected) | ⏳ queued | TBD | one-sided | this rerun |
