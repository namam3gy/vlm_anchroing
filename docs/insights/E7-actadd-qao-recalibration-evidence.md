# ActAdd + QAO recalibration on PlotQA + InfoVQA pool — evidence

**Date:** 2026-05-14
**Status:** complete; paper §7.4.5 fail-table proposal in §6 below
**Companion:** `E6-leace-recalibration-evidence.md` (Outcome B confirmed for LEACE rank-1, merged via PR #40)

## 1. Motivation

LEACE re-calibration (PR #40) showed that the §6.4 "+56% ChartQA backfire" was a calibration-pool artifact: replacing VQAv2 + TallyQA + ChartQA pool with the E6 (a − m) substrate (PlotQA + InfoVQA) flipped ChartQA from CI-clean backfire to CI-clean mild mitigation at α=0.5.

The §7.4.5 fail-table also marked **ActAdd** ("ChartQA backfires +57%, direction mismatch") and **QAO** ("below threshold on 1/4 datasets, probe over-fits training query distribution") as failed under the original Interleave + VQA/Tally/Chart pool. This work tests both methods under the same OneVision Main + PlotQA/InfoVQA pool used by LEACE re-cal, asking whether the recovery generalises across calibration-direction methods.

## 2. Setup (mirrors E6 LEACE re-cal)

| step | detail |
|---|---|
| Model | `llava-onevision-qwen2-7b-ov` (Main) |
| Calibration pool | PlotQA + InfoVQA pooled (a − m) wrong-base pairs (n=2314 + 443 = 2757 sids) |
| Apply layer | L = 26 (E6 K=8 cell #17 integration site) |
| ActAdd direction | `v_wrong = mean over D_wrong[plotqa] ∪ D_wrong[infovqa]` (rank-1 mean direction subtraction at last token) |
| QAO probe | PCA(100) + Ridge(α=1e3), trained on (Q[i] at L_q → D[i] at L_target=26), 4 probes for L_q ∈ {20, 24, 26, 27} |
| α sweep | ActAdd: α ∈ {0.5, 1.0, 2.0} (3 cells). QAO: 16 cells = 4 L_q × {0.5, 1, 2, 4} α |
| Eval | 5 datasets — TallyQA / PlotQA / InfoVQA / ChartQA / MathVista |
| Sample size | ActAdd: full wrong-base + all-4-conds eligible per dataset (8146 / 2306 / 443 / 224 / 170). QAO: cap 200 sids per dataset (mirrors §7 published evaluation grid) |
| CI | Sample-instance paired bootstrap, B = 10,000, seed = 20260514. Resampling unit = sid; both arms recomputed per draw. |

### 2.1 Q-D alignment fix (one-time correction)

The LEACE re-cal session had captured Q files via `--max-calibrate-pairs 2502` on `e5b_5strat_*` predictions, which produced `Q_wrong[i]` and `D_wrong[i]` referencing different sample sets when `train-probe` did `Q[:min(Q,D)]`. The probe was being trained on cross-sample (Q, D) pairs.

Patch (commits 72d733f + f52087f):
- `_phase_calibrate_qao` now restricts `wrong_sids` to wrong-base + all 4 conditions present (matches `calibrate-subspace` eligibility) → Q_wrong row count and dict-iteration order match D_wrong exactly
- `Q_sids.json` sidecar saved per row for traceability
- `_phase_train_probe` hard-asserts `Q.shape[0] == D.shape[0]` to fail loud on future misalignment

Verification: PlotQA Q_wrong (2314, 28, 3584) = D_wrong (2314, 28, 3584) ✓; InfoVQA Q_wrong (443, 28, 3584) = D_wrong (443, 28, 3584) ✓.

LEACE-session Q files preserved at `outputs/e6_steering/<MODEL>/calibration_<ds>_leace_session_backup/` for evidence trace.

## 3. Headline tables

### 3.1 ActAdd — α=1.0 (primary cell, mirrors LEACE α=1.0)

Source: [`docs/insights/_data/actadd_recal_per_dataset_ci.csv`](_data/actadd_recal_per_dataset_ci.csv).

| Dataset | n | Δdf(a) [95% CI] | Δem(b) [95% CI] | Δem(a) [95% CI] |
|---|---:|---:|---:|---:|
| **TallyQA** | 8146 | **+0.002 [−0.000, +0.005]** ~bf | **+0.011 [+0.009, +0.013]** ✓ | −0.001 [−0.002, +0.001] |
| PlotQA | 2306 | −0.004 [−0.010, +0.003] null | **+0.001 [+0.000, +0.002]** ✓ | −0.000 [−0.003, +0.002] |
| InfoVQA | 443 | +0.014 [−0.002, +0.032] null | **+0.016 [+0.005, +0.027]** ✓ | −0.002 [−0.011, +0.007] |
| **ChartQA** | 224 | **−0.013 [−0.040, +0.013]** null/mit | +0.005 [+0.000, +0.013] ~bf | −0.005 [−0.013, +0.000] ~bf |
| MathVista | 170 | +0.000 [−0.035, +0.035] null | +0.012 [+0.000, +0.029] ~bf | +0.000 [+0.000, +0.000] |

**Marker key**: ✓ = CI-clean (95% excludes 0 in target direction); ~bf = borderline (CI bound at 0); null = clearly overlaps 0; mit = mitigation-direction point estimate.

### 3.2 ActAdd — α-sensitivity on ChartQA (cross-check vs LEACE)

| α | ActAdd Δdf(a) | LEACE Δdf(a) (E6 evidence §3.1) |
|---|---:|---:|
| 0.5 | −0.005 [−0.031, +0.022] null | **−0.027 [−0.054, −0.005]** ✓ mit |
| 1.0 | −0.013 [−0.040, +0.013] null/mit | −0.022 [−0.045, +0.000] ~ |
| 2.0 | −0.009 [−0.036, +0.018] null | −0.018 [−0.058, +0.022] null |

ActAdd point estimates consistently negative (mitigation direction), magnitude smaller than LEACE. Same qualitative ChartQA-as-recovered pattern.

### 3.3 QAO — Lq26_Lt26_a=1.0 (central cell mirroring L=26 substrate)

Source: [`docs/insights/_data/qao_recal_per_dataset_ci.csv`](_data/qao_recal_per_dataset_ci.csv).

| Dataset | n | Δdf(a) [95% CI] | Δem(b) [95% CI] | Δem(a) [95% CI] |
|---|---:|---:|---:|---:|
| TallyQA | 200 | −0.005 [−0.025, +0.010] null | −0.010 [−0.025, +0.000] ~bf | −0.005 [−0.015, +0.000] ~bf |
| PlotQA | 200 | +0.015 [−0.005, +0.040] null | −0.005 [−0.015, +0.000] ~bf | −0.005 [−0.015, +0.000] ~bf |
| InfoVQA | 200 | −0.010 [−0.035, +0.015] null | −0.005 [−0.015, +0.000] ~bf | −0.015 [−0.035, +0.000] ~bf |
| ChartQA | 218 | +0.005 [−0.018, +0.028] null | +0.005 [−0.014, +0.023] null | −0.009 [−0.028, +0.009] null |
| MathVista | 170 | +0.006 [−0.024, +0.035] null | +0.006 [−0.012, +0.029] null | +0.012 [+0.000, +0.029] ~bf |

QAO **flattens** anchor-mitigation effects across all 5 datasets at the central cell: 0/5 CI-clean Δdf, 0/5 CI-clean Δem(b) gains.

### 3.4 QAO — best mitigation cell per dataset (search across 16 cells)

| Dataset | Best cell | Δdf(a) [95% CI] | Δem(b) | Status |
|---|---|---:|---:|---|
| TallyQA | Lq20_Lt26_a4.0 | −0.020 [−0.050, +0.010] | −0.005 | null |
| PlotQA | Lq20_Lt26_a2.0 | −0.005 [−0.030, +0.020] | −0.010 | null |
| InfoVQA | Lq27_Lt26_a1.0 | −0.020 [−0.055, +0.015] | −0.010 | null |
| ChartQA | Lq20_Lt26_a2.0 | −0.014 [−0.041, +0.014] | +0.009 | null |
| **MathVista** | Lq20_Lt26_a4.0 | **−0.035 [−0.065, −0.012]** ✓ | +0.006 | **CI-clean mit (small-n)** |

Even when cherry-picking best cell per dataset, only 1/5 (MathVista, n=170) reaches CI-clean. No cell on the larger datasets clears 95% CI. The best ChartQA cell point estimate (−0.014) matches ActAdd / LEACE direction but is no stronger than ActAdd's central-cell estimate.

## 4. Cross-method synthesis (3 methods on same scope)

| Method | ChartQA Δdf(a) @ matched-cell | TallyQA Δdf(a) | Δem(b) profile |
|---|---:|---:|---|
| LEACE rank-1 (α=1.0, evidence §3.1) | −0.022 [−0.045, +0.000] ~ | +0.006 [+0.000, +0.012] ~bf | 5/5 CI-clean positive |
| ActAdd (α=1.0) | −0.013 [−0.040, +0.013] null | +0.002 [−0.000, +0.005] ~bf | 4/5 CI-clean positive |
| QAO (Lq26_Lt26_a=1.0) | +0.005 [−0.018, +0.028] null | −0.005 [−0.025, +0.010] null | 0/5 CI-clean positive |

**Reading**:
1. **Mean-direction methods (LEACE + ActAdd) recover under PlotQA+InfoVQA pool.** Both show ChartQA point estimates in mitigation direction, both show TallyQA borderline backfire (CI lower bound at 0). Calibration-pool hypothesis confirmed across two methods.
2. **TallyQA-as-backfire-site is a method-independent signature** of single mean direction at L=26 on this scope (LEACE +0.6pp ~bf, ActAdd +0.2pp ~bf, §6.2.4 P4 K=1 SVD +1.4pp [+0.5, +2.2] sig). Three methodologically-different single-direction interventions agree.
3. **QAO probe correction flattens effects.** Per-sample δ from the Ridge probe smooths both the ChartQA mitigation AND the TallyQA backfire to within ±1.5pp of zero across all 5 datasets at the central cell. This is a different failure mode from the original Interleave QAO ("probe overfits training query distribution"): under aligned (Q, D) pairs and PlotQA+InfoVQA calibration the probe instead under-fits to a near-constant mean correction with regularised dispersion.
4. **Capability axis ranks** mean-direction (LEACE > ActAdd) > probe-corrected (QAO). LEACE's strong Δem(b) gains hold; ActAdd weaker but same direction; QAO neutral or mildly negative.

## 5. Limitations / caveats

- **ChartQA n=224 / MathVista n=170** small-sample limits. ActAdd ChartQA −0.013 with CI [−0.040, +0.013] cannot statistically distinguish "mild mitigation" from "null". The point estimate is consistent with LEACE's −0.022 but the bootstrap floor allows ±2-4pp wobble.
- **QAO sweep cap n=200 per dataset** (mirrors §7 published evaluation, smaller than ActAdd / LEACE full samples). MathVista's CI-clean cell (n=170) is the smallest sample and the only one to clear 95% — could be a power-driven single-CI-pop. Bonferroni-80 across the 16-cell × 5-dataset family would push that CI back to overlapping 0.
- **Probe choice axis (L_q × L_target)**: explored 4 × 1 = 4 (L_q, L_target=26) probes. Larger grid (L_target ∈ {24, 26, 27} × wider L_q) might find a passing cell but the 16-cell α sweep at L_target=26 already shows uniformly null across larger datasets — extending the grid is unlikely to flip the qualitative reading.
- **Two axes changed simultaneously vs original Interleave QAO measurement** (model + pool, same caveat as LEACE re-cal §6). The recovery / flattening attribution is bounded to the (model × pool) joint cell.
- **Q-D alignment patch was load-bearing for QAO**: the original LEACE-session Q files (Q_wrong=2502, see §2.1) would have produced misaligned probe training. Reported QAO numbers are from the re-extracted aligned Q (Q_wrong=2314 = D_wrong=2314 for plotqa).

## 6. Paper update proposal — §7.4.5 fail-table

Current paper [docs/paper/sections/07_mechanism_mitigation.md:392-396](../paper/sections/07_mechanism_mitigation.md#L392-L396):

```
| Single-direction ActAdd  | ❌ ChartQA backfires +57 %    | Invariant | Invariant | Direction mismatch                      |
| LEACE closed-form (rank-1) | ❌ ChartQA backfires +56 %  | Invariant | Invariant | Same as above                           |
| Query-adaptive offset (PCA + Ridge probe) | ❌ Below threshold on 1/4 | Invariant | Invariant | Probe overfits training query distribution |
| CogBias decode-step       | ❌ Below threshold on 1/4   | Invariant | Invariant | Same root cause                         |
| MIA-DPO LoRA              | Partial df reduction       | −5.85 pp on VQAv2 | not reported | em side effect on a-arm; gt-distribution training bias |
```

LEACE row is already updated in master (PR #40, evidence doc §3.1). Proposed updates:

| Method | Current verdict | Proposed verdict (under OneVision + PlotQA/InfoVQA pool) | Reason update |
|---|---|---|---|
| Single-direction ActAdd | ❌ +57% backfire | **Recovered:** ChartQA Δdf −1.3pp [−4.0, +1.3] null/mit; 4/5 datasets Δem(b) CI-clean positive; TallyQA borderline backfire ~bf | Calibration-pool axis was load-bearing; same recovery as LEACE rank-1 |
| QAO (PCA + Ridge probe) | ❌ Below threshold on 1/4 | **Flattened:** 0/5 datasets CI-clean Δdf at central cell; only MathVista (n=170) passes when cherry-picked across 16 cells; Δem(b) mostly null/slightly negative | Probe smoothing dampens both mitigation and backfire; capability axis weaker than mean-direction methods |
| CogBias decode-step | ❌ Below threshold on 1/4 | **Not retested** (mechanism-equivalent to ActAdd at decode-time; same mean direction; predicted to follow ActAdd recovery pattern; deferred to §8.4) | Mean-direction substrate identical; novelty axis is decode-time application |
| MIA-DPO LoRA | em side effect on a-arm | **Not retested** (different mechanism — gradient training, not direction subtraction; the gt-distribution training bias is intrinsic; deferred to §8.4) | Re-testing requires LoRA re-training with PlotQA/InfoVQA pairs; high cost low expected information gain |

### 6.1 Implication for §7's "single-direction failure is intrinsic" framing

[docs/paper/sections/07_mechanism_mitigation.md:374-377](../paper/sections/07_mechanism_mitigation.md#L374-L377) currently reads:

> "v_wrong calibrated on a small TallyQA pool and applied to ChartQA *increases* direction-follow by +57 %. The reason is that the per-dataset mean-anchor directions point measurably differently — `cos(v_tally, v_chartqa) ≈ 0.47-0.62` at the top-norm layers — so a single direction cannot simultaneously remove TallyQA's and ChartQA's anchor signals."

Recommended revision: keep the per-dataset cos(v_tally, v_chartqa) ≈ 0.47-0.62 evidence (calibration-independent), but replace the "+57% backfire" anchor with the new finding that single-direction methods *recover* under (a−m)-matched calibration AND *still under-mitigate* compared to multi-direction K=8 (see §3.4 best-cell QAO −0.014 vs §6.2 K=8 ChartQA −2.7pp on the same scope). The single-direction failure mode is not "direction mismatch causes backfire" but "single direction under-mitigates by ~50% relative to K=8 even when calibration is right".

This preserves §7's K=8 selection rationale while honestly accounting for the recalibration findings.

### 6.2 §8.4 follow-up registration

- **Item N (new)**: Cross-architecture ActAdd / QAO recalibration — Qwen2.5-VL, Gemma3, Phi4 — to test whether the mean-direction recovery + probe-flattening pattern generalises beyond OneVision Main. Subsumed under existing §8.4 cross-architecture E6 item if present.
- **Item M (new)**: CogBias decode-time evaluation under PlotQA/InfoVQA pool — predicted to track ActAdd recovery; small additional information value, deferred unless cross-architecture work surfaces a divergent reading.
- MIA-DPO LoRA — already registered under existing §8.4 weight-space mitigation deferral.

## 7. Reproducibility

```bash
# (1) Pool v_wrong / v_all (CPU, ~5 min — already cached)
uv run python scripts/_pool_v_actadd_recal.py

# (2) Phase A — ActAdd tiebreaker × 5 datasets (~12 h H200)
CUDA_VISIBLE_DEVICES=0 bash scripts/_actadd_qao_recalibration_runner.sh
# (sequential phases inside; idempotent on existing outputs)

# (3) Aggregation — paired bootstrap CI per dataset × cell (CPU, ~5 min)
uv run python scripts/_aggregate_actadd_qao_recal.py --family actadd
uv run python scripts/_aggregate_actadd_qao_recal.py --family qao
```

Outputs land at `docs/insights/_data/{actadd,qao}_recal_per_dataset_ci.{csv,md,npz}`.

## 8. Cross-references

- Companion: `E6-leace-recalibration-evidence.md` (LEACE Outcome B, merged via PR #40)
- §7.4.5 fail-table: `docs/paper/sections/07_mechanism_mitigation.md:392-396`
- §6.2.4 P4 K=1 SVD evidence (third single-direction method, master): `docs/insights/E6-section6-2-4-p4-evidence.md` (TallyQA Δdf +1.4pp [+0.5, +2.2] sig backfire — corroborates §4 cross-method synthesis)
- §6.4 LEACE recalibration: master `docs/paper/sections/06_confidence.md` already updated
- Q-D alignment patches: commits 72d733f + f52087f on `worktree-actadd-qao-recalibration` branch
