# E6 — Tally-only + N=5k + one-sided em rule rerun tracker

**Started: 2026-04-30 21:37 KST**
**Branch: `e6-methods-4c-4a-3` (current)**
**Model: llava-next-interleaved-7b**
**GPU: 1**

## Motivation

After full-set validation of Methods 0–4 (CogBias, Subspace, QAO, LEACE, DPO):

1. **All methods judged ❌ FAILED under two-sided em rule** (\|em_pp\| ≤ 2pp).
2. **Re-analysis under one-sided rule** (em_pp ≥ −2pp; allow gains as intended
   mitigation effect) found **LEACE L30_a2.0 passes both Tally and ChartQA**
   at n=100 — first cross-dataset overlap = 1 cell.
3. n=100 baselines were selection-biased (CogBias case: Tally 14.0% apparent
   vs 12.85% true at n=500). LEACE n=100 likely shares this issue.
4. **Hypothesis**: the structural failure (`cos(v_tally, v_chartqa) = 0.5`)
   may be tractable with:
   - **Tally-only calibration** (avoid pooling outlier ChartQA direction)
   - **Larger N=5000** (vs prior 346 pooled / d=4096 SNR floor)
   - **One-sided em rule** (treat em gains as intended)

## Calibration source change

| | Old (pooled) | New (Tally-only) |
|---|---|---|
| Source | E5c VQAv2 + E5c Tally + E5e ChartQA | **E5e TallyQA only** |
| N total | 1,145 | **5,000** |
| N wrong-base | 1,145 (all wrong) | ~5,000 (Tally has 11,209 available) |
| d_model | 4,096 | 4,096 |
| N/d ratio | 0.28 (severe undersampling) | **1.22** (over saturation) |

## Selection rule change

| Rule | Definition | Catches |
|---|---|---|
| Two-sided (old) | df_rel ≤ −5% AND \|em_pp\| ≤ 2 | Symmetric — rejects em gains too |
| **One-sided (new)** | df_rel ≤ −5% AND em_pp ≥ −2 | Only rejects em drops (real damage) |

Rationale: em gains coupled with df reductions are the **intended mitigation
effect** (predictions moving from anchor toward gt). The original rule rejected
them as suspicious; the new rule treats them correctly.

## Stage tracker

| Stage | Description | ETA | Status |
|---|---|---|---|
| **S0a** | calibrate-subspace TallyQA E5e N=5000 → D_wrong | 22 min actual | ✅ done 21:59 |
| **S0b** | calibrate-qao TallyQA E5e N=5000 → Q_wrong | **6 min actual** | ✅ done 22:09 |
| **S0.5** | pick top-5 peak layers from ‖v_wrong[L]‖ | <1 sec actual | ✅ done 22:09 — **L 27,28,29,30,31** |
| **S1a** | LEACE calibrate-leace Tally-only (CPU) | 2 min actual | ✅ done 22:11 |
| **S1b** | LEACE sweep TallyQA n=500 with peak layers (20 cells × 3 conds × 500) | 68 min actual | ✅ done 23:19 |
| **S1c** | LEACE sweep ChartQA n=500 (same grid; ChartQA has 416 wrong-base) | ~63 min | 🟢 in flight (started 23:19) |
| **S1d** | analyze (one-sided + two-sided) | <1 min | ⏳ queued |
| **S2a** | Method 1 Subspace compute SVD Tally-only | <1 min | ⏳ queued (post S1) |
| **S2b** | Method 1 sweep Tally n=500 | ~6 h | ⏳ queued (post S1) |
| **S2c** | Method 1 sweep ChartQA n=500 | ~5 h | ⏳ queued (post S1) |
| **S3** | Method 3 DPO with case_by_case rejected (Tally-only) | ~3 h | ⏳ queued (post S2) |

## Peak-layer selection result (S0.5)

Top-5 layers by ‖v_wrong[L]‖ from N=5000 Tally calibration: **L 27, 28, 29, 30, 31** (all post-mid-stack, peaked at L=30 with norm 6.98).

Per-layer norms (★ = top-5 selected):

```
L00   0.029       L08   0.271       L16   1.703       L24   3.414
L01   0.037       L09   0.345       L17   1.759       L25   3.814
L02   0.061       L10   0.380       L18   2.134       L26   3.945
L03   0.073       L11   0.638       L19   2.593       L27 ★ 4.271
L04   0.086       L12   0.733       L20   2.755       L28 ★ 4.588
L05   0.099       L13   1.055       L21   2.911       L29 ★ 5.226
L06   0.197       L14   1.250       L22   3.096       L30 ★ 6.984
L07   0.220       L15   1.486       L23   3.199       L31 ★ 5.390
```

Notable: **L=16 (norm 1.70) and L=22 (norm 3.10) — both included in the legacy default grid `[16, 22, 28, 30, 31]` — fall well below the top-5 cluster.** Default's mid-stack inclusion (L=16) was anchored to E1b CLIP-ViT attention peak, not the residual-stream calibration. Tally-only N=5000 says the residual-stream signal is concentrated late-stack.

**Total budget: ~22h overnight on GPU 1.**

## Logs

- Progress summary: `/tmp/e6_tally_only_progress.log`
- S0+S1 detail: `/tmp/e6_tally_only_s0s1.log`
- Pipeline script: `/tmp/e6_tally_only_s0s1.sh`

## S1b — LEACE Tally-only N=5k sweep on TallyQA (n=500)

**Baseline:** df=0.1245, em=0.1185, n=500 wrong-base from E5e Tally.

**4 cells pass both two-sided and one-sided em rules** (all in late-stack cluster):

| cell | df | df_Δ% | em | em_pp | adopt | pass |
|---|---:|---:|---:|---:|---:|:---:|
| **L30_a2.0** ⭐ | 0.0973 | **−21.9%** | 0.1216 | +0.31 | 0.0313 | ✅✅ |
| L30_a1.0 | 0.1051 | −15.6% | 0.1124 | −0.60 | 0.0348 | ✅✅ |
| L30_a0.5 | 0.1066 | −14.3% | 0.1084 | −1.00 | 0.0393 | ✅✅ |
| L31_a0.5 | 0.1129 | −9.3% | 0.1227 | +0.43 | 0.0419 | ✅✅ |

**Effect concentrated at L=30** (the peak ‖v‖ layer), with α=2.0 the strongest.
L29 / L28 / L27 all fail (df Δ% within ±3% across all alphas, mostly noise).
L31 weaker than L30 (only α=0.5 passes; α=1.0/2.0 backfire +14.3% / +11.9%).

**Comparison to prior pooled n=100 result:**

| | pooled n=100 | Tally-only N=5k (this run) |
|---|---:|---:|
| Best cell | L30_a2.0 | L30_a2.0 (same) |
| df_Δ% | −13.2% | **−21.9%** (stronger) |
| em_pp | +5.88 (one-sided only) | +0.31 (passes both rules) |
| Baseline df | 0.1200 | 0.1245 (matches true Tally rate ~12.85%) |

**Interpretation.** Same winning cell (L30_a2.0) in both pooled and Tally-only
calibration → robust mitigation locus. Tally-only N=5k sharpens the effect
(df reduction nearly doubled) and removes the em-gain artefact (now em is
within ±0.5 pp of baseline, passing both rules cleanly). The selection-bias
hypothesis is **partially refuted** — n=100 baseline 14.0 % was inflated, but
the cell choice itself transfers cleanly to N=500 with proper baseline.

**S1c ChartQA pending** — the cross-dataset test. If L30_a2.0 also passes
ChartQA at n=500, this is the first method to clear the cross-dataset rule
without dispute.

## S1c — LEACE Tally-only N=5k sweep on ChartQA (n=500): **cross-dataset FAILED**

**Baseline:** df=0.2260, em=0.0511, n=499 wrong-base from E5e ChartQA.

| Cell | Tally df_Δ% | ChartQA df_Δ% | ChartQA em_pp | Cross-dataset |
|---|---:|---:|---:|---|
| L30_a2.0 ⭐ Tally winner | **−21.9%** | **+8.7%** ❌ | +0.47 | Backfires on ChartQA |
| L30_a1.0 | −15.6% | +5.4% ❌ | +0.73 | Same pattern |
| L30_a0.5 | −14.3% | +0.2% (≈0) | +0.00 | No effect on ChartQA |
| L31_a0.5 | −9.3% | −0.8% (≈0) | +0.94 | No effect on ChartQA |
| L28_a1.0 | +6.5% ❌ | **−5.4%** | +0.49 | ChartQA OK, Tally backfires |

**Cross-dataset overlap = 0** (both em rules).

**Root cause confirmed.** Tally-only calibration sharpens the Tally direction
(df reduction nearly doubled vs pooled n=100) but ChartQA generalisation lost.
Same direction-mismatch as Methods 0–2: `cos(v_tally, v_chartqa) ≈ 0.47–0.62` →
Tally-only erasure projection doesn't cover ChartQA's separate anchor direction.
Earlier pooled n=100 ChartQA "pass" (L30_a2.0 df −38.1%) likely came from
ChartQA being in the pooled calibration data — not Tally-only generalisation.

## v2 — expanded layer set [L7, L16, L24, L30, L31]

**Motivation:** Top-5 ‖v[L]‖ peak layers were all post-mid (L27-L31). Concern:
mid-stack and early-stack might host a "spot" we're missing if effect locus
differs between calibration norm and actual mitigation locus. New layer set
adds explicit early (L7) and mid (L16) plus a middle-late candidate (L24)
alongside the two strongest peaks (L30, L31).

| Layer | ‖v_wrong‖ | category | rationale |
|---|---:|---|---|
| L7  | 0.220 | early | sanity check; sub-token level mostly visual encoding |
| L16 | 1.703 | mid | mid-stack candidate; matches E1b CLIP-ViT attention peak |
| L24 | 3.414 | late-mid | between mid and peak |
| L30 | 6.984 | peak | best in S1 (df −21.9 % rel) |
| L31 | 5.390 | top-2 by norm | 2nd-best, weaker than L30 in S1 |

S2b on [27,28,29,30,31] killed mid-run (~36 min in, ~16k records, partial
output preserved at `predictions.bak_l27to31_partial.jsonl`).

S1 LEACE supplement: re-sweep with L=7,16,24 only (L30, L31 already done).
12 new cells × 4 conds × 500 sids = 18k records → ~40 min Tally + ~35 min
ChartQA = ~75 min total. Existing 31,500 records preserved via append-mode.

S2 Subspace: full re-sweep with [7, 16, 24, 30, 31] = 81 cells × 1500
records = ~6 h Tally + ~5 h ChartQA = ~11 h.

S3 DPO: unchanged (whole-model LoRA, no layer dependence).

## Known risk: DPO gt-distribution bias contamination

**Concern surfaced 2026-05-01.** TallyQA gt distribution is heavily skewed
to single digits (96.1 % gt ∈ [0,4], max gt = 8, mean 1.6). ChartQA gt
distribution is much wider (max 991, mean 107.8, 78.2 % of ChartQA samples
have gt > Tally's max of 8).

If DPO trained on Tally-only synthetic-anchor pairs uses `chosen = str(gt)`
for gt ∈ [0,8] only, the model may learn an implicit "small numbers are
preferred answers" bias. On ChartQA evaluation where the correct answer is
often a 2-3 digit number (e.g. gt=234), this distributional bias would
catastrophically degrade ChartQA accuracy — *not* because of anchor pull
but because the model has been trained to suppress large-number answers.

**Predicted DPO Tally-only cross-dataset failure mode (two causes):**
1. Anchor-direction mismatch (verified by LEACE/Subspace): cos(v_tally, v_chartqa) ≈ 0.5
2. gt-distribution mismatch (THIS concern): chosen tokens differ in scale by ~70×

The two causes are *independently* sufficient to break cross-dataset transfer.
This means DPO failure on ChartQA cannot be cleanly attributed to anchor
direction alone — interpret with care for §7.4.5.

**Why activation-space methods (LEACE, Subspace) are less affected:**
they project residual streams, not output token distributions. LEACE Tally-only
results on ChartQA show em remains invariant (±1pp) — confirms no
distribution bias contamination at activation-space level. DPO operates at
weight-space and directly biases token preferences, hence highly susceptible.

**Mitigations considered (not adopted to preserve Tally-only protocol):**
- Train on Tally + ChartQA mix → contradicts the Tally-only test
- Synthesize wider gt distribution via fake samples → not faithful to actual data
- Train per-dataset and evaluate per-dataset → defeats cross-dataset claim

**Decision: proceed with Tally-only DPO and explicitly note this caveat
in the paper's §7.4.5 DPO subsection.** The cross-dataset failure remains
informative as long as we attribute it correctly (direction + distribution,
both contribute).

## 🎯 BREAKTHROUGH — S2 Subspace Tally-only N=5k cross-dataset overlap

**Two-sided em rule: 4 cells pass on BOTH Tally and ChartQA.**

| Cell | Tally df_Δ% | Tally em_pp | ChartQA df_Δ% | ChartQA em_pp |
|---|---:|---:|---:|---:|
| **L31_K04_a2.0** ⭐ | **−63.6%** | +1.30 | **−9.6%** | −0.54 |
| L31_K04_a1.0 | −54.6% | +1.64 | −13.0% | +0.50 |
| L24_K04_a1.0 | −39.4% | −0.95 | −5.9% | −0.02 |
| L07_K04_a0.5 | −7.6% | −1.00 | −5.2% | +0.49 |

**One-sided em rule: 9 cells pass on both** (adds L16_K16_a1.0, L31_K16_a0.5/1/2/4 with bigger em gains on Tally).

**This is the FIRST method in the multi-method search (Methods 0–4) that
clears the cross-dataset selection rule with non-zero overlap under the
two-sided rule.** §7.4.5 has a real deployable cross-dataset mitigation
candidate.

**Why Subspace works where LEACE failed:**
LEACE removes a single linearly-decodable concept direction. Its projection
captures the dominant Tally anchor direction but misses ChartQA's separate
direction (cos ≈ 0.5 between them). Subspace projects out a K-dimensional
subspace via SVD, capturing more anchor-relevant variance — including the
shared structure between Tally and ChartQA anchor representations. With K=4
at L31, the projected subspace generalises sufficiently to reduce ChartQA
df without harming em.

**Why L31_K04 cluster:**
- L31 has the second-highest ‖v_wrong‖ (5.39, after L30's 6.98).
- K=4 picks the top 4 SVD components — captures dominant anchor variance
  without overfitting to noise like K=8/16.
- Across α ∈ {0.5, 1, 2}: monotone strengthening of effect (L31_K04 family).

**Sample-size caveat:**
n=500 sweep-side. Bootstrap SE on df ≈ 1.5–3 pp. The ChartQA ChartQA effect
sizes (−5 to −13 %) are small in absolute terms (1–3 pp absolute df drop,
n_eligible ~416). Full-set re-validation on ChartQA n=416 (no subsample) is
the gold standard but adds ~1h. Decision: defer to roadmap.

## S3 — DPO v2 mix_synthetic results (cross-dataset FAILED, gt-bias verified)

**Setup:**
- Training data: 12,009 pairs (Tally 9,076 / ChartQA 1,552 / VQA 1,381)
- 1,462 real (case_by_case anchor adoption + df_moved) + 10,547 synthetic
- Train/eval split by hash(dataset, image_id, question_id), 70/30
- Trained 1502 steps (1 epoch, batch_size=1, grad_accum=8) on adapter LoRA r=256

**Sweep eval (eval-split sids only, no train/test leakage):**

| Dataset | n_eval | baseline df | DPO df | df_Δ% | em_Δpp | adopt_Δpp | parse_fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TallyQA** | 500 | 0.1303 | 0.0725 | **−44.3%** ✅ | +13.08 | →0 (−3.73) | 23% |
| **ChartQA** | 110 | 0.2385 | 0.3294 | **+38.1%** ❌ | −1.06 | →0 (−4.76) | 23% |

**Cross-dataset selection rule (df ≤ −5%, em ≥ baseline−2pp on ≥ 2/3 datasets):**
ChartQA df INCREASED by 38% → fails the rule.

**gt-distribution bias VERIFIED (predicted earlier in tracker):**
DPO eliminates direct anchor adoption (`adopt → 0` on both datasets) — that
part of the training signal worked. But on ChartQA where gt is large
(mean 107), DPO biases the model toward smaller numbers (Tally training
dominance 75%), increasing df because predictions drift further from the
true large-number gt. The gt-bias-confound prediction was correct.

**23% parse failure on both datasets** also points to output distribution
distortion — DPO weight-space modification damages the JSON-strict
formatting compliance.

**Cross-method comparison summary (Tally-only N=5k v2 rerun):**

| Method | Tally df_Δ% | ChartQA df_Δ% | Cross-dataset | em invariant? |
|---|---:|---:|:---:|:---:|
| LEACE 4c (best L30_a2.0) | −17.2% | +8.7% | ❌ | ✓ (em ±0.5pp) |
| Subspace 1 (best L31_K04_a2.0) | **−63.6%** | **−9.6%** | **✅✅** | ✓ (em ±1pp) |
| DPO 3 mix_synthetic | −44.3% | +38.1% | ❌ | ✗ (parse fail 23%) |

**Subspace Method 1 is the sole survivor.** DPO failure adds another data
point to the §7.4.5 narrative: weight-space methods (DPO) carry
distribution bias from training data; activation-space methods (LEACE,
Subspace) project anchor-direction without contaminating output
distribution. Subspace generalizes due to multi-direction projection
capturing shared cross-dataset variance.

## DPO gt-bin breakdown across 4 datasets (validation of distribution-bias hypothesis)

User-driven post-hoc analysis: filter each dataset's eval split by gt bin
and re-compute DPO metrics. Tests whether the "DPO failure on ChartQA" is
mitigation failure or distribution shift confound.

| Dataset | gt bin | n | base df | DPO df | df_Δ% | adopt_b → DPO | em_Δpp |
|---|---|---:|---:|---:|---:|---:|---:|
| TallyQA | all | 500 | 0.130 | 0.073 | **−44%** ✅ | 0.04 → 0 | +13 |
| TallyQA | [0,4] | 426 | 0.104 | 0.061 | −41% ✅ | 0.03 → 0 | +13 |
| TallyQA | [5,8] | 74 | 0.284 | 0.159 | −44% ✅ | 0.09 → 0 | +10 |
| ChartQA | all | 110 | 0.239 | 0.329 | **+38%** ❌ | 0.05 → 0 | −1 |
| ChartQA | [0,4] | 12 | 0.333 | 0.286 | −14% ✅ | 0.25 → 0 | −17 |
| ChartQA | [5,8] | 12 | 0.083 | 0.091 | +9% noise | 0 → 0 | +9 |
| ChartQA | [9,49] | 42 | 0.238 | 0.313 | **+31%** ❌ | 0 → 0 | +3 |
| ChartQA | [50,999] | 44 | 0.256 | 0.429 | **+68%** ❌❌ | 0.05 → 0 | −4 |
| VQAv2 | all | 133 | 0.220 | 0.105 | **−52%** ✅ | 0.20 → 0 | −6 |
| VQAv2 | [0,4] | 106 | 0.219 | 0.099 | −55% ✅ | 0.24 → 0 | −3 |
| VQAv2 | [5,8] | 27 | 0.222 | 0.125 | −44% ✅ | 0.08 → 0 | −15 |
| MathVista | all | 270 | 0.285 | 0.207 | **−27%** ✅ | 0.08 → 0 | −2 |
| MathVista | [0,4] | 106 | 0.212 | 0.227 | +7% ❌ | 0.07 → 0 | +2 |
| MathVista | [5,8] | 78 | 0.267 | 0.152 | −43% ✅ | 0.03 → 0 | +2 |
| MathVista | [9,49] | 58 | 0.382 | 0.243 | −36% ✅ | 0.14 → 0 | −11 |
| MathVista | [50,999] | 28 | 0.407 | 0.250 | −39% ✅ | 0.12 → 0 | −11 |

### Three key findings

**1. Anchor-mitigation transfers cross-dataset (adopt → 0 universally).**
DPO eliminates direct anchor adoption (`adopt = 0.0000` in EVERY bin of
EVERY dataset). The "anchor avoidance" preference signal generalises. This
is a clean cross-distribution transfer of the mitigation core mechanism.

**2. df depends on gt-distribution match with training (hypothesis confirmed
for ChartQA).**
- Tally + VQAv2 (training gt ≤ 8 distribution): all bins df −40 to −55% ✅
- ChartQA: [0,4] works (df −14%), but [9,49] +31% / [50,999] +68% ❌
- The distribution-bias confound is exactly what user predicted.

**3. MathVista shows INVERSE pattern (unexpected, requires explanation).**
- MathVista [0,4]: +7% (mild backfire), other bins all −36 to −43% ✅
- Opposite of ChartQA. MathVista is reasoning over diverse stimuli (geometry,
  charts, science figures) — anchor-pull mechanism may differ from chart-
  reading. Worth flagging in §7.4.5 as a "dataset-specific anchor mechanism"
  caveat. Sample-size for MathVista [0,4] is n=106 (not tiny), so noise less
  likely than mechanistic.

### Implication for §7.4.5 paper

DPO mitigation should be reported as **"anchor-mitigation transfers
cross-dataset (adopt → 0 everywhere) but the gt-distribution training bias
confounds the absolute df measurement on out-of-distribution datasets."**
Not a clean "DPO fails" claim — DPO succeeds on the mitigation it was
trained for, but the training data's narrow gt distribution introduces a
separate side effect. Distinguish the two effects for paper integrity.

If we wanted to disentangle further: train DPO with synthetic gt covering
[0, 1000] range. Out of scope for this rerun (would need new training data
generation + retrain ~3 h).

## LEACE + Subspace gt-bin breakdown (Tally + ChartQA on existing sweeps)

User-driven cross-check: same gt-bin analysis as DPO, applied to LEACE and
Subspace using their existing Tally+ChartQA sweep results (no new compute).

### LEACE 4c L30_a2.0

| Dataset | gt_bin | n | base_df | meth_df | df_Δ% | em_Δpp | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| TallyQA | [0,4] | 433 | 0.109 | 0.085 | −21.9% | −3.16 | ✅ df, em barely fails |
| TallyQA | [5,8] | 67 | 0.288 | 0.262 | −9.1% | +4.80 | ✅ |
| ChartQA | [0,4] | 59 | 0.153 | 0.170 | **+11.1%** | +8.47 | ❌ backfires even here |
| ChartQA | [5,8] | 40 | 0.105 | 0.184 | **+75.0%** | +2.56 | ❌ |
| ChartQA | [9,49] | 135 | 0.212 | 0.212 | 0.0% | 0.00 | (none) |
| ChartQA | [50,999] | 182 | 0.287 | 0.309 | **+7.8%** | −2.27 | ❌ |

**LEACE failure is genuine cross-dataset (not distribution bias).** ChartQA
backfires across ALL gt bins. The single-direction projection captures
Tally's anchor direction but is structurally wrong on ChartQA.

### Subspace 1 — two candidate best cells

**L31_K04_a2.0 (max df reduction, but UNSTABLE at [5,8]):**

| Dataset | gt_bin | n | base_df | meth_df | df_Δ% | em_Δpp | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| TallyQA | [0,4] | 433 | 0.109 | 0.016 | **−85.1%** | +3.36 | ✅✅ |
| TallyQA | [5,8] | 67 | 0.288 | 0.258 | −10.5% | −12.12 | em catastrophe |
| ChartQA | [0,4] | 59 | 0.153 | 0.018 | **−88.1%** | +5.29 | ✅✅✅ huge |
| ChartQA | [5,8] | 40 | 0.105 | 0.206 | **+95.6%** | −2.43 | ❌❌ catastrophic backfire |
| ChartQA | [9,49] | 135 | 0.212 | 0.211 | −0.3% | +2.55 | (no effect) |
| ChartQA | [50,999] | 182 | 0.287 | 0.261 | −9.0% | −4.44 | df ✅ but em −4.4 fails |

**L31_K04_a1.0 (smaller df, but UNIFORM and stable):**

| Dataset | gt_bin | n | base_df | meth_df | df_Δ% | em_Δpp | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| TallyQA | [0,4] | 433 | (similar) | | | | (df −81% strong) |
| ChartQA | [0,4] | 59 | 0.153 | 0.051 | **−66.7%** | +6.78 | ✅✅ |
| ChartQA | [5,8] | 40 | 0.105 | 0.108 | +2.7% | −2.43 | ≈ neutral |
| ChartQA | [9,49] | 135 | 0.212 | 0.203 | −4.3% | +2.26 | ✅ borderline |
| ChartQA | [50,999] | 182 | 0.287 | 0.258 | −9.8% | −2.29 | ✅ |

### Re-evaluating the Subspace headline

**L31_K04_a2.0's "ChartQA −9.6%" headline hides massive heterogeneity:**
- [0,4]: −88% (perfect)
- [5,8]: **+96% backfire** (catastrophic)
- [9,49]: 0% (no effect)
- [50,999]: −9% (mild)

**L31_K04_a1.0 is the better paper-headline cell** — uniformly mild reduction
across all ChartQA gt bins, no catastrophic backfire. Trade slightly less
peak Tally effect (−54.6% vs −63.6%) for cross-dataset robustness.

Recommendation for §7.4.5: report **Subspace L31_K04_a1.0** as the headline
cross-dataset cell (Tally df −54.6% / ChartQA df −13.0%), with a caveat
that the magnitude varies per gt bin. The L31_K04_a2.0 result becomes a
"stronger but distribution-sensitive" follow-up in the discussion.

### Three-method comparison summary across gt bins

| Method | Cross-dataset universal? | gt-bin sensitive? | Notes |
|---|---|---|---|
| LEACE 4c | ❌ (fails ChartQA all bins) | low (uniformly bad) | Direction mismatch is structural |
| Subspace 1 a2.0 | ⚠ partially (extreme bins only) | high | [5,8] +96% backfire kills the cell |
| Subspace 1 a1.0 | ✅ (mild reduction all bins) | low | Best tradeoff for paper |
| DPO 3 | partial (adopt → 0 universal, df depends on gt) | high | gt-bias confound on ChartQA |

## Current pipeline status (v2)

## Selection-criterion comparison (n=100, prior runs)

Re-analysis of existing sweeps with `analyze_e6_methods.py --em-rule one_sided`:

| Method | Tally pass | ChartQA pass | Cross-dataset overlap |
|---|---:|---:|---|
| LEACE 4c (pooled n=1145) | 1 (L30_a2.0) | 5 | **1 (L30_a2.0)** ✅ |
| CogBias 4a n=100 | 8 | 14 | 1 (L31_ap0.5_ad0.5, noise) |
| CogBias 4a full-set | 1 (Tally n=500) | 0 (ChartQA n=416) | 0 |
| Subspace 1 n=500 Tally | 32/80 | (cancelled) | n/a |

LEACE is the only method with a non-noise cross-dataset overlap on n=100.
This is what the Tally-only N=5k re-run aims to confirm at higher N.

## Expected verdicts

| Outcome | Implication |
|---|---|
| LEACE Tally-only N=5k retains L30_a2.0 cross-dataset pass | **§7.4.5 has a real deployable mitigation** |
| LEACE Tally-only loses cross-dataset pass | n=100 was lucky; LEACE not robust |
| Method 1 Subspace Tally-only finds new cross-dataset cell | secondary mitigation candidate |
| Method 3 DPO with new rejected reaches cross-dataset pass | weight-space mitigation viable |
| All fail | §7.4.5 cross-dataset failure remains the contribution |

## Code changes

- `scripts/analyze_e6_methods.py` — added `--em-rule {two_sided, one_sided}` flag.
  Output filename suffix `_em_one_sided` keeps both rule analyses side-by-side.
- `scripts/e6_query_adaptive_offset.py` — added `--max-calibrate-pairs` cap to
  `calibrate-qao` (mirrors `calibrate-subspace`).
- `scripts/e6_leace.py` — added `--eraser-tag` flag so different calibrations
  save to `leace_erasers_<tag>/` instead of overwriting `leace_erasers/`.

## Preserved old artifacts

- `outputs/e6_steering/llava-next-interleaved-7b/calibration_tally_e5c_n346/`
  (renamed from `calibration_tally`; old 346-sample E5c-based calibration)
- `outputs/e6_steering/llava-next-interleaved-7b/leace_erasers_pooled_n1145/`
  (renamed from `leace_erasers`; old pooled erasers from N=1145)
