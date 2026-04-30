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
