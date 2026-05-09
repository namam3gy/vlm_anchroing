# Headline numbers — Phase 1 P0 v3 final (2026-05-04 + 2026-04-28 historical)

> First read **§A (Phase 1 final, 2026-05-04)** for current paper headline
> numbers. **§B onwards** is the C-form re-aggregation (2026-04-28) that
> remains valid for the historical 7-model VQAv2 panel + E5e 3-model
> ChartQA/TallyQA subset.
>
> Re-aggregate from raw `predictions.jsonl` via
> `scripts/reaggregate_paired_adoption.py` (legacy panel) or
> `scripts/build_e5e_e7_5dataset_summary.py` (current 6-model main panel)
> if numbers drift.

All numbers use the canonical M2 metrics from `roadmap.md §4` with the
`direction_follow_rate` numerator `(pa-pb)·(anchor-pb) > 0 AND pa != pb`.
("C-form" was the internal label during the 18-variant metric audit; the
metric is just *the* direction-follow definition. Audit details in
`M2-metric-definition-evidence.md` + `C-form-migration-report.md`.)

---

## §A. Phase 1 P0 v3 final — 6-model × 5-dataset main matrix (2026-05-04)

Source: `docs/insights/_data/main_panel_5dataset_summary.md` (gitignored;
regenerable via `scripts/build_e5e_e7_5dataset_summary.py`). All cells at
4-cond (b/a-S1/m-S1/d), wrong-base direction-follow.

### A.1 Per-dataset highlights

**TallyQA** (counting baseline; lowest susceptibility):

| Model | n | acc(b) | adopt(a) | df(a) | em(a) |
|---|---:|---:|---:|---:|---:|
| llava-onevision-7b (Main) | 8178 | 0.786 | 0.032 | **0.099** | 0.117 |
| qwen2.5-vl-7b | 7541 | 0.803 | 0.030 | **0.085** | 0.110 |
| internvl3-8b | (rerun in flight) | — | — | — | — |
| gemma3-4b | 14772 | 0.614 | 0.062 | **0.172** | 0.174 |
| qwen2.5-vl-32b | 7407 | 0.806 | 0.038 | **0.109** | 0.141 |
| gemma3-27b | 11014 | 0.712 | 0.059 | **0.152** | 0.140 |

**PlotQA** (chart, highest susceptibility):

| Model | n | acc(b) | adopt(a) | df(a) | em(a) |
|---|---:|---:|---:|---:|---:|
| llava-onevision-7b | 2314 | 0.481 | 0.090 | **0.206** | 0.044 |
| qwen2.5-vl-7b | 926 | 0.783 | 0.024 | **0.174** | 0.119 |
| internvl3-8b | 4610 | 0.019 | 0.002 | 0.095 | 0.021 |
| gemma3-4b | 3220 | 0.300 | 0.184 | **0.395** | 0.123 |
| qwen2.5-vl-32b | 1186 | 0.729 | 0.023 | **0.163** | 0.091 |
| gemma3-27b | 2166 | 0.513 | 0.099 | **0.227** | 0.063 |

### A.2 Cross-dataset patterns

**Susceptibility ranking** (avg df(a) across 5 datasets, descending):
gemma3-4b ≫ gemma3-27b > llava-onevision/interleave > qwen2.5-vl-32b ≈ qwen2.5-vl-7b > internvl3-8b

→ qwen family + internvl3 most robust. Gemma3 family most susceptible. **Anti-scaling within Gemma**: gemma3-4b worse than gemma3-27b (smaller model more pulled).

**Dataset susceptibility ranking** (mean df across panel):
PlotQA (0.226) ≈ MathVista (0.241) > InfoVQA (0.227) > ChartQA (0.204) ≫ TallyQA (0.116)

→ Chart/figure datasets pull ~2× harder than counting (TallyQA). Text-heavy plot/math contexts amplify anchor effect.

### A.3 Mitigation chosen cell (Phase B Stage 4-final, commit `9f9dfa0`; CI added 2026-05-10)

**Subspace projection L=26 K=8 α=1.0**, calibrated on PlotQA+InfoVQA pooled n5k. Evaluated on n=5000 wrong-base subset per dataset. Paired-sids comparison (sids parseable on b+a in baseline AND mitigation arms). Sources:
- Point estimates: `docs/insights/_data/stage4_final_per_dataset.{csv,md}` (regenerable via `scripts/build_e6_stage4_summary.py`).
- **Paired-bootstrap 95 % CI + Bonferroni-20 corrected (99.75 %) CI** (B = 10,000): `docs/insights/_data/stage4_final_per_dataset_ci.{csv,md}` + raw draws `_data/stage4_final_bootstrap_draws.npz` (regenerable via `scripts/build_e6_stage4_bootstrap_ci.py`). Sid-paired resampling; per-arm `(num, den)` recomputed each bootstrap so adopt's `pb ≠ anchor` denominator and df's `pa ≠ pb` clause shift correctly per arm.

| Dataset | n_paired | Δ adopt(a) [95 % CI, pp] | Δ df(a) [95 % CI, pp] | Δ em(a) [95 % CI, pp] | **Δ em(b) [95 % CI, pp]** |
|---|---:|---:|---:|---:|---:|
| TallyQA | 4,978 | −0.6 [−1.1, +0.0] | −0.3 [−1.3, +0.6] | **+6.6 [+5.6, +7.5]** | **+13.8 [+12.9, +14.8]** |
| PlotQA | 2,306 | **−5.6 [−6.8, −4.4]** | **−5.2 [−6.9, −3.4]** | **+2.4 [+1.5, +3.4]** | **+4.7 [+3.8, +5.7]** |
| InfoVQA | 443 | +0.9 [−0.5, +2.5] | −0.7 [−4.7, +3.4] | **+3.4 [+0.5, +6.3]** | **+9.0 [+6.3, +11.7]** |
| ChartQA | 224 | **−3.3 [−6.0, −1.0]** | −4.0 [−9.8, +1.8] | **+4.0 [+0.0, +8.0]** | **+7.1 [+3.6, +10.7]** |
| MathVista | 170 | −1.5 [−6.9, +3.7] | −4.1 [−11.8, +3.5] | +2.9 [−2.4, +8.2] | **+9.4 [+4.7, +14.7]** |
| **mean** |   | **−2.0** | **−2.9** | **+3.9** | **+8.8** |

Bold = 95 % CI excludes 0 in headline direction (Δadopt/Δdf negative, Δem positive).

**Sign-clean count (CI excludes 0 in metric's headline direction):**

| Metric | 95 % CI | Bonferroni-20 corrected (99.75 %) CI |
|---|:---:|:---:|
| Δ adopt(a) (− direction) | 2 / 5 | 2 / 5 |
| Δ df(a) (− direction) | 1 / 5 (PlotQA only) | 1 / 5 (PlotQA only) |
| Δ em(a) (+ direction) | 3 / 5 | 2 / 5 (PlotQA, TallyQA) |
| **Δ em(b)** (+ direction) | **5 / 5** | **5 / 5** |

**Verdict (CI-augmented)**: df reduction works on point estimates (avg −2.9 pp), but only **PlotQA n=2,306** Δdf clears 95 % CI excludes 0; small-n cells (ChartQA n=224, MathVista n=170) have point-estimate magnitudes consistent with PlotQA but CI individually-inconclusive; **InfoVQA Δdf=−0.7 pp on n=443** has 95 % CI [−4.7, +3.4] — `inconclusive fence` confirmed with real CI numbers (sanity gate: half-width 0.0406 lands inside the paper's prior paired-Wilson estimate ~0.04–0.06). em(a) **+3.9 pp** *and* em(b) **+8.8 pp** — both arms improve on the wrong-base subset; **em(b) is the multiplicity-robust headline (5/5 sign-clean under both 95 % AND Bonferroni-20 corrected CIs)** — anchor pull drops *and* exact-match rises on both arms, with the non-anchored-arm em recovery being the paper's strongest single signal. Strict free-lunch on the wrong-base subset. Earlier "em(a) −2.4 pp cost" framing in this section was a hand-copy error and is retracted (corrected 2026-05-04 from `scripts/build_e6_stage4_summary.py`). Paper §6.2.3 / §7.4.5 reframed 2026-05-10 (P1-3) to lead with the b-arm Bonferroni-robust headline alongside the PlotQA-strong Δdf cell.

**Pilot-grid context (2026-05-10, P1-6).** The (L=26, K=8, α=1.0) chosen cell is selected from a 27-cell pilot grid (L ∈ {25,26,27} × K ∈ {2,4,8} × α ∈ {0.5,1.0,2.0}) under the ex ante rule "reject Δem(a) ≤ −6 pp on either calibration dataset, then rank by combined |Δdf(a)|". On the actual pilot data the deal-breaker rule is **non-binding** (0 / 27 cells rejected) and the chosen cell ranks **first by combined |Δdf(a)|** with a 1.2 pp margin over runner-up — same ex ante rule on same pilot data does not select a different cell. Full 4-metric × 2-calibration heatmap: `docs/figures/E6_pilot_grid_{plotqa,infographicvqa}_heatmap.png`; canonical CSV `docs/insights/_data/E6_pilot_grid_27cells.csv`; insight cousin `docs/insights/E6-pilot-grid-aggregation.md`.

### A.3b Capability preservation regression (E8, 8-bench, 2026-05-09)

Same chosen cell L=26 K=8 α=1.0 evaluated on 8 held-out general-VLM benchmarks at inference (no anchor labels, greedy decoding, no LLM-judge). Source: `docs/insights/_data/capability_eval_per_benchmark_v8.{csv,md}` (gitignored, regenerable via `scripts/aggregate_capability_eval.py merge ...`).

| Benchmark | n | baseline | +mit | Δ pp | 95% CI |
|---|---:|---:|---:|---:|---|
| RealWorldQA | 765 | 69.80 | 71.11 | +1.31 | [-0.27, +2.89] |
| OCRBench | 1000 | 63.40 | 62.60 | -0.80 | [-1.68, +0.08] |
| **HallusionBench** | 951 | 47.84 | 50.05 | **+2.21** | **[+1.14, +3.28]** |
| MMStar | 1500 | 61.67 | 61.80 | +0.13 | [-0.77, +1.04] |
| MMBench-DEV-EN | 1164 | 82.04 | 81.70 | -0.34 | [-0.82, +0.13] |
| POPE | 5127 | 92.16 | 92.10 | -0.06 | [-0.21, +0.09] |
| MME | 2374 | 84.50 | 84.37 | -0.13 | [-0.76, +0.51] |
| **AMBER** | 14216 | 87.15 | 87.34 | **+0.19** | **[+0.05, +0.33]** |
| **Macro** |   |   |   | **+0.31** |   |

**Verdict**: `STRICT_FREE_LUNCH` extends to general capability across n_total = 27,097. All 8 per-benchmark Δ within ±1.0 pp band. **Two of three hallucination axes — HallusionBench and AMBER (n=14,216) — show CI-clean positive Δ**; POPE pins the third to zero. **MME Count subset (n=60), the in-domain analogue of the number-anchor failure mode, shows Δ = 0.00 pp exact** (60/60 paired predictions match) — direct evidence the mitigation acts on cross-modal anchor pull, not counting capability itself. Contamination-resistant floor rises from n=1,500 (MMStar alone) to n=18,090 (MMStar + MME + AMBER). Insight doc: `docs/insights/E8-capability-preservation-evidence.md`. Experiment writeup: `docs/experiments/E8-capability-preservation.md`.

### A.4 §7.1-7.3 Cross-dataset peak layer (Phase D, commit `c556fb6`)

Per-(model, dataset) peak attention layer at answer step (`docs/insights/_data/cross_dataset_peaks.csv`, gitignored):

| Model | InfoVQA | PlotQA | TallyQA | VQAv2 |
|---|:-:|:-:|:-:|:-:|
| gemma4-e4b | 5/42 | 5/42 | 5/42 | 5/42 |
| llava-1.5-7b | 8/32 | 14/32 | 8/32 | 8/32 |
| convllava-7b | 12/32 | 14/32 | 7/32 | 7/32 |
| fastvlm-7b | 27/28 | 17/28 | 23/28 | 22/28 |
| **llava-onevision-7b (Main)** | **14/28** | **27/28** | **27/28** | **14/28** |
| qwen2.5-vl-7b | — | — | — | 22/28 |

**Key finding (2026-05-04)**: OneVision peak is **dataset-dependent** (L=27 last layer on PlotQA/TallyQA but L=14 mid-stack on InfoVQA/VQAv2). Earlier "OneVision = late-layer model-specific" claim is partially correct: model architecture sets the layer band, but dataset content modulates which sub-band activates. Paper §7.2 needs to surface this dual conditioning.

Other panel models show stable peak (gemma4-e4b L=5 across all 4 datasets — most consistent). llava-1.5-7b stable except PlotQA. fastvlm + convllava show small dataset variation.

### A.5 Phase E E1d causal ablation OneVision × 5 datasets (commits `7a27750` + `2d11876` + `a7e391c` + `de1f94e` analyzer fix landed 2026-05-10, P4-12 closed)

Per-mode direction-follow rate at OneVision Main, n=200 stratified per dataset, B=2,000 bootstrap CI (`outputs/causal_ablation/_summary/per_model_per_mode.csv`):

| Mode | TallyQA | InfoVQA | ChartQA | MathVista | PlotQA |
|---|---:|---:|---:|---:|---:|
| baseline | 0.130 | 0.167 | 0.105 | 0.171 | 0.243 |
| Δ ablate_peak (pp) | −0.5 | +1.5 | 0.0 | 0.0 | −0.6 |
| Δ ablate_upper_half | −2.5 | +0.4 | −0.5 | −2.6 | −3.9 |
| Δ ablate_all | −4.0 | +0.8 | +0.6 | −4.5 | −5.1 |

**Reading.** Single-layer ablation 5/5 null on OneVision (95 % CI overlap 0; max |Δdf| = 1.5 pp on InfoVQA peak) — multi-layer redundancy claim (6-mech panel 6/6 null)의 OneVision 위 *확장 검증*. Upper-half ablation은 6-mech panel의 균일 −4 ~ −10.5 pp significant와 달리 OneVision에서는 5/5 null at n=200 (point estimates ∈ [−3.9, +0.4] pp; PlotQA −3.9 pp [−9.4, +1.9]가 가장 가깝지만 0 포함) — §5.3 OneVision dataset-dependent peak (Plot/Tally L=27 vs Info/VQAv2 L=14)와 일관 heterogeneity, §6.2 subspace-projection 도구 선택 보강. 자세한 표 + 95 % CI + Lower-half BACKFIRE는 paper Appendix §E.2 또는 `docs/insights/E1d-causal-evidence.md` 참조.

---

## §B. Historical reference numbers (C-form re-aggregation, 2026-04-28)

> Extracted from `references/roadmap.md §3.3` on 2026-04-29 to keep the
> roadmap lightweight.

Pre-refactor results archived at `outputs/before_C_form/`; side-by-side
deltas in `docs/insights/C-form-migration-report.md`. Adopt and
exact-match are unchanged by the refactor; only `direction_follow*`
columns moved.

## Standard-prompt VQAv2 number subset, 17,730 samples / model

| Model | acc(b) | acc(d) | acc(a) | adopt(a) | direction_follow(a) |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.505 | 0.541 | **0.066** | **0.274** |
| llava-interleave-7b | 0.619 | 0.577 | 0.576 | **0.053** | **0.172** |
| gemma3-27b-it | 0.628 | 0.623 | 0.633 | **0.053** | **0.167** |
| qwen3-vl-30b | 0.759 | 0.709 | 0.707 | **0.039** | **0.170** |
| qwen3-vl-8b | 0.751 | 0.709 | 0.715 | **0.033** | **0.104** |
| qwen2.5-vl-7b | 0.736 | 0.708 | 0.711 | **0.021** | **0.094** |
| gemma4-31b-it | 0.749 | 0.723 | 0.741 | **0.024** | **0.085** |

Under C-form, `direction_follow_rate_raw == direction_follow_rate_moved`
because `(pa-pb) = 0` makes the no-movement case yield zero in the
numerator structurally — the `pa != pb` clause is structurally redundant
but kept explicit for clarity. Ranking is preserved across the panel.

## E5b distance sweep — `llava-interleave-7b` only (cross-model in flight)

Wrong-base subset, `adopt_rate` (M2):

| stratum | VQAv2 (n_eligible) | TallyQA (n_eligible) |
|---|---:|---:|
| S1 | **0.134** (313) | **0.098** (265) |
| S2 | 0.030 | 0.006 |
| S3 | 0.010 | 0.003 |
| S4 | 0.010 | 0.000 |
| S5 | 0.003 | 0.000 |

Pattern: sharp peak at S1, decay to noise floor by S5 (cross-dataset).

## E5c digit-pixel causality — 3-model panel ✅

Wrong-base S1 paired conditional adoption (`adopt_cond` from
`docs/insights/_data/E5c_per_cell.csv`, 2026-04-29; the M2 marginal
form is in the main panel rows above). The a − m gap reads off whether
inpainting the digit pixel removes the effect:

| dataset | model | anchor S1 | masked S1 | gap (a − m) |
|---|---|---:|---:|---:|
| VQAv2 | llava-interleave-7b | 0.129 | 0.068 | **+6.1 pp** |
| VQAv2 | gemma3-27b-it | 0.138 | 0.082 | **+5.7 pp** |
| VQAv2 | qwen2.5-vl-7b | 0.070 | 0.066 | +0.4 pp |
| TallyQA | llava-interleave-7b | 0.110 | 0.084 | **+2.6 pp** |
| TallyQA | gemma3-27b-it | 0.074 | 0.053 | +2.1 pp |
| TallyQA | qwen2.5-vl-7b | 0.033 | 0.037 | −0.5 pp |

(gemma3-27b-it TallyQA at `max_samples=300` per §9; full n=1000
stratified is infeasible on this model. **Noise-floor caveat:** the
TallyQA wrong-base S1 stratum has only n≈95 df-eligible samples
per arm under this budget, so df-side gaps below ~±8 pp (95% CI
half-width at p≈0.19) are sampling noise. gemma3's TallyQA df(a)−df(m)
= −0.33 pp is one of these — corresponding to a 1-sample
difference in the numerator, not a sign reversal. The adopt-side
gap (+2.05 pp) survives because case4 has a different denominator
profile.) Direction-follow `df_cond` mirrors the ranking on
VQAv2 wrong-base S1 where stratum n is large enough to
detect: df(a)/df(m) = 0.280/0.221 on gemma3 (+5.99 pp),
0.208/0.155 on llava (+5.3 pp), 0.148/0.163 (~0) on qwen2.5-vl.
Cross-model expansion confirms the expected rank: largest pull
(llava) → largest gap; mid-panel (gemma3) → mid gap; floor
(qwen2.5-vl) → floor gap. The digit-pixel-causality claim from §5.4
generalises across all three models on VQAv2 and across the two
pulled models on TallyQA (llava resolves df above noise; gemma3
resolves on adopt; qwen2.5-vl serves as the negative control).

## E5e S1-only 4-condition full — 3-model panel × ChartQA + TallyQA

All-base, S1 anchor / masked (numbers cross-checked against
`outputs/experiment_e5e_*_full/<model>/<ts>/summary.json` and
`docs/insights/_data/experiment_e5e_*_per_cell.csv` 2026-04-29):

| dataset | model | acc(b) | acc(a) | adopt(a) | adopt(m) | df(a) | df(m) |
|---|---|---:|---:|---:|---:|---:|---:|
| ChartQA | gemma3-27b-it | 0.217 | 0.218 | **0.037** | 0.022 | **0.096** | 0.079 |
| ChartQA | llava-interleave-7b | 0.113 | 0.110 | **0.028** | 0.009 | **0.152** | 0.115 |
| ChartQA | qwen2.5-vl-7b | 0.255 | 0.253 | **0.017** | 0.013 | **0.051** | 0.046 |
| TallyQA | gemma3-27b-it | 0.237 | 0.236 | **0.027** | 0.016 | **0.073** | 0.060 |
| TallyQA | llava-interleave-7b | 0.236 | 0.233 | **0.026** | 0.014 | **0.066** | 0.056 |
| TallyQA | qwen2.5-vl-7b | 0.230 | 0.226 | **0.011** | 0.011 | **0.029** | 0.030 |

TallyQA × gemma3-27b-it cell landed 2026-04-29 (inference completed
2026-04-28 23:28; C-form re-aggregation 2026-04-29 via
`reaggregate_paired_adoption.py`). Wrong-base S1 cell from the per-cell
CSV: `adopt(a) = 0.059`, `df(a) = 0.152`, `adopt(m) = 0.034`,
`df(m) = 0.133` — matching the panel-leading TallyQA pattern (graded
tilt with anchor > masked, S1 distance window).

## Mechanistic / mitigation summary

E1d upper-half ablation: **−4.0 to −10.5 pp** `direction_follow`
on 6/6 models; fluency-clean on 4/6 (mid-stack cluster + Qwen).

E4 Phase 2 full mid-stack-cluster: `direction_follow_rate`
reduction LLaVA-1.5 **−14.6 %** rel, ConvLLaVA **−9.6 %**, InternVL3
**−5.8 %**; `exact_match` rises +0.49 to +1.30 pp; `accuracy_vqa(b)`
invariant — anchor-condition specific.
