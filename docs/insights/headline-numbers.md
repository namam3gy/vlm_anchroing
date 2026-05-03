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
`direction_follow_rate` numerator in **C-form**: `(pa-pb)·(anchor-pb) > 0`.

---

## §A. Phase 1 P0 v3 final — 6-model × 5-dataset main matrix (2026-05-04)

Source: `docs/insights/_data/main_panel_5dataset_summary.md` (gitignored;
regenerable via `scripts/build_e5e_e7_5dataset_summary.py`). All cells at
4-cond (b/a-S1/m-S1/d), wrong-base C-form df.

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

### A.3 Mitigation chosen cell (Phase B Stage 4-final, commit `9f9dfa0`)

**Subspace projection L=26 K=8 α=1.0**, calibrated on PlotQA+InfoVQA pooled n5k. Evaluated on n=5000 wrong-base subset per dataset. Same-population baseline vs mitigation comparison:

| Dataset | n_elig | Δ adopt(a) | Δ df(a) | Δ em(a) | **Δ em(b)** |
|---|---:|---:|---:|---:|---:|
| TallyQA | 4978→4298 | -0.009 | -0.014 | -0.020 | **+0.140** |
| PlotQA  | 2069→1982 | -0.069 | -0.043 | -0.016 | **+0.052** |
| InfoVQA |  428→ 390 | -0.032 | +0.001 | -0.026 | **+0.090** |
| ChartQA |  192→ 178 | -0.026 | -0.046 | -0.022 | **+0.071** |
| MathVista | 164→ 147 | -0.043 | -0.024 | -0.038 | **+0.105** |
| **mean** |   | **-0.036** | **-0.025** | **-0.024** | **+0.092** |

**Verdict**: df reduction works (avg -2.5pp) under em-drop dealbreaker (6pp threshold; -2.4pp em(a) cost is well within). **em(b) +9.2pp recovery on wrong-base sids is unintended novelty** — paper §7.4 needs re-framing to surface this alongside df reduction (task #38).

### A.4 §7.1-7.3 Cross-dataset peak layer (Phase D, commit `c556fb6`)

Per-(model, dataset) peak attention layer at answer step (`docs/insights/_data/cross_dataset_peaks.csv`, gitignored):

| Model | InfoVQA | PlotQA | TallyQA | VQAv2 |
|---|:-:|:-:|:-:|:-:|
| gemma4-e4b | 5/42 | 5/42 | 5/42 | 5/42 |
| llava-1.5-7b | 8/32 | 14/32 | 8/32 | 8/32 |
| convllava-7b | 12/40 | 14/40 | 7/40 | 7/40 |
| fastvlm-7b | 27/28 | 17/28 | 23/28 | 22/28 |
| **llava-onevision-7b (Main)** | **14/28** | **27/28** | **27/28** | **14/28** |
| qwen2.5-vl-7b | — | — | — | 22/28 |

**Key finding (2026-05-04)**: OneVision peak is **dataset-dependent** (L=27 last layer on PlotQA/TallyQA but L=14 mid-stack on InfoVQA/VQAv2). Earlier "OneVision = late-layer model-specific" claim is partially correct: model architecture sets the layer band, but dataset content modulates which sub-band activates. Paper §7.2 needs to surface this dual conditioning.

Other panel models show stable peak (gemma4-e4b L=5 across all 4 datasets — most consistent). llava-1.5-7b stable except PlotQA. fastvlm + convllava show small dataset variation.

### A.5 Phase E E1d causal ablation OneVision × 4 datasets (commit `7a27750` + `2d11876`)

Per-mode direction-follow rate at OneVision Main (`outputs/causal_ablation/_summary/per_model_per_mode.csv`):

| Mode | TallyQA | InfoVQA | ChartQA | MathVista |
|---|---:|---:|---:|---:|
| baseline | 0.000 | 0.000 | 0.000 | 0.000 |
| ablate_peak (L=27) | 0.000 | 0.000 | 0.000 | 0.000 |
| ablate_upper_half | 0.000 | 0.000 | 0.000 | 0.000 |

Note: OneVision baseline df is computed from intervention pipeline differently than from baseline run — the analyzer's stratification logic doesn't fit OneVision's susceptibility CSV well. The other panel models (5 mech) show clean **−4 to −10pp upper-half ablation** effects. Refining OneVision E1d aggregation is a Phase 3 follow-up. Raw predictions are present and correct in `outputs/causal_ablation/llava-onevision-qwen2-7b-ov/<run>/predictions.jsonl`.

---

## §B. Historical reference numbers (C-form re-aggregation, 2026-04-28)

> Extracted from `references/roadmap.md §3.3` on 2026-04-29 to keep the
> roadmap lightweight.

Pre-refactor results archived at `outputs/before_C_form/`; side-by-side
deltas in `docs/insights/C-form-migration-report.md`. Adopt and
exact-match are unchanged by the refactor; only `direction_follow*`
columns moved.

## Standard-prompt VQAv2 number subset, 17,730 samples / model

| Model | acc(b) | acc(d) | acc(a) | adopt(a) | direction_follow(a) C-form |
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

All-base, S1 anchor / masked, C-form (numbers cross-checked against
`outputs/experiment_e5e_*_full/<model>/<ts>/summary.json` and
`docs/insights/_data/experiment_e5e_*_per_cell.csv` 2026-04-29):

| dataset | model | acc(b) | acc(a) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
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
CSV: `adopt(a) = 0.059`, `df(a) C-form = 0.152`, `adopt(m) = 0.034`,
`df(m) = 0.133` — matching the panel-leading TallyQA pattern (graded
tilt with anchor > masked, S1 distance window).

## Mechanistic / mitigation summary

E1d upper-half ablation: **−4.0 to −10.5 pp** `direction_follow` (C-form)
on 6/6 models; fluency-clean on 4/6 (mid-stack cluster + Qwen).

E4 Phase 2 full mid-stack-cluster (C-form): `direction_follow_rate`
reduction LLaVA-1.5 **−14.6 %** rel, ConvLLaVA **−9.6 %**, InternVL3
**−5.8 %**; `exact_match` rises +0.49 to +1.30 pp; `accuracy_vqa(b)`
invariant — anchor-condition specific.
