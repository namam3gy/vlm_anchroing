# Round 1 — Author Response to Methodology Reviewer

**Paper version BEFORE:** `docs/paper/emnlp_draft_ko.md` @ mtime 2026-05-09 01:23 KST (516 lines, workspace head `d8710b2`).
**Paper version AFTER:** `docs/paper/emnlp_draft_ko.md` @ mtime 2026-05-09 (524 lines, +8 lines).
**Date:** 2026-05-09.
**Reviewer round addressed:** `docs/paper/reviews/round1_methodology.md` (8 must-fix, 11 should-fix, borderline, high confidence).

## Summary

We accept all four CRIT items (data-correction or traceability), apply edits to all eight must-fix items (with one PARTIAL EDIT — the "6/6 → 5/5+OneVision pending" framing is reviewer mis-reading of canonical mech-panel composition; the right fix is panel-disambiguation, not reframing). We apply five of the eleven should-fix items (high-signal ones: Q2/Q3 caption fix, raw-pool-vs-per-cell-n disambiguation, PlotQA free-lunch source pin, γ-β N=1 limitation, acc(d) relabel), and DEFER three (small-n bootstrap CI on §6.2.3, 27-cell pilot grid appendix table, FLUX rendering seed) since they require either (a) new bootstrap aggregation runs or (b) sweeping the full 27-cell SVD pilot output that is not currently summarized in any canonical table. The remaining items are noted but addressed only by existing mechanism (e.g. ablate_lower_half heterogeneity is now surfaced via §5.2 prose update and §8.2 limitation; B3 max-tokens-16 cross-ref via existing Appendix D).

The story is preserved end-to-end: every headline number (E6 Table 6, E8 Table 8, γ-β ratios, L1 monotonicity 51/85, capability macro +0.41) was independently re-verified against the canonical CSV/MD source during this revision and remains untouched. All edits are local data corrections, panel-cardinality disambiguations, or methodological transparency adds.

## Decision summary table

| # | Reviewer point (verbatim) | Class | Section affected | Status |
|---|---|---|---|---|
| 1 | CRIT-1 §5.1 Table 5 Δem column does not match canonical (LLaVA/ConvLLaVA swap, "+0.85 ConvLLaVA" not in canonical) | EDIT | §6.1 Table 5 | done |
| 2 | CRIT-2 §4.3 Table 3 TallyQA gemma3-27b row absolute values wrong (0.138/0.117 → 0.074/0.053) | EDIT | §4.3 Table 3 | done |
| 3 | CRIT-3 §6.5 Table 7 ActAdd "+57 % ChartQA backfire" not traceable | EDIT (remove) | §6.5 Table 7, §5.2 Insight 2, §6.4 prose | done |
| 4 | CRIT-4 Abstract "5 dataset × 7 model = 85 cells" arithmetic error (5×7=35, also actual is 6 datasets) | EDIT | Abstract, §1.2.3, §4.5 | done |
| 5 | MAJOR-5 §5.2 + abstract "single-layer mask ablation은 6/6 모델에서 null" misleading; OneVision is analyzer-bug | PARTIAL EDIT | Abstract, §1.3, §5.2 | done — disambiguated to "6-model mech panel" with explicit names, OneVision flagged as Phase E extension pending analyzer fix; reviewer's "5/5 + OneVision pending" reframing not adopted because OneVision was never in the canonical original 6-model mech panel (E1d Phase E extension) |
| 6 | MAJOR-6 Bootstrap CI methodology never specified | EDIT | §7 added paragraph | done — McNemar paired SE for proportion-style benchmarks + paired percentile bootstrap n=1000 for OCRBench, references `aggregate_capability_eval.py` |
| 7 | MAJOR-7 §5.1 Insight 2 "OneVision peak L20-L23 안정" conflates concentration peak vs attention peak | EDIT | §5.1 Insight 2 | done — disambiguated as digit-bbox concentration peak on calibration set; cross-dataset attention peak (L14/L27 bimodal) cited via §5.3 |
| 8 | MAJOR-8 §5.1 "각 모델은 정확히 하나의 peak layer" overstated for FastVLM (4 different layers across 4 datasets) | EDIT | §5.1 archetype para | done — softened to "calibration dataset (VQAv2)에서 단일 peak"; FastVLM/OneVision cross-dataset variability cited explicitly with `_data/cross_dataset_peaks.csv` rows |
| 9 | MAJOR-9 §6.2.3 small-n cells (n=170 / n=224) report no bootstrap CI | DEFER (with note) | §6.2.3 / §8.2 | partially addressed in §7 bootstrap paragraph (acknowledges absence and states next-revision plan); §8.2 limitation added |
| 10 | MAJOR-10 §4.3 Insight 3 PlotQA "+0.6 ~ +5.0 pp" range not pinned | EDIT | §4.3 Insight 3 | done — pinned to `docs/insights/E7-plotqa-infovqa-evidence.md` §3 |
| 11 | MAJOR-11 §4.5 Figure 6 caption Q2/Q3 wrong (paper has adopt_rate values where df should be); also "wrong-base" mislabel | EDIT | Figure 6 caption | done — Q2 0.062→0.044, Q3 0.158→0.137; relabel "wrong-base" → "all-base × cross_entropy proxy", source CSV row cited |
| 12 | MAJOR-12 §3.3 dataset n misleading (raw n vs per-cell n) | EDIT | §3.3 | done — both raw n and per-cell n reported; explicit ranges cited from `_data/main_panel_5dataset_summary.md` |
| 13 | MINOR-13 §4.6 Insight 3 "acc(b)" mislabel — actually d-arm correct fraction | EDIT | §4.6 Insight 3 | done — relabeled to acc(d) with computation traced |
| 14 | MINOR-14 §4.5 proxy choice (cross_entropy vs log_prob_sum) explanation | EDIT | §4.5 | done — both proxies reported (43/85 cross_entropy, 51/85 log_prob_sum), rationale "length-invariant default + length-aware headline" added |
| 15 | MINOR-15 §5.1 perfect-square panel cardinality not stated | EDIT | §5.1 Insight 2 | done — 4-model panel made explicit (gemma4-e4b/llava-1.5/convllava/qwen2.5-vl) |
| 16 | MINOR-16 [Bae et al., 2025] citation underspecified | EDIT | §4.6 Insight 3 | done — relabeled "Idis" reference to "test-time-compute inverse-scaling" matching paper title |
| should-fix #4 | 27-cell pilot grid appendix table | DEFER | §6.2.2 | not in any canonical aggregation; requires new sweep over `outputs/e6_steering/.../sweep_subspace_*` predictions |
| should-fix #5 | §5.2 lower-half heterogeneity surface | EDIT | §5.2 | done — surfaced 3/6 backfire pattern explicitly (Gemma +0.27 / LLaVA-1.5 +0.165 / InternVL3 +0.068) |
| should-fix #9 | §A.2 FLUX rendering seed | DEFER | §A.2 | seed lookup pending — script regenerated multiple times; need to extract from prediction archive |
| should-fix #11 | §A.1 cross-ref to D B3 max-tokens-16 | DEFER | §A.1 | low-signal cross-reference; not in this revision |

## Edit log (every paper change in this round)

### Edit 1 — §6.1 Table 5: corrected E4 Phase 2 Δem column

**Reviewer point addressed:** #1 (CRIT-1).
**Reviewer reasoning:** Paper had LLaVA-1.5 +1.30 / ConvLLaVA +0.85 / InternVL3 +0.49; canonical `docs/insights/E4-mitigation-evidence.md` Phase 2 headline is LLaVA-1.5 +0.77 / ConvLLaVA +1.30 / InternVL3 +0.49. ConvLLaVA paper +0.85 has no canonical source; LLaVA paper +1.30 is the canonical ConvLLaVA value miscopied.

**Before:**
> | **LLaVA-1.5-7b** | **−14.6 %** | **+1.30 pp** | 불변 | ±0.5 pp |
> | ConvLLaVA-7b | −9.6 % | +0.85 pp | 불변 | ±0.5 pp |
> | InternVL3-8b | −5.8 % | +0.49 pp | 불변 | ±0.5 pp |

**After:**
> **Table 5.** E4 Phase 2 결과 (88,650 records / 모델, C-form, 출처 `docs/insights/E4-mitigation-evidence.md`). Bold = 열 단위 가장 큰 효과.
>
> | **LLaVA-1.5-7b** | **−14.6 %** | +0.77 pp | 불변 | ±0.5 pp |
> | **ConvLLaVA-7b** | −9.6 % | **+1.30 pp** | 불변 | ±0.5 pp |
> | InternVL3-8b | −5.8 % | +0.49 pp | 불변 | ±0.5 pp |

**Rationale:** Bold formatting now per-column (LLaVA largest |Δdf|, ConvLLaVA largest Δem(a)) — avoids the ambiguity of bolding a row that no longer has the largest single-cell effect after correction. Source citation added inline so any future reader can re-verify against the canonical evidence file. Range "0.49–1.30 pp" cited in §1.3 / abstract is preserved (endpoints unchanged: 0.49 InternVL3, 1.30 ConvLLaVA).

### Edit 2 — §4.3 Table 3: corrected TallyQA gemma3-27b absolute values

**Reviewer point addressed:** #2 (CRIT-2).
**Reviewer reasoning:** Paper had 0.138 / 0.117 / +2.1 pp; canonical `_data/E5c_per_cell.csv` (`gemma3-27b-it,TallyQA,wrong,anchor,S1`) gives adopt_cond = 0.0737 (≈ 0.074) and (`,masked,S1`) gives 0.0532 (≈ 0.053), gap +2.05 pp ≈ +2.1 pp. The 0.138 in paper is gemma3-27b's VQAv2 row value miscopied; 0.117 has no canonical source.

**Before:**
> | TallyQA | **gemma3-27b** | 0.138 | 0.117 | **+2.1 pp** |

**After:**
> | TallyQA | **gemma3-27b** | 0.074 | 0.053 | **+2.1 pp** |

**Rationale:** Gap rounds to the same +2.1 pp, so the prose conclusion "(a − m) gap is monotonic with effect size" is preserved. The hand-copy mix-up is fixed at the cell level.

### Edit 3 — §6.5 Table 7 + §5.2 Insight 2 + §6.4: ActAdd "+57 %" untraceable

**Reviewer point addressed:** #3 (CRIT-3).
**Reviewer reasoning:** "+57 % ChartQA backfire" for ActAdd appears nowhere in `docs/insights/`, `docs/insights/_data/`, or `docs/experiments/E6-*.md`. Only +56 % LEACE is canonical (`E6-tally-only-rerun-tracker.md:480`).

**Before (Table 7):**
> | Single-direction ActAdd | ❌ ChartQA backfire +57 % | 불변 | 불변 | direction mismatch |
> | LEACE closed-form (rank-1) | ❌ ChartQA backfire +56 % | 불변 | 불변 | 동일 원인 |

**After (Table 7):**
> | Single-direction ActAdd | ❌ Cross-dataset 실패 (qualitative; per-bin TallyQA-cal v → ChartQA self-test backfire α=1, `E6-steering-vector.md`) | 불변 | 불변 | direction mismatch |
> | LEACE closed-form (rank-1) | ❌ ChartQA backfire +56 % (gt ∈ [0,8], `E6-tally-only-rerun-tracker.md:480`) | 불변 | 불변 | 동일 원인 |

**Before (§5.2 Insight 2):**
> 이는 §6.4에서 single-direction ActAdd / LEACE의 ChartQA 역행 +57 % 결과로 *직접 확인*된다.

**After (§5.2 Insight 2):**
> 이는 §6.4에서 single-direction ActAdd cross-dataset 실패 + LEACE ChartQA 역행 +56 % 결과로 *직접 확인*된다.

**Before (§6.4):**
> 이를 시도했고 cross-dataset *실패*: TallyQA에서 calibrate한 `v_wrong`을 ChartQA에 적용하면 direction-follow가 +57 % *증가*한다.

**After (§6.4):**
> 두 단일-방향 방법 (ActAdd + LEACE) 모두 cross-dataset *실패*: ActAdd는 TallyQA-calibrated `v` self-test 자체가 α=1에서 backfire (`E6-steering-vector.md`); LEACE rank-1 closed-form은 gt ∈ [0,8]로 제한해도 ChartQA에서 direction-follow를 +56 % *증가* (`E6-tally-only-rerun-tracker.md:480`).

**Rationale:** All three sites (Table 7, §5.2, §6.4) now cite an exact canonical source (file path + line number where applicable). The "single-direction failure" mechanism prediction in §5.2 and §6.4 still holds — both ActAdd and LEACE fail cross-dataset, just one is qualitative + one is the +56 % numeric. The rhetorical move is preserved without the untraceable +57 %.

### Edit 4 — Abstract + §1.2.3 + §4.5: "5×7=85" arithmetic fix

**Reviewer point addressed:** #4 (CRIT-4).
**Reviewer reasoning:** 5×7=35, not 85; and actual L1 panel is 6 datasets {VQAv2, TallyQA, ChartQA, MathVista, PlotQA, InfoVQA} × 7 models with heterogeneous coverage = 85 anchor cells (per `L1-confidence-modulation-evidence.md` 2026-05-04 update header).

**Before (Abstract):** "(5 dataset × 7 model = 85 cells)"
**After (Abstract):** "(6 dataset × 7 model heterogeneous coverage, 총 85 anchor cell)"

**Before (§1.2.3):** "5 dataset × 7 model 패널의 85개 cell 중 51 cell (60 %)"
**After (§1.2.3):** "6 dataset × 7 model heterogeneous-coverage panel의 85개 anchor cell 중 51 cell (60 %, `log_prob_sum`)"

**Before (§4.5):** "5 dataset × 7 model panel의 **85개 anchor cell에서 평균 Q4 − Q1 gap이 df +0.156** (`cross_entropy`) ~ **+0.191** (`log_prob_sum`), 완전 monotonic cell은 **51 / 85 (60 %, log_prob_sum)**"

**After (§4.5):** "**6 dataset {VQAv2, TallyQA, ChartQA, MathVista, PlotQA, InfoVQA} × 7 model heterogeneous-coverage panel의 85 anchor cell에서 평균 Q4 − Q1 gap이 df +0.156** (`cross_entropy`, length-invariant paper-clean default) ~ **+0.191** (`log_prob_sum`, length-aware), 완전 monotonic cell은 cross_entropy 43 / 85 (51 %), log_prob_sum **51 / 85 (60 %)**" + "본문 headline은 `log_prob_sum`을 보고하고 부록에서 두 proxy를 모두 표로 제공" rationale appended.

**Rationale:** Three callsites now consistent. Both proxies (cross_entropy 43/85, log_prob_sum 51/85) are now reported per MINOR-14 reviewer ask. Source citation added for traceability.

### Edit 5 — Abstract + §1.3 + §5.2 + §1.5: "6/6 mech panel" disambiguation

**Reviewer point addressed:** #5 (MAJOR-5, PARTIAL EDIT).
**Reviewer reasoning:** Reviewer asked us to restate as "5/5 mech panel + OneVision pending", but per `E1d-causal-evidence.md` the canonical 6-model mech panel is **gemma4-e4b, llava-1.5, ConvLLaVA, InternVL3, qwen2.5-vl, fastvlm** — OneVision Main was never in the original mech panel; it was added later as a Phase E extension on 4 datasets (TallyQA / InfoVQA / ChartQA / MathVista) and hit an analyzer-format bug (0.000 baseline). So "6/6 null" is canonically correct; the right fix is **disambiguating which 6 models** and **flagging the OneVision Phase E extension separately**.

**After (Abstract):** "single-layer mask ablation은 6-model 메커니즘 panel (gemma4-e4b, llava-1.5, ConvLLaVA, InternVL3, qwen2.5-vl, fastvlm)에서 6/6 null — signal은 multi-layer redundant이다 (OneVision Main 확장은 §5.3, analyzer fix pending)."

**After (§1.3):** "single-layer ablation은 6-model 메커니즘 panel에서 6/6 null — signal은 multi-layer redundant이다 (OneVision Main 확장은 분석기 수정 pending, §5.3)."

**After (§5.2 lead):** "6-model 메커니즘 panel (gemma4-e4b, llava-1.5, ConvLLaVA, InternVL3, qwen2.5-vl, fastvlm) × 200 자극 × 6 ablation mode" + appended note "OneVision Main에 대한 4-dataset 확장 (E1d Phase E)은 aggregator 0.000 baseline 분석기 버그로 본 ablation 표에 포함되지 않음 — raw prediction은 정확하며 분석기 수정 후 §5.3 narrative에 통합."

**After (§1.5):** Contribution (4) updated to "(4) 6-model 메커니즘 panel single-layer ablation null + multi-layer redundancy 분석".

**Rationale:** Reviewer's confusion ("6/6 = mech panel including OneVision?") is resolvable by naming the 6 mech-panel models and labeling OneVision as a separate Phase E extension. We do not adopt the reviewer's "5/5 mech panel + OneVision pending" reframing because that would misrepresent the canonical evidence (OneVision was never the missing 6th — fastvlm was always the 6th). See Rebuttal 1 below.

### Edit 6 — §7: bootstrap CI methodology paragraph

**Reviewer point addressed:** #6 (MAJOR-6) and partially #9 (MAJOR-9 forward acknowledgement).
**Reviewer reasoning:** §7 Table 8 reports "95 % CI" for HallusionBench / POPE / others without specifying procedure (paired vs cluster, n_resamples, percentile vs BCa, what unit gets resampled). Code source `aggregate_capability_eval.py` does specify it; just needs paper-side surfacing.

**After (§7, appended after baseline-comparison sentence):**
> **95 % CI 산출.** Proportion-style benchmark (RealWorldQA / HallusionBench / MMStar / MMBench-DEV-EN / POPE)는 baseline과 mitigation의 per-question correctness를 paired Bernoulli로 두고 McNemar 분산 추정 `SE(Δ) = sqrt(b + c) / n`으로 normal-approximation CI 산출 (`b`, `c` = paired discordant counts). Sum-style benchmark (OCRBench)는 paired percentile bootstrap (n=1,000 resample, seed 0)으로 동일 baseline-mitigation pair의 per-question score 차분에 적용. 두 절차 모두 paired (per-question 수준 resample, baseline·mitigation 간 의존성 보존), 구현 `scripts/aggregate_capability_eval.py:mcnemar_paired / bootstrap_score_diff`. §6.2.3 paired-sids cell의 점추정 deltas는 동일 paired-bootstrap 절차를 추가 적용하지 않은 상태로, 작은 denominator cell (n=170 MathVista, n=224 ChartQA)에 대한 CI 보고는 후속 revision의 직접 항목 (§8.2 한계).

**Rationale:** One paragraph, code-sourced — matches scope. Also forward-references MAJOR-9 (§6.2.3 small-n CIs) into the §8.2 limitation list (Edit 11) instead of fabricating CIs in this round.

### Edit 7 — §5.1 Insight 2: disambiguate "OneVision peak L20-L23"

**Reviewer point addressed:** #7 (MAJOR-7).
**Reviewer reasoning:** §5.1 conflates digit-bbox concentration peak (`_data/E1_patch_concentration_per_layer.csv` OneVision L20-L23 cluster on calibration set) with answer-step attention peak depth (`_data/cross_dataset_peaks.csv` OneVision L14/L27 bimodal dataset-dependent). §5.3 already correctly states the latter; §5.1 currently confuses the two.

**Before:** "5/5 main matrix dataset에서 OneVision Main의 peak이 L20-L23에 안정. Attention은 *수치 단서를 운반하는 픽셀 영역에 우선 정렬*되며..."

**After:** "OneVision Main의 *digit-bbox concentration peak*는 calibration set (VQAv2)에서 L20-L23 cluster (`L20=0.507, L23=0.492, L25=0.459`, 출처 `_data/E1_patch_concentration_per_layer.csv`); 이는 attention-mass의 *answer-step peak depth*와 다른 양으로 후자는 dataset-dependent (§5.3). Attention은 *수치 단서를 운반하는 픽셀 영역에 우선 정렬*되며..."

**Rationale:** Names the quantity (digit-bbox concentration) vs the alternate (attention-mass peak depth) and pins both to specific CSVs. §5.3 contradiction with §5.1 is resolved.

### Edit 8 — §5.1 archetype para: soften "정확히 하나의 peak layer" + add 4-model perfect-square panel cardinality

**Reviewer point addressed:** #8 (MAJOR-8) and #15 (MINOR-15).

**Before:** "각 모델은 *정확히 하나의 peak layer*를 가진다."

**After:** "각 모델은 *calibration dataset (VQAv2)에서 단일 peak layer*를 가진다. ... FastVLM과 OneVision의 *cross-dataset peak variability*는 §5.3에서 별도 다룬다 (FastVLM: VQAv2 L22 / TallyQA L23 / InfoVQA L27 / PlotQA L17 — `_data/cross_dataset_peaks.csv`; OneVision: bimodal L14/L27)."

Also for §5.1 Insight 2 (E1-patch panel cardinality): now states "4-model perfect-square panel (`gemma4-e4b`, `llava-1.5-7b`, `convllava-7b`, `qwen2.5-vl-7b` — non-perfect-square AnyRes 모델은 부록)".

**Rationale:** Categorical phrasing softened to "calibration set" qualification; cross-dataset variability for FastVLM and OneVision now surfaced explicitly with per-dataset L values from canonical CSV. Perfect-square panel cardinality (4 models) is now explicit so readers don't conflate with 6-model mech panel.

### Edit 9 — Figure 6 caption: Q2/Q3 numeric fix + "wrong-base" → "all-base × cross_entropy proxy"

**Reviewer point addressed:** #11 (MAJOR-11).
**Reviewer reasoning:** Canonical `_data/L1_confidence_quartile_long.csv` row `experiment_e5c_vqa,VQAv2,llava-next-interleaved-7b,a,S1,cross_entropy` gives df Q1=0.024 / Q2=0.044 / Q3=0.137 / Q4=0.254. Paper caption had Q2 0.062 (which is adopt_rate Q2 = 0.0649) and Q3 0.158 (adopt_rate Q3 = 0.1585) — mixed columns. Also no `base_correct` filter on this CSV row, so "wrong-base" mislabels.

**Before:** "Worked example E5c VQAv2 wrong-base S1 × LLaVA에서 df 0.024 → 0.062 → 0.158 → 0.254 (gap +23 pp)."

**After:** "Worked example E5c VQAv2 all-base × LLaVA-Interleave-7b S1 × `cross_entropy` proxy: df 0.024 → 0.044 → 0.137 → 0.254 (Q4-Q1 gap +23 pp; 출처 `_data/L1_confidence_quartile_long.csv` row `experiment_e5c_vqa,VQAv2,llava-next-interleaved-7b,a,S1,cross_entropy`)."

**Rationale:** All four df values now match canonical CSV; label "all-base" matches data shape (no base_correct stratification); CSV row pinned for traceability. Q4-Q1 gap +23 pp is preserved (Q1 and Q4 endpoints were always correct; only Q2 and Q3 needed fixing).

### Edit 10 — §3.3: raw n vs per-cell n disambiguation + main panel HF model IDs

**Reviewer point addressed:** #12 (MAJOR-12) and Reproducibility F (model IDs).
**Reviewer reasoning:** Paper §3.3 cites "ChartQA n=5,390" / "TallyQA n≈38k" / "PlotQA n≈5,000" but per-cell n in `_data/main_panel_5dataset_summary.md` is much smaller after stratification (ChartQA 129–517, TallyQA 6,934–14,772, PlotQA 926–4,610). Reviewer also wanted full HF model IDs.

**Before:** "(자연 이미지 카운팅, n≈38k), ChartQA (차트 정수 GT, n=5,390) ... PlotQA (과학 plot V1, n≈5,000) ... **Main panel 6 모델**: `llava-onevision-qwen2-7b-ov` (Main, 28-layer Qwen2-7B), Gemma3-4b/27b-it, InternVL3-8b, Qwen2.5-VL-7b/32b-Instruct."

**After:** "(자연 이미지 카운팅, raw n≈38k), ChartQA (차트 정수 GT, raw n=5,390) ... PlotQA (과학 plot V1, raw n≈5,000) ... 위 raw n은 stratification·eligibility 필터 *이전* count이며, 실제 본문 표에 사용된 per-cell n은 stratified 부분집합으로 ChartQA 129–517 / TallyQA 6,934–14,772 / PlotQA 926–4,610 / InfoVQA 218–865 / MathVista 127–274 (모델별 변동, 출처 `_data/main_panel_5dataset_summary.md`). ... **Main panel 6 모델**: `llava-onevision-qwen2-7b-ov` (Main, 28-layer Qwen2-7B), `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, `OpenGVLab/InternVL3-8B`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`. **Mechanism panel 6 모델**: gemma4-e4b, llava-1.5-7b, ConvLLaVA-7b, InternVL3-8b, qwen2.5-vl-7b, fastvlm-7b — 4 encoder archetype 분리."

**Rationale:** Both raw and per-cell ns reported, source CSV cited. All Main panel HF model IDs now explicit (was ambiguous before). Mech panel 6 models also named.

### Edit 11 — §8.2: limitations expanded (γ-β N=1, small-n CI, 27-cell grid, OneVision E1d analyzer)

**Reviewer point addressed:** #6 (should-fix γ-β limitation), #9 (MAJOR-9 small-n), and reviewer Confounds D / Reproducibility F partial mitigations.

**Before:** 4-bullet list (single prompt / open-weight / human baseline / mid-stack cluster).

**After:** 7-bullet list (added):
- "γ-β N=1 reasoning pair." — single Qwen3-VL pair, cross-architecture generalisation deferred
- "§6.2.3 small-n cell CI 미보고." — paired bootstrap CI deferred to next revision
- "§6.2.2 27-cell pilot grid." — only chosen cell shown in body, 26 rejected cells deferred to appendix
- "OneVision E1d analyzer fix." — Phase E extension pending analyzer fix

**Rationale:** All four caveats are paper-relevant methodological boundaries that reviewer flagged. Surfacing them in §8.2 is honest and matches the "no fabricated experiments" principle. Owners + timelines implicit in language ("후속 revision", "후속 부록").

### Edit 12 — §4.6 Insight 3: acc(b) → acc(d) relabel

**Reviewer point addressed:** #13 (MINOR-13).

**Before:** "Thinking은 acc(b)도 0.587 vs Instruct 0.647로 더 *낮다*."

**After:** "Thinking은 acc(d) (no-anchor neutral arm baseline proxy, γ-β 단일-stratum 설정에서 별도 b-arm run 없음 — d-arm correct fraction 사용; Instruct 0.647 = 249/385, Thinking 0.587 = 226/385 출처 `experiment_e5e_mathvista_reasoning_per_cell.csv`)도 더 *낮다*."

**Rationale:** Computation explicit; readers can re-derive both numbers from the CSV; metric label correctly identifies the d-arm proxy.

### Edit 13 — §4.3 Insight 3: PlotQA "+0.6 ~ +5.0 pp" range pinned

**Reviewer point addressed:** #10 (MAJOR-10).

**Before:** "(em delta +0.6 ~ +5.0 pp)" — no source.
**After:** "(em delta +0.6 ~ +5.0 pp; 출처 `docs/insights/E7-plotqa-infovqa-evidence.md` §3 PlotQA per-model em table)" — source pinned.

InfoVQA non-generalisation also pinned: "혼합 부호 — 같은 evidence file §4".

**Rationale:** Reader can now navigate to per-model em values directly.

### Edit 14 — §5.2: lower-half ablation heterogeneity surfaced

**Reviewer point addressed:** Should-fix #5 (Negative results E).
**Reviewer reasoning:** `E1d-causal-evidence.md` reports lower-half "3/6 BACKFIRE, 1/6 reduce, 2/6 flat" with Gemma +0.27, LLaVA-1.5 +0.165, InternVL3 +0.068 backfires. Paper hid this as "~0".

**Before:** "**Lower-half ablation**: 6/6 null."

**After:** "**Lower-half ablation**: heterogeneous (3/6 backfire — Gemma +0.27 / LLaVA-1.5 +0.165 / InternVL3 +0.068; 1/6 reduce; 2/6 flat — `E1d-causal-evidence.md`). 본문은 panel-mean ~0으로 보고하나 single-architecture-cluster 일반화 caveat 부록 §E.2 참조."

**Rationale:** Honest negative-results disclosure that distinguishes correlational signature from causal locus, exactly the methodological caveat reviewer asked for.

### Table edits

- **Table 5 (E4 Phase 2):** LLaVA-1.5 Δem(a) +1.30→+0.77; ConvLLaVA Δem(a) +0.85→+1.30; bold convention switched to per-column. Source verified: `docs/insights/E4-mitigation-evidence.md` Phase 2 headline (88,650 records).
- **Table 3 (E5c 3-model panel):** TallyQA gemma3-27b row adopt(a) 0.138→0.074, adopt(m) 0.117→0.053, gap +2.1 pp unchanged (rounding). Source verified: `_data/E5c_per_cell.csv` `gemma3-27b-it,TallyQA,wrong,{anchor,masked},S1`.
- **Table 7 (multi-method comparison):** ActAdd "+57 %" cell rewritten with qualitative ChartQA backfire claim + source citation; LEACE "+56 %" cell unchanged but source pinned to `E6-tally-only-rerun-tracker.md:480`.

### Figure edits

- **Figure 6 caption:** Q2 0.062→0.044, Q3 0.158→0.137; "wrong-base"→"all-base × cross_entropy proxy"; CSV row pinned. PNG itself unchanged (the figure is a panel of cells; the worked-example numbers in the caption text were the only error).

## Rebuttals (DISAGREE class)

### Rebuttal 1 — Reviewer point #5 (MAJOR-5, partial pushback within PARTIAL EDIT)

**Reviewer claim:** "Replace '6/6 모델에서 null' with '5 mech panel models null + OneVision pending analyzer fix'... The abstract and §5.2 should state '5 mech panel + OneVision pending' rather than '6/6'."

**Our position:** Per `E1d-causal-evidence.md` lines 40–55, the canonical 6-model mech panel is **gemma4-e4b, llava-1.5, ConvLLaVA, InternVL3, qwen2.5-vl, fastvlm**. The OneVision Main extension on 4 datasets (TallyQA / InfoVQA / ChartQA / MathVista) is explicitly described as a "Phase E extension" added later — line 8: *"OneVision aggregator output shows 0.000 baseline df on all 4 datasets — the analyzer's stratification logic does not match OneVision's per-dataset susceptibility CSV format. Raw predictions are correct... Refining the OneVision-aware analyzer is a Phase 3 follow-up."* So OneVision was never the "missing 6th" — fastvlm is always the 6th. Reframing "6/6 → 5/5 + OneVision pending" would misrepresent which model is missing.

**Why we believe the paper's position is correct:** The reviewer's underlying concern (don't claim 6/6 if OneVision is pending) is valid; we adopted it via panel-disambiguation (naming the 6 mech-panel models in Abstract / §1.3 / §5.2 / §1.5) and a separate Phase E extension flag for OneVision. This is more accurate than the reviewer's proposed reframing because it preserves the canonical 6-model panel cardinality while honestly flagging the orthogonal OneVision Phase E status. Both halves of the reviewer's concern (cardinality clarity + OneVision pending) are addressed without distorting the panel composition.

### Rebuttal 2 — Reviewer point on Figure 6 caption "wrong-base × LLaVA"

This was reviewer-correct (Edit 9) — included here only to note that the +23 pp Q4-Q1 gap survives unchanged (Q1=0.024, Q4=0.254 are correct). The correction is purely about Q2 / Q3 intermediate values and the all-base label. No rebuttal — full EDIT applied.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| MAJOR-9 §6.2.3 paired-sids small-n bootstrap CIs (n=170 MathVista, n=224 ChartQA) | `scripts/build_e6_stage4_summary.py` does not currently emit bootstrap CIs; need to extend it to apply paired bootstrap on per-paired-sid Δdf / Δem with the same procedure as §7. New aggregation run required (~30 min CPU). | Owner: paper author. Estimate: <1 day. Will use the same paired-bootstrap pipeline as `aggregate_capability_eval.py:bootstrap_score_diff` adapted to paired-sid units. Acknowledged in §8.2 limitation list (Edit 11). |
| Should-fix #4 §6.2.2 27-cell pilot grid appendix table | The 27-cell grid (`L ∈ {25,26,27} × K ∈ {2,4,8} × α ∈ {0.5,1.0,2.0}`) is in `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_*` but is not summarized in any canonical `_data/*.csv`. Building the heatmap requires aggregating ~27 prediction file pairs through the Stage-4 metric pipeline. | Owner: paper author. Estimate: 1–2 days. Will produce `_data/E6_27cell_pilot_per_cell.csv` + appendix figure. Acknowledged in §8.2 limitation list (Edit 11). |
| Should-fix #9 §A.2 FLUX rendering seed | The 128-image inventory was generated with `scripts/generate_irrelevant_number_images.py --steps 1 --guidance-scale 0`; default seed handling needs to be checked against the actual `inputs/` snapshot metadata. | Owner: paper author. Estimate: <1 hr (lookup-only). Will append to §A.2 after inspection. |
| Should-fix #11 §A.1 cross-ref to D B3 max-tokens-16 | Low-signal cross-reference; reviewer noted in passing. | Skipped this round; not paper-blocking. |

## Open questions for next round

- **Calibration-set ablation for §6.2.2 (Open methodological question 2 from reviewer):** repeat 27-cell pilot on calibration sets other than PlotQA+InfoVQA pooled (e.g. TallyQA-only, ChartQA+MathVista). Would test whether the "amplitude-correlated dataset-shared subspace" interpretation in §6.2 Insight 1 generalises beyond the chosen calibration. New GPU run required (~5–8 H100-hours per alternate calibration).
- **Cross-encoder InternVL3 H7 reversal test (Open methodological question 3 from reviewer):** currently a single-model finding; adding a second InternViT model would either reproduce or break the reversal. Owner-decision: out of scope for paper, in scope for follow-up note.
- **(a−m) wrong/correct subspace structural test (Open methodological question 4 from reviewer):** SVD on (a-arm wrong-base) − (m-arm wrong-base) vs (a-arm correct-base) − (m-arm correct-base). Would tighten the §6.3 "two effects merged in K=8" reading.
- **Hallucination task transfer (Open methodological question 5 from reviewer):** test §7 prediction on TextVQA / ScienceQA where b-arm wrong-base errors are not necessarily confidence-low. Falsifiable prediction documented.

## Internal consistency check

After all edits:

- [x] **Abstract numbers still match §4–§7 tables.** Verified: 7-model VQAv2 panel adopt 2-7 % / df 7/7 / +6.9-19.6 pp wrong-correct gap (§4.1 Table 2 ✓), 85 anchor cell / 51 monotone (§4.5 ✓), 6-model mech panel single-layer null (§5.2 ✓), Δdf [-5.2,-0.3] mean -2.9 / Δem(a) +3.9 / Δem(b) +8.8 (§6.2.3 Table 6 ✓), macro +0.41 / HB +2.21 / POPE -0.06 (§7 Table 8 ✓), γ-β ×1.6 / ×2.9 / ×12.7 (§4.6 Table 4 ✓).
- [x] **§1.5 contributions still match §4-§7 deliveries.** All six contributions trace to corresponding sections; updated (4) explicitly names "6-model 메커니즘 panel single-layer ablation null" matching §5.2; updated (6) flags "N=1 first-evidence claim" matching §4.6 + §8.2 limitation.
- [x] **§8.1 종합 still consistent with body.** Δdf avg -2.9 / Δem(a) +3.9 / Δem(b) +8.8 / macro +0.41 / HB +2.21 — all unchanged.
- [x] **All figure embeds still resolve to existing PNG paths.** No PNG paths were changed; only Figure 6 caption text changed.
- [x] **No figure or table renumbering issues introduced.** Tables 1–8 unchanged; Figures 1–7 + A1 + B1–B2 + C1–C4 + F1 + G1 unchanged.
- [x] **Canonical sources still cited where appropriate.** Six new `docs/insights/_data/` and `docs/insights/*-evidence.md` citations added inline (Edits 1, 4, 7, 8, 9, 10, 13, 14).
- [x] **§1.3 / abstract "0.49–1.30 pp" range preserved.** Range endpoints survive after Table 5 swap (0.49 InternVL3, 1.30 ConvLLaVA — unchanged endpoints, just attributed to correct rows).
- [x] **§5.1 vs §5.3 contradiction resolved.** §5.1 Insight 2 now disambiguates "digit-bbox concentration peak" (calibration set, L20-L23 cluster) from "answer-step attention peak depth" (dataset-dependent, §5.3); both quantities now traceable to distinct CSVs.

## Diff stat

- Lines: 516 → 524 (+8 lines, +1.6 %).
- Sections fully rewritten: none.
- Tables modified: 3 (Table 3 row, Table 5 column values + bold convention, Table 7 baseline-comparison cells).
- Figures modified: 1 caption (Figure 6).
- New paragraph: 1 (§7 bootstrap methodology).
- Limitations expanded: §8.2 from 4 bullets to 7.
- Word count delta (rough): +250–300 Korean words (mostly source citations + disambiguation prose, not new claims).
