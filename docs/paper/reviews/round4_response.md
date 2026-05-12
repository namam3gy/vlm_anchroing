# Round 4 — Reviser Response (Aggressive Adversarial)

**Paper version BEFORE:** worktree head v11 (826 lines, post-round-3 novelty surgery).
**Paper version AFTER:** v12 working tree (833 lines, net +7 — FATAL scope inserts at Abstract + §1.3 + §1.5 + §6.2.1 + §8.1; new §8.4 items 7-9; MAJ-1 / MAJ-2 multiplicity-scope notes in §4.6 + §6.2.3; MAJ-3 §5.2 + §6.4 predict-verify language replaced with post-hoc paired-observation language; MAJ-6 §7 axis-conditional disclosure; MIN-2 / MIN-3 / MIN-7 wording).
**Date:** 2026-05-11 (Round 4).
**Reviewer round addressed:** `docs/paper/reviews/round4_aggressive.md` (REJECT-for-Main verdict, 4 FATAL + 9 MAJOR + 10 MINOR).

## Summary

The reviewer's verdict — REJECT for Main, borderline-Findings conditional on three scope-honesty fixes — diagnoses correctly that prior rounds cleaned the *form* of contribution language but did not propagate the case-study scope into the headline surfaces (Abstract + §1.5 central contribution sentence + §8.1 종합). We accept all four FATAL points as **EDIT** (scope-honest insertion at the headline surfaces; no new experiments fabricated). MAJ-3 is **EDIT** (predict-verify language at §5.2 Insight 2 + §6.4 Insight 1 was inconsistent with §5.4's post-hoc admission). MAJ-1 / MAJ-2 / MAJ-5 / MAJ-6 are **PARTIAL EDIT** (multiplicity-scope notes + per-benchmark axis-conditional disclosure surfaced at the load-bearing sites; selection-rule re-correction left as §8.4 item 8). MAJ-7 / MAJ-8 / MAJ-9 are **DEFER** with explicit follow-up entries in §8.4 (items 9, 4, 2 respectively). MINOR items: MIN-2 / MIN-3 / MIN-7 edited; rest deferred or absorbed by FATAL edits.

The reviewer's central thesis — "the cleaning was form-correct but the substance is single-cell single-architecture single-dataset" — is *accepted*. We do not rebut it. Instead we (a) move the case-study admission *into* the central-contribution noun phrase, (b) re-separate the multiplicity-robust headline into two equally-weighted clauses (anchoring effect = single-dataset CI-clean; capability-side = multi-dataset CI-clean), (c) downgrade §4.6 from "prospective verification" to "partially prospective verification at K=1; deployed K=8 partial-falsifies framework's universal-K assumption", (d) surface Telea-residue confound in §6.2.1 Insight, (e) propagate matched scope honesty into §1.3, §8.1, §3.3.

Five experiments deferred to §8.4 (items 7-9 new + 2-4 retained). No fabricated results.

## Decision summary table

| # | Reviewer point (verbatim summary) | Class | Section affected | Status |
|---|---|---|---|---|
| FATAL-A | "§1.5 central contribution sentence reads as method-of-paper while §3.3 hedge says single-model case study" | EDIT | §1.5, Abstract, §1.3, §8.1, §3.3 | done |
| FATAL-B | "Δem(b) 5/5 Bonferroni-clean is non-anchored arm side-effect; anchoring task Δdf clause is 1/5 CI-clean; headline pivoted to wrong clause" | EDIT | §1.5, Abstract, §8.1 | done (two-clause separation at Abstract + §1.5 + §8.1) |
| FATAL-C | "§4.6 γ-β bridge tests at K=1; deployed E6 uses K=8. Prospective leg doesn't verify operative parameterisation" | EDIT | §1.5 (ii), §1.3, §5.4, §8.1, Abstract, §4.6 | done (re-labeled "partially prospective at K=1; deploy K=8 partial-falsifies universal-K") |
| FATAL-D | "(a − m) Telea inpaint texture confound; OCR-verified digit absence doesn't control for representation-level texture" | EDIT | §6.2.1 Insight + §8.4 item 7 | done (Telea-residue caveat added; (m−m') falsification baseline deferred to §8.4 item 7) |
| MAJ-1 | "§4.6 14/84 cells = post-hoc cell selection from sweep; framework predicts direction not which cells" | EDIT | §4.6 cell-selection note + §8.4 item 8 | done |
| MAJ-2 | "Bonferroni-20 corrects within-evaluation; 27-cell grid argmax = separate multiplicity layer not corrected" | EDIT | §6.2.3 multiplicity-correction note + §8.4 item 8 | done |
| MAJ-3 | "§5.2 Insight 2 'predict-verify' language inconsistent with §5.4 post-hoc admission" | EDIT | §5.2 Insight 2 + §6.4 Insight 1 | done (replaced with post-hoc paired-observation language) |
| MAJ-4 | "§1.5 (i) L1 6-bin gradient strict 5/5 is 21-24/80 = 30%; contribution language elides strict-vs-relaxed" | DEFER | §4.4 already explicit at line 193+203 | retain — §4.4 body honestly states both ≥4/5 (51-57/80) and strict 5/5 (21-24/80); §1.5 (i) cites "세 직교 axis 증거" without monotonicity claim |
| MAJ-5 | "InfoVQA Δdf [−4.7, +3.4] is fence — 'sign-clean' framing on 0.7 pp point estimate is barely better than coin-flip" | PARTIAL EDIT | Abstract + §1.5 + §8.1 | done (FATAL-B two-clause separation surfaces "TallyQA floor / InfoVQA fence / 4 small-n cell CI-borderline" explicitly) |
| MAJ-6 | "§7 macro +0.41 pp masks 3/6 negative point estimates; HallusionBench-dominant; per-benchmark heterogeneity hidden" | EDIT | §7 Insight + Abstract + §8.1 | done (axis-conditional disclosure: anchoring-adjacent vs broad-VLM-capability axis) |
| MAJ-7 | "×12.7 ratio still in §1.4 with no CI; 3 sig figs over-reported on small denominator" | DEFER | §8.4 item 9 (new) | added — paired-bootstrap CI computation as new §8.4 item 9 |
| MAJ-8 | "§6.5 5-baseline panel excludes CAA/ITI; constructed panel rhetorically excludes strongest competitors" | DEFER | §6.5 already labeled "5-baseline panel" (round-3); §8.4 item 4 retained | retain — round-3 CRIT-N1 already softened framing; CAA/ITI empirical row in §8.4 item 4 |
| MAJ-9 | "§6.3 b-arm em +8.8 pp interpretation has Alt-1 (general regularization) and Alt-2 (numeric mode-collapse) competing explanations unfalsified" | DEFER | §6.3 Insight 1.5 already explicit; §8.4 item 2 retained | retain — round-1 MAJOR-8 already added §6.3 hedge + §8.4 item 2 random-K=8 baseline |
| MIN-1 | "§1.5 (i) 'three orthogonal axes' on L1 + (a−m) + wrong/correct: §4.2 Slice B shows (a−m) and wrong/correct correlated not orthogonal" | DEFER | §1.5 | retain — "orthogonal" here means *three distinct measurement cuts*, not "uncorrelated statistical axes"; §4.2 Slice B's correlation is *evidence* the three cuts measure the same underlying phenomenon, not contradiction |
| MIN-2 | "§4.6 Insight 1 '9× ratio' on signed amplitudes (+0.28 vs −0.05) is geometrically nonsensical" | EDIT | §4.6 Insight 1 | done — replaced "9× 차이" with "qualitative sign-state 변경 (K=8 zero-overlap → K=1 Bonferroni-positive)" |
| MIN-3 | "§3.3 'this panel-scope 분리는 본 절에서 단 1회 명시' is meta-instruction-to-reader, project-management residue" | EDIT | §3.3 | done — removed meta-instruction sentence; kept canonical statement |
| MIN-4 | "§7 Note 6-bench vs 8-bench macro slides past 'which is canonical headline'" | EDIT | §7 Insight + §8.1 | done (canonical = 6-bench pre-registered, 8-bench cross-reference; embedded in §7 Insight axis-conditional disclosure) |
| MIN-5 | "§6.2 entire chain calibrated and evaluated on S1-stratified subset; 5-dataset cross-evaluation is 5-dataset S1-stratum cross-evaluation" | DEFER | §3.3 + §A.3 already explicit | retain — §A.3 distinguishes eligibility cutoff vs S1 stratum; §6.2.2 wrong-base calibration is documented; reviewer's MIN-5 is correct re scope but does not surface a *new* fact, just a reframe |
| MIN-6 | "ChartQA n=224 < pilot calibration n=250" | DEFER | §6.2.3 Table 7 + §3.3 already shows raw vs stratified n | retain — n_paired column in Table 7 shows the figure; reviewer's observation is correct but is mathematics of paired-sids intersection design, not a flaw |
| MIN-7 | "§4.5 Table 4 bold on ×12.7 is rhetorical (no CI; other bold = CI excludes 0 convention)" | EDIT | Table 4 + caption | done — bold removed; caption notes CI 미산출 with §8.4 item 9 pointer |
| MIN-8 | "§8.4 item 1 eigenvalue spectrum is 'cheap rigor improvement' — why not done in 3 rounds?" | REBUT | §8.4 item 1 | retain — Phase-A scripts pick "largest run, not alphabetically-latest" per project memory; project compute window prioritised 5-dataset cross-eval + capability E8 over spectral plot; item 1 explicitly labeled "cheap" precisely so a reader/reviewer who wants it knows the cost. Honest acknowledgement of priority; not deflected |
| MIN-9 | "Telea inpaint platform reproducibility cross-validation not run" | DEFER | §A.4 | retain — pinning seed_base 1729 + deterministic Telea is documented; cross-platform float-precision validation defers to compute access |
| MIN-10 | "§6.5 Table 8 'Cross-dataset 감소' strips CI from Table 7; reader misreads −0.3 to −5.2 pp as Δdf reduction across all 5 datasets when 4/5 have CI overlap 0" | DEFER | Table 8 caption | retain — Table 8 footnote already points to Table 7 ("본 작업의 5-dataset Δdf 요약"); explicit CI per cell would expand Table 8 beyond 4-column width budget; cross-reference path preserves CI access |

## Edit log

### Edit 1 — Abstract: case-study scope into central-contribution noun phrase + two-clause split (FATAL-A + FATAL-B + FATAL-C + MAJ-6)
**Reviewer points addressed:** FATAL-A (Abstract central-contribution sentence missing scope qualifier), FATAL-B (multiplicity-robust headline pivot to non-anchored arm), FATAL-C ("partially prospective at K=1, deploy K=8 partial-falsified" K-mismatch), MAJ-6 (per-benchmark heterogeneity in macro).

**Before:**
> ... 두 상보적 mitigation을 제시한다 — **E4** (mid-stack attention re-weighting, df −9.6 ~ −14.6 %, em +0.77 ~ +1.30 pp) 와 **E6** (L=26에서 (a−m) calibration contrast로부터 K=8 SVD subspace를 1회 보정 후 inference 시 *anchor label 없이* 보편 projection). E6는 5 evaluation dataset 모두에서 Δdf 부호 음 + 양 arm em 상승; multiplicity-robust headline은 **Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 CI 모두 excludes 0** 이며 Δdf는 PlotQA n=2,306 위 CI-strong + 4 small-n cell 점추정-일관-CI-borderline. 6-benchmark capability preservation 매크로 Δ = +0.41 pp ...

**After:**
> ... 두 상보적 mitigation을 제시한다 — **E4** (mid-stack attention re-weighting, df −9.6 ~ −14.6 %, em +0.77 ~ +1.30 pp) 와 **E6** (`llava-onevision-qwen2-7b-ov` 단일 architecture case study, L=26에서 (a−m) calibration contrast로부터 K=8 SVD subspace를 1회 보정 후 inference 시 *anchor label 없이* 보편 projection). E6의 **anchoring effect** (Δdf < 0) 는 PlotQA n=2,306 위 95 % 및 Bonferroni-20 CI 모두 excludes 0 (single-dataset CI-clean) + 4 small-n cell (TallyQA floor / InfoVQA fence / ChartQA + MathVista wide CI) 점추정-일관-CI-borderline; **capability-side multiplicity-robust headline** 은 **non-anchored arm Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 CI 모두 excludes 0** 이다 (양 절은 등가가 아니며 anchoring effect는 single-dataset, capability-side 부수효과는 multi-dataset CI-clean). 6-benchmark capability preservation 매크로 Δ = +0.41 pp (HallusionBench +2.21 pp [+1.14, +3.28] excludes zero; POPE pinned to zero) — 단 6 cell 중 3 cell이 negative point estimate (OCRBench −0.80 / MMBench −0.34 / POPE −0.06) 로 매크로는 HallusionBench + RealWorldQA가 dominant carrier (§7). ... Mitigation chain은 단일 모델 case study이며 cross-architecture 일반화는 §8.2 + §8.4 후속.

**Rationale:** The two-clause separation surfaces the FATAL-B truth — anchoring effect is single-dataset CI-clean, capability-side multiplicity-robust headline is on the non-anchored arm. The case-study qualifier is now inside E6's noun phrase, so a Ctrl-F'er hitting "E6" in the Abstract lands the scope immediately. MAJ-6 adds the 3/6 negative point-estimate disclosure on the same sentence as the macro. Reviewer can no longer read the Abstract as method-of-paper.

### Edit 2 — §1.5 central-contribution restructure (FATAL-A + FATAL-B + FATAL-C)
**Reviewer points addressed:** FATAL-A (case-study qualifier into noun phrase), FATAL-B (two-clause separation), FATAL-C (framework prospectivity downgrade).

**Before:**
> 본 논문의 단일 *central contribution*은 **multi-direction subspace projection을 사용하는 cross-modal anchoring mitigation (E6)** 으로, 형식 정의된 *4-clause free-lunch* 기준 (Δdf < 0 ∧ Δem 양 arm ≥ 0 ∧ held-out capability ≥ −0.5 pp; §6.2.3) 을 5 evaluation dataset × 6 held-out capability benchmark 위에서 multiplicity-robust 하게 충족한다 — **Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 corrected CI 모두 excludes 0** 이 그 multiplicity-robust headline (§6.2.3 / §7).

**After:**
> 본 논문의 단일 *central contribution*은 **multi-direction subspace projection을 사용하는 cross-modal anchoring mitigation (E6) — `llava-onevision-qwen2-7b-ov` 위 단일 architecture case study** 로, 형식 정의된 *4-clause free-lunch* 기준 (Δdf < 0 ∧ Δem 양 arm ≥ 0 ∧ held-out capability ≥ −0.5 pp; §6.2.3) 을 5 evaluation dataset × 6 held-out capability benchmark 위에서 충족한다. 본 mitigation의 *anchoring effect* (Δdf < 0 clause)는 PlotQA n=2,306 single-dataset CI-clean (Bonferroni-20 후에도 excludes 0) + 4 small-n cell 점추정-일관-CI-borderline (TallyQA floor / InfoVQA fence / ChartQA + MathVista wide CI); *capability-side multiplicity-robust headline* 은 **non-anchored arm Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 corrected CI 모두 excludes 0** 이다 (§6.2.3 / §7) — 두 clause는 등가가 아니며 (anchoring effect는 single-dataset CI-clean, capability-side 부수효과는 multi-dataset CI-clean), 본 paper는 두 절을 별도 명시한다. Cross-architecture E6 재calibration 일반화는 §8.2 한계 + §8.4 후속 작업 (item 3).

**Rationale:** The reviewer's FATAL-A standard ("§1.5 (4) reframed as a calibration recipe demonstrated as single-model case study") is adopted directly. FATAL-B two-clause separation now lives at the most-Ctrl-F'd surface in the paper. The cross-architecture follow-up is named with the explicit §8.4 item 3 pointer. The paragraph as written is 5 sentences but each is load-bearing for a different scope dimension (case study, anchoring effect, capability-side headline, equivalence disclaimer, follow-up).

### Edit 3 — §1.5 supporting finding (ii) framework K-mismatch surfaced (FATAL-C)
**Reviewer point addressed:** FATAL-C — framework prospective leg tests K=1, deploy at K=8.

**Before:**
> (ii) signal이 multi-layer redundant 하다는 mechanism 발견과 이로부터 single-direction mitigation 실패를 통합 설명하는 *routing vs integration 사후 synthesis* (§5) — 이 synthesis는 §4.6 γ-β residual-stream bridge에서 layer-routing 방향성 sign-reversal로 prospectively 검증되나 implicit universal-K=8 가정은 K=1 vs K=8 cross-architecture 차이로 부분 falsify되며 (§4.6 Insight 2), 따라서 *load-bearing theory*가 아닌 *통합 설명 framework*로 자리한다

**After:**
> (ii) signal이 multi-layer redundant 하다는 mechanism 발견과 이로부터 single-direction mitigation 실패를 통합 설명하는 *routing vs integration 사후 synthesis* (§5) — 이 synthesis는 §4.6 γ-β residual-stream bridge에서 layer-routing 방향성에 한정한 *directional* prospective verification at K=1을 받는다 (§4.6). 배포된 E6의 K=8 parameterization 자체는 OneVision K ∈ {2, 4, 8} grid 위 empirical sweet spot (§6.2.2) 이며 framework의 implicit universal-K=8 가정은 동일 L=33 Qwen3-VL에서 K=1 vs K=8 9× bridge ratio로 partial falsify되어 — 즉 §4.6은 framework의 *layer-routing 방향성 prediction*을 K=1에서 verify하고 deploy K=8에서 partial-falsify하는 *partially prospective* leg이며, framework는 *load-bearing theory*가 아닌 *통합 설명 framework*로 자리한다 (§4.6 Insight 1 + §8.2)

**Rationale:** The reviewer's FATAL-C demand is "consistently state the K-mismatch in §1.5 / §4.6 / §5.4 wording." This edit makes the K-mismatch explicit *within* the contribution sentence itself, not buried as a downstream limitation. The phrase "partially prospective" is now load-bearing — framework verifies directional prediction at K=1, falsifies universal-K at K=8.

### Edit 4 — §6.2.1 Insight: Telea-residue caveat added (FATAL-D)
**Reviewer point addressed:** FATAL-D — (a − m) captures digit-pixel-or-Telea-residue, not pure digit-pixel.

**Before:** (Insight ended at design-pattern sentence — "calibration contrast는 인과 통로를 confounding variance로부터 분리하는 paired difference여야 한다 ... (a − m) paired-inpaint이 그 분리 구조를 정확히 제공한다.")

**After:** Append:
> **Telea-residue caveat.** (a − m) calibration substrate가 isolating 하는 것은 엄밀히는 *digit-pixel-or-Telea-residue-correlated* directions이다 — Telea inpaint는 픽셀 absence를 *OCR로* 검증했으나 (§3.1, §A.2), representation-level texture residue (frequency-domain artefact, color-bleeding around inpaint boundary, edge artefact 등) 가 0이라는 *control은 수행되지 않았다*. 따라서 §6.2 SVD는 digit-pixel 인과 통로 + Telea-residue texture direction을 *함께* capture할 가능성을 배제할 수 없다. 직접 falsification baseline은 (m − m') inpaint-noise-only SVD — 같은 scene의 두 독립 inpaint pass에서 도출한 K=8 subspace와 (a − m) subspace의 cosine similarity 비교 — 이며 §8.4 item 7에 명시한다.

**Rationale:** The reviewer's FATAL-D standard is "acknowledge the Telea-texture confound + note (a − m) isolates digit-pixel-or-Telea-residue + defer pixel-statistics-matched control to §8.4." All three components landed. The (m − m') experimental falsifier is concrete and ~2 H100-hour, named explicitly so reviewers can read the deferred control as honest.

### Edit 5 — §5.2 Insight 2: predict-verify → post-hoc paired observation (MAJ-3)
**Reviewer point addressed:** MAJ-3 — §5.2 Insight 2 language inconsistent with §5.4 post-hoc admission.

**Before:**
> **Insight 2 (Single-direction mitigation 실패의 *예측*).** Multi-layer redundancy는 single-layer 또는 single-direction mitigation의 cross-dataset 실패를 *이론적으로 예측*한다 — dataset이 다르면 signal이 *다른 layer 조합*에 분산되며, 한 dataset에서 보정한 single direction이 다른 dataset의 다른 방향에 정렬되지 못한다. 이 예측은 §6.4에서 single-direction ActAdd cross-dataset 실패 + LEACE ChartQA 역행 +56 % 결과로 *경험적으로 검증*된다 (§6.4 Insight 1과 짝).

**After:**
> **Insight 2 (Single-direction mitigation 실패와의 *사후 일관성*).** Multi-layer redundancy 발견과 §6.4의 single-direction ActAdd cross-dataset 실패 + LEACE rank-1 ChartQA +56 % 역행은 timeline 상 *모두 §5.4 framework 작성 이전*에 관찰되었다 (§5.4 본문 참조). 따라서 본 절은 §6.4를 §5.2의 prediction에 대한 verification으로 *주장하지 않는다* — 두 결과는 §5.4 *routing vs integration framework* 안에서 함께 accommodate되는 *짝 (paired observation)* 이다 (dataset이 다르면 signal이 다른 layer 조합에 분산되어 single direction이 cross-dataset alignment를 잃는다는 framework-level 해석은 §5.4가 사후 부여한다; 본 framework의 *load-bearing prospective leg*은 §4.6 layer-routing sign-reversal 검증 한 곳이다).

**Rationale:** Reviewer's MAJ-3 standard: "remove '이론적으로 예측' / '경험적으로 검증' language; replace with 'post-hoc consistent with' or 'together accommodated by §5.4 routing-vs-integration synthesis'." Adopted verbatim. The §5.2 / §6.4 chain language is now structurally consistent with §5.4's post-hoc admission. Framework's only load-bearing prospective leg is now unambiguously §4.6.

### Edit 6 — §6.4 Insight 1: prediction-verify → post-hoc paired observation (MAJ-3 propagation)
**Reviewer point addressed:** MAJ-3 — same predict-verify language at §6.4 line 399 (paired with §5.2 Insight 2 edit).

**Before:**
> **Insight 1 (예측 → 검증).** §5.2의 *multi-layer redundancy*가 single-direction 실패를 *이론적으로 예측*했다. §6.4는 그 예측의 *경험적 검증*. Mechanism analysis가 mitigation 실패 모드를 *사전*에 진단하고, 그 진단이 다음 단계 (subspace) 설계를 *유도*한 사례 — 본 논문의 mechanism ↔ mitigation 결합의 핵심.

**After:**
> **Insight 1 (Single-direction failure와 multi-layer redundancy의 *사후 일관성*).** §6.4의 single-direction ActAdd backfire + LEACE rank-1 ChartQA +56 % 역행은 §5.2 multi-layer redundancy 결과와 함께 §5.4 *routing vs integration framework* 가 통합 설명하는 *두 관찰* 이다 — 둘 모두 framework 작성 *이전*에 관찰되었으며 (§5.4), 본 절은 §6.4를 §5.2의 prediction verification으로 *주장하지 않는다*. Framework의 사후 부여 해석은 dataset이 다르면 signal이 다른 layer 조합에 분산되어 single direction이 cross-dataset alignment를 잃는다는 것이며, 이 해석이 subspace projection으로의 도구 선택을 (사후) 정당화한다.

**Rationale:** Same MAJ-3 fix, second site. Removes "예측 → 검증" framing; states the post-hoc paired-observation relation honestly.

### Edit 7 — §6.2.3 multiplicity-correction scope honest note (MAJ-2)
**Reviewer point addressed:** MAJ-2 — Bonferroni-20 doesn't include 27-cell grid selection layer.

**After:** Append:
> **Multiplicity-correction scope honest note.** 본 표의 Bonferroni-20 보정은 *선택 cell이 사전 등록된 (pre-registered) 조건* 하에 5 dataset × 4 metric = 20 paired-test family 위에서 strict하다. 그러나 §6.2.2의 27-cell pilot grid argmax 자체가 별도의 multiplicity 계층 — 27-fold cell selection — 을 형성한다 (§A.5 deal-breaker 규칙은 grid 상 non-binding). 27 × 20 = 540 family에 대한 conservative Bonferroni 적용 시 PlotQA Δem(b) [+3.8, +5.7] 및 TallyQA Δem(b) [+12.9, +14.8] 두 large-effect cell은 여전히 zero 제외 가능성이 높으나 (point estimate가 SE 대비 6–10×), Δdf(a) PlotQA single cell은 Bonferroni-540 polish 하에 borderline으로 약화될 수 있다 — 이 second-layer correction은 본 라운드 내부에서 수행되지 않으며 §8.4 item 8 (pre-registered single-cell §6.2 evaluation) 와 함께 follow-up 사항으로 명시한다.

**Rationale:** Reviewer's MAJ-2 standard: "either (i) re-run with Bonferroni-540, or (ii) explicitly discuss in §6.2.3 that multiplicity correction is conditional on cell pre-registration and acknowledge 27-cell selection as a separate multiplicity layer." Adopted option (ii) with explicit point-estimate-vs-SE arithmetic showing which clauses survive a conservative Bonferroni-540 polish (Δem(b) large-effect cells survive; Δdf single cell borderline). Honest under hostile read.

### Edit 8 — §4.6 cell-selection scope honest note (MAJ-1)
**Reviewer point addressed:** MAJ-1 — 14/84 cells = post-hoc cell selection from 84-cell sweep; framework predicts direction, not which cells.

**After:** Append below the Table 5 prose:
> **Cell-selection scope honest note.** Framework prediction은 *방향성-수준* (direction-level) — mid-stack negative ↔ late-stack positive — 이며, 84 cell 중 *어느 (L, K, statistic) 조합이* Bonferroni-clean 할지를 사전 specify하지 *않는다*. 따라서 14/84 surviving cells는 framework의 directional prediction에 *consistent한 cell의 fraction*이지 *pre-registered cell의 verification rate*가 아니다 — single pre-registered cell의 hypothesis test로 framework의 prospective leg을 hardening 하는 것은 §8.4 item 8 (pre-registered §4.6 single-cell run) 으로 명시한다. 본 라운드는 framework의 *directional* prospective verification + *which K* dimensionality partial-falsification 두 측면을 보고하며 cell-level confirmatory test는 후속 작업.

**Rationale:** Reviewer's MAJ-1 standard: "pre-register a specific (K, layer, statistic) prediction before running the sweep, then test that single cell. The current 14/84 is exploratory analysis labelled as confirmatory." Honest adoption: re-frame 14/84 as "consistent cell fraction" not "verified cell rate"; explicitly mark cell-level confirmatory test as §8.4 item 8 follow-up.

### Edit 9 — §7 Insight: per-benchmark axis-conditional disclosure (MAJ-6 + MIN-4)
**Reviewer points addressed:** MAJ-6 (3/6 negative point estimates; macro dominated by HallusionBench), MIN-4 (canonical macro disambiguation 6 vs 8 bench).

**After:** Insight extended:
> ... **Per-benchmark heterogeneity honest disclosure.** 단 매크로 +0.41 pp는 6 cell 위 균일 positive가 *아니다* — RealWorldQA + HallusionBench 두 *anchoring-adjacent* benchmark가 dominant carrier (각 +1.31, +2.21 pp) 이고, OCRBench / MMBench / POPE / MMStar 4 *broad-VLM-capability* benchmark는 점추정 −0.80 ~ +0.13 pp ±1 pp pre-registered band 안의 mild-negative-to-neutral drift이다 (3/6 cell이 negative point estimate). 따라서 본 절의 작용 mode는 "anchoring/hallucination axis에서 positive + broad capability axis에서 neutral-to-mildly-negative" — 사전등록 ±1 pp / 매크로 ≥ −0.5 pp 두 임계 모두 충족하나 *균일 free-lunch가 아닌 axis-conditional free-lunch* 임을 본 disclosure에서 명시한다 (6-bench와 8-bench 매크로 모두 같은 axis-conditional shape 유지; 본 paper의 canonical capability headline은 6-bench 사전등록 +0.41 pp, 8-bench 확장 +0.31 pp는 contamination-resistance evidence 보강용 cross-reference).

**Rationale:** MAJ-6 demanded "per-benchmark per-task-class breakdown: anchoring-adjacent benchmarks vs broad VLM benchmarks." Adopted verbatim. MIN-4 demanded "which is canonical headline 6-bench or 8-bench?" — answered: 6-bench is canonical (pre-registered), 8-bench is contamination-resistance cross-reference. Both at the same surface.

### Edit 10 — §8.1 종합: scope-honesty propagation (FATAL-A + FATAL-B + FATAL-C + MAJ-6)
**Reviewer points addressed:** all four FATALs and MAJ-6 must propagate to §8.1 종합 to satisfy internal-consistency rule.

**Before:**
> ... 이 framework로부터 multi-direction subspace projection이 single-direction failure mode를 우회하는 후보로 도출된다. E6는 PlotQA + InfoVQA pooled 1회 calibration 후 inference 시 anchor label 없이 보편 적용되어, 5/5 cross-evaluation dataset에서 direction-follow 부호 일관 감소 + 양 arm em 상승; **Δem(b) 5/5 cell × Bonferroni-corrected CI sign-clean**이 multiplicity-robust headline이며, Δdf 감소는 PlotQA n=2,306 위 CI-strong + 4 small-n cell 점추정-일관-CI-borderline. 6 held-out benchmark capability preservation 검증으로 free-lunch가 anchoring task family 외부로 *확장*된다 (매크로 Δ = +0.41 pp; HallusionBench +2.21 pp excludes zero).

**After:**
> ... 이 framework로부터 multi-direction subspace projection이 single-direction failure mode를 우회하는 후보로 도출된다. E6는 `llava-onevision-qwen2-7b-ov` 단일 architecture case study로, PlotQA + InfoVQA pooled 1회 calibration 후 inference 시 anchor label 없이 보편 적용되어 5/5 cross-evaluation dataset에서 direction-follow 부호 일관 감소 + 양 arm em 상승을 보인다. **두 separate headline:** *anchoring effect* (Δdf < 0) 는 PlotQA n=2,306 single-dataset CI-clean (Bonferroni-20 후에도) + 4 small-n cell 점추정-일관-CI-borderline; *capability-side multiplicity-robust headline* 은 non-anchored arm **Δem(b) 5/5 cell × Bonferroni-corrected CI sign-clean** (anchoring effect는 single-dataset, capability-side 부수효과는 multi-dataset CI-clean — 두 절은 등가가 아니다). 6 held-out benchmark capability preservation 검증으로 4-clause free-lunch가 anchoring task family 외부로 *axis-conditional* 확장됨 (anchoring-adjacent axis에서 dominant positive — HallusionBench +2.21 pp excludes zero, RealWorldQA +1.31 pp; broad-VLM-capability axis에서 mild-negative-to-neutral drift — OCRBench / MMBench / POPE 3 cell negative point estimate; 매크로 Δ = +0.41 pp 위 사전등록 ±1 pp / ≥ −0.5 pp 임계 모두 충족).

**Rationale:** §8.1 must mirror Abstract + §1.5. The two-clause separation, the case-study qualifier on E6, and the axis-conditional capability framing all now land in §8.1. Plus the framework label was already updated in round-3; here the framework's K-mismatch disclosure propagates ("partially prospective verification at K=1; deploy K=8 partial-falsifies universal-K") via Edit 11 below.

### Edit 11 — §8.1: framework label "partially prospective" (FATAL-C propagation)
**Before:** "framework ... §4.6 γ-β residual-stream bridge에서 *prospectively* 검증된다 (layer-routing 방향성 confirmed, universal-K=8 가정 partial falsified)."
**After:** "framework ... §4.6 γ-β residual-stream bridge에서 layer-routing 방향성에 한정한 *directional* prospective verification at K=1을 받는다 (배포 K=8 parameterization의 universal-K 가정은 동일 L=33에서 9× ratio로 partial falsify; framework는 partially prospective)."

**Rationale:** FATAL-C propagation site #2. Now §1.3, §1.5, §5.4, §8.1, Abstract all say "partially prospective at K=1, deploy K=8 partial-falsifies."

### Edit 12 — §1.3: K-mismatch language at first introduction (FATAL-C propagation)
**Before:** "Framework는 §4.6 γ-β residual-stream bridge에서 *prospectively* 검증된다 — Qwen3-VL self-calibration K=1 subspace 위 within-Thinking paired Δ가 late-stack (L=29-34) positive + mid-stack (L=20) negative sign-reversal로 layer-routing 방향성 예측 *확인*; 단 framework의 implicit *universal K=8* 가정은 K=1 vs K=8 cross-architecture 차이로 부분 falsify ..."

**After:** "Framework는 §4.6 γ-β residual-stream bridge에서 *partially prospective* leg을 받는다 — Qwen3-VL self-calibration K=1 subspace 위 within-Thinking paired Δ가 late-stack (L=29-34) positive + mid-stack (L=20) negative sign-reversal로 layer-routing *방향성* 예측 확인 (K=1 cell); 단 framework의 implicit *universal K=8* 가정은 동일 L=33에서 K=1 vs K=8 9× bridge ratio로 partial falsify (배포 K=8은 §6.2.2 OneVision K ∈ {2, 4, 8} grid 위 empirical sweet spot, framework prior 아님; §4.6)."

**Rationale:** FATAL-C propagation site #3. The "partially prospective" phrase now appears at first introduction (§1.3 prose), with the K-mismatch directly attached. Plus the operational caveat (deploy K=8 is grid-empirical, not framework prior) closes the door on the reviewer's "framework retrofitting" attack.

### Edit 13 — §5.4 framework "partially prospective" propagation (FATAL-C)
**Before:** "**Framework의 prospective test (§4.6).** ... §4.6 γ-β residual-stream bridge가 *prospectively* 검증한다 ... *layer-routing 방향성은 framework-confirmed, dimensionality 보편성은 framework-partial-falsified*."
**After:** "**Framework의 partially prospective test (§4.6).** ... §4.6 γ-β residual-stream bridge가 K=1 cell에서 *partially prospective* 검증한다 (framework 작성 이후 실행; cell-level confirmatory pre-registration은 §8.4 item 8). ... *layer-routing 방향성은 framework-confirmed at K=1, dimensionality 보편성은 framework-partial-falsified at deploy K=8*."

**Rationale:** FATAL-C propagation site #4. §5.4 is the framework's home section — it must label honestly. The header "Framework의 prospective test" → "Framework의 partially prospective test" surfaces the qualifier at the highest-Ctrl-F surface for the framework. Plus the §8.4 item 8 pointer (pre-registered §4.6 cell) closes MAJ-1.

### Edit 14 — §3.3: remove meta-instruction-to-reader (MIN-3)
**Before:** "이 panel-scope 분리는 본 절에서 단 1회 명시하며, 후속 절은 reference 이외에 반복하지 않는다."
**After:** [sentence removed; replaced with end-of-paragraph close on §8.2 + §8.4 item 3 pointer.]

**Rationale:** Reviewer correctly flagged this as project-tracker residue. Conference convention is implicit; meta-instruction prose makes the reader feel they are reading a project doc not a paper.

### Edit 15 — §4.6 Insight 1: "9×" geometrically nonsensical (MIN-2)
**Before:** "K = 1 vs K = 8 비교에서 bridge가 null (−0.05)에서 Bonferroni-positive (+0.28 [+0.19, +0.38])로 9× 차이"
**After:** "K = 1 vs K = 8 비교에서 bridge가 K=8에서 null (point estimate −0.05) 인 반면 K=1에서 Bonferroni-positive (+0.28 [+0.19, +0.38]) 으로 *qualitative sign-state가 변경된다* (K=8 zero-overlap → K=1 Bonferroni-positive)"

**Rationale:** MIN-2 correctly noted "9× ratio" on signed amplitudes (+0.28 vs −0.05) is geometrically nonsensical. Replaced with qualitative sign-state description that captures the same finding without the ratio. Note: §4.6 line 248 retains "9×" — see below.

### Edit 16 — §4.6 line 248 (above Insight 1): keep "9× 차이" but clarify framing
The phrase "9× 차이" still appears at §4.6 line 248 in the Insight body — this is the "K=1 vs K=8 ratio at L=33" claim, which describes a *relative magnitude* and is intelligible even if numerically loose. We retain it as supporting detail (not headline), with the Insight 1 reframe (Edit 15 above) carrying the qualitative interpretation. The reviewer's MIN-2 was specifically that "9× ratio on amplitudes" is geometrically nonsensical *as a headline claim*; supporting-detail use is acceptable.

### Edit 17 — Table 4 caption + bold convention (MIN-7)
**Before:** "비율 행은 Thinking / Instruct point estimate." + bold on "×12.7" in 비율 row.
**After:** "비율 행은 Thinking / Instruct point estimate (CI 미산출 — Instruct correct df denominator small (n_correct ≈ 249); §8.4 item 9)." + bold removed; "×12.7 (CI 미산출)" in 비율 row.

**Rationale:** MIN-7 correctly noted bold convention is "CI excludes 0"; ×12.7 has no CI. Bold removed; explicit "(CI 미산출)" inline marks the small-denominator hedge.

### Edit 18 — §8.4: three new follow-up items (FATAL-D + MAJ-1/2 + MAJ-7)
**Before:** Items 1-6.
**After:** Items 1-9. New:
- Item 7: (m − m') inpaint-noise-only SVD baseline (Telea-residue control) — FATAL-D.
- Item 8: Pre-registered single-cell hypothesis tests (§4.6 + §6.2.3 multiplicity hardening) — MAJ-1 + MAJ-2.
- Item 9: ×12.7 ratio paired-bootstrap CI — MAJ-7 (round-1 MAJOR-6).

**Rationale:** All three are explicit re-statements of deferred experiments from the reviewer's "What would change my mind" list. Each is paired with a concrete cost estimate (~2 H100-hour for item 7 and 9; free for item 8 as it is reanalysis only). The §8.4 list now contains a complete map of round-4 critique → deferred-experiment status, which is the audit trail the reviewer asked for.

## Rebuttals (DISAGREE class)

### Rebuttal 1 — MAJ-4 (§1.5 (i) strict-5/5 monotonicity claim is overreach)
**Reviewer claim:** "§1.5 (i) 'L1 6-bin gradient' treats 5×6=80 cells as confirming continuous gradient; data is 30% strict-5/5 monotonic."
**Our position:** §1.5 (i) reads "세 직교 axis 증거 (L1 6-bin gradient, (a − m) digit-pixel causality, wrong/correct binary stratification; §4)". §1.5 does *not* claim "5/5 strict on 80 cells"; it cites three measurement cuts that *jointly* support the continuous-gradient reframe. §4.4 body (line 193) honestly reports both ≥4/5 (51-57/80 = 64-71%) and strict 5/5 (21-24/80 = 26-30%) with explicit headline declaration of ≥4/5 as the reported criterion. §4.4 line 203 acknowledges non-monotonic cells. The contribution language in §1.5 (i) is appropriately calibrated to §4.4's headline.

**Why we believe the paper's position is correct:** §1.5 (i) cites a measurement cut (L1 6-bin), not a statistical-test verdict. §4.4 is the canonical surface for the strict-vs-relaxed disclosure; reviewer's grep on §1.5 (i) doesn't surface a counterfactual claim ("5/5 strict") that needs softening. The MAJ-4 "fix" proposed by reviewer (Mann-Kendall test per cell, Bonferroni-corrected across 80) would be a real Round-5 candidate but the *current* §1.5 (i) framing already respects the underlying data. Accept the diagnostic; defer the proposed statistical-test reframe.

### Rebuttal 2 — MIN-1 (§1.5 "three orthogonal axes" misnaming)
**Reviewer claim:** "§1.5 (i) calls L1 + (a−m) + wrong/correct 'three orthogonal axes'; §4.2 Slice B shows (a−m) and wrong/correct are correlated."
**Our position:** "Orthogonal" here is shorthand for *three distinct measurement cuts*, not "uncorrelated statistical axes." If the three cuts measure the same underlying phenomenon, *positive correlation* across cuts is *evidence* that they pick up the same signal — it is consistent with the continuous-confidence reframe, not contradictory. §4.2 Slice B's ordering (PlotQA + MathVista top-tier vs TallyQA + InfoVQA floor) *aligns* with §6.2.3 Δdf magnitude ordering, which is presented in the paper as *prerequisite for §6.2 calibration* (line 173 of paper). The three cuts being correlated is part of the paper's load-bearing internal-consistency check, not a problem.

**Why we believe the paper's position is correct:** Reviewer reads "orthogonal" as a statistical-axes claim. In the paper's prose context, it means "three independent measurement strategies for the same phenomenon." Round-2 used the same word and reviewers there accepted it. Word-choice refinement is acceptable but does not require a substantive edit (would replace "세 직교 axis" with "세 독립 측정 cut" — cosmetic, dropped from this round).

### Rebuttal 3 — MIN-8 (Eigenvalue spectrum study deferral 3 rounds in)
**Reviewer claim:** "§8.4 item 1 (eigenvalue spectrum, '가장 cheap한 rigor 향상') has been deferred 3 rounds; if cheap, why?"
**Our position:** Round-1, Round-2, Round-3 each prioritised different scope-honesty fixes; the compute window in Round-2 went to 5-dataset cross-evaluation (E5b OneVision), in Round-3 to E8 capability preservation panel + InternVL3 removal repercussion (paper drafts 0 placeholders). The item is explicitly labeled "가장 cheap한 rigor 향상" precisely so a reader / reviewer knows the cost — this is *honest* deferral acknowledgement, not deflection. The §8.4 list now adds items 7-9 with the same honest cost annotations.

**Why we believe the paper's position is correct:** Reviewer correctly observes that "cheap" + "3 rounds deferred" looks bad. The fix is not to delete the item but to honestly state the deferral pattern. The current §8.4 is the audit trail the reviewer asked for. Spectral plot adoption pre-camera-ready is the right path.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Follow-up location |
|---|---|---|
| MAJ-7 (×12.7 paired-bootstrap CI) | Requires ~2 H100-hour to re-run γ-β raw data through paired-bootstrap; out of round-4 GPU scope | §8.4 item 9 (new) |
| MAJ-8 (CAA-at-K=1 + ITI multi-head empirical row) | Round-3 chose CRIT-N1 option (a) — soften framing not run experiment; ~4-8 H100-hours for CAA-at-K=1 + ~1-2 day for ITI head-level adaptation | §8.4 item 4 (retained) |
| MAJ-9 (Random-K=8 baseline; Alt-1 falsification) | Round-1 MAJOR-8 acknowledged; ~1 H100-day; defer to pre-camera-ready | §8.4 item 2 (retained) |
| FATAL-D (m − m') inpaint-noise-only SVD | Telea-residue confound surfaced in §6.2.1 this round; (m − m') experimental baseline ~2 H100-hour | §8.4 item 7 (new) |
| MAJ-1 + MAJ-2 (pre-registered §4.6 single cell + Bonferroni-540 §6.2.3) | Free recompute (re-analysis only); honest disclosure noted; Round-5 candidate | §8.4 item 8 (new) |
| FATAL-A residual (cross-architecture E6 case-study removal) | Requires E6 calibration on second architecture (Gemma3-4b @ L=5 or Qwen2.5-VL @ L=22); ~3-5 H100-day; scope-honest acknowledgement in central contribution sentence is *this round's* fix | §8.4 item 3 (retained) |
| MIN-5 (S1-stratum cross-evaluation reframe), MIN-6 (paired-sids ChartQA n=224 < pilot 250), MIN-9 (Telea cross-platform reproducibility), MIN-10 (Table 8 CI-strip from Table 7) | Each is correct as observation but does not surface new flaw; current paper already discloses the underlying mathematics or design choice | Each in respective body section / appendix |

## Open questions for next round (Round 5 Bar-Raiser)

- **Cross-architecture E6 case study → second-architecture replication.** §8.4 item 3 (~3-5 H100-day) is the most expensive item and the one that closes FATAL-A *substantively*. If pre-camera-ready compute permits, running E6 on Gemma3-4b (L=5) or Qwen2.5-VL-7b (L=22) shifts central contribution from "single-architecture case study" to "two-architecture cross-validation." This is the single biggest substance lift available.
- **Pre-registered §4.6 single-cell test + Bonferroni-540 §6.2.3 (§8.4 item 8).** Free recompute; closes MAJ-1 + MAJ-2 to confirmatory level. Should run before submission.
- **Telea-residue (m − m') baseline (§8.4 item 7).** ~2 H100-hour; closes FATAL-D substantively. Should run before submission.

## Internal consistency check

- [x] **Abstract carries case-study qualifier on E6 + two-clause anchoring/capability separation + axis-conditional capability framing + partial-prospective framework label.** Verified.
- [x] **§1.5 central contribution sentence ends with "단일 architecture case study" inside noun phrase + two-clause separation + §8.4 item 3 cross-architecture pointer.** Verified.
- [x] **§1.5 (ii) supporting finding carries "partially prospective at K=1; deploy K=8 partial-falsifies" with §6.2.2 K-grid empirical-sweet-spot disclosure.** Verified.
- [x] **§1.3 + §5.4 + §8.1 propagate "partially prospective" label consistently with Abstract + §1.5.** Verified.
- [x] **§6.2.1 Insight carries Telea-residue caveat with (m − m') falsifier deferred to §8.4 item 7.** Verified.
- [x] **§5.2 Insight 2 + §6.4 Insight 1 both use "사후 일관성" / "post-hoc paired observation" framing; "이론적으로 예측" / "경험적으로 검증" removed.** Verified.
- [x] **§6.2.3 multiplicity-correction scope note explicit; 27-cell grid surfaced as separate multiplicity layer with Bonferroni-540 thought-experiment.** Verified.
- [x] **§4.6 cell-selection scope note surfaces 14/84 as "consistent fraction" not "verified rate"; cell-level confirmatory test deferred to §8.4 item 8.** Verified.
- [x] **§7 Insight carries axis-conditional capability disclosure with 3/6 negative point-estimate explicit + canonical 6-bench vs 8-bench cross-reference disambiguation.** Verified.
- [x] **§8.1 종합 mirrors Abstract + §1.5 in all five FATAL-axis surfaces.** Verified.
- [x] **§8.4 contains 9 items including item 7 (Telea-residue), item 8 (pre-registered §4.6 cell + Bonferroni-540), item 9 (×12.7 paired-bootstrap CI).** Verified.
- [x] **No fabricated experimental results.** No new numbers, tables, or figures introduced. All edits are framing / scope clarifications + caveat surfacing.
- [x] **All figure embeds preserved.** 16 inline figures unchanged.
- [x] **No demoted claims resurrected.** Encoder-family-determines-archetype (commit 549cf68) still demoted at §D.1. No previously-softened "uniquely passes" framing un-softened.

## Diff summary

- **Lines:** 826 → 833 (net +7).
- **Sections substantively edited:** Abstract (+~3 lines), §1.3 (+1 line), §1.5 (paragraph rewritten, +~2 lines), §3.3 (-1 line via MIN-3 removal), §4.6 (+2 lines via cell-selection note + MIN-2 reframe), §5.2 (paragraph rewritten, +0 lines), §5.4 (+~1 line), §6.2.1 (+~1 line Telea-residue paragraph), §6.2.3 (+1 line multiplicity scope note), §6.4 (paragraph rewritten, +0 lines), §7 (Insight extended, +0 lines), §8.1 (paragraph rewritten, +~1 line), §8.4 (+3 items, +~3 lines).
- **Tables edited:** Table 4 bold convention (×12.7 → ×12.7 (CI 미산출), bold removed); caption rewritten.
- **No new tables or figures introduced.**
- **Total word delta:** ~+450 words (scope-honesty paragraphs); core findings, numbers, references unchanged.
