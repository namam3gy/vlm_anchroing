# Round 3 — Author Response to Novelty / Positioning / Related-Work Review

**Paper version BEFORE:** v10 working tree (828 lines, post-Round-2 writing surgery, 2026-05-11 PM).
**Paper version AFTER:** v11 working tree (826 lines — net −2 lines despite §6.5 expansion + §1.5 restructure).
**Date:** 2026-05-11 evening.
**Reviewer round addressed:** `docs/paper/reviews/round3_novelty.md`.

## Summary

The reviewer's verdict (borderline-accept-for-EMNLP-Main contingent on three CRITs) hinged on a single structural diagnosis — that residual experiment-log feel had migrated from the writing axis (cleaned in Round 2) to the *contribution-framing axis* — and three positioning issues that any informed reviewer would catch. We accept all three CRITs in full. The MAJ-N1 single-central-contribution restructure is adopted (§1.5 collapsed from four parallel contributions to one central + three supporting + one auxiliary, near-verbatim using the reviewer's drafted Korean paragraph at line 176). This restructure subsumes MAJ-N5 (§1.5(3)+(4) merge) and propagates CRIT-N3 (relabel "이론적 기여" → "통합 설명 framework") naturally. CRIT-N2 (×12.7 contamination) is addressed by removing the ×12.7 ratio from the Abstract auxiliary sentence and keeping it in §1.4 / §4.5 / §8.2 only — option (a) per the task brief. CRIT-N1 (§6.5 "uniquely passes" framing) is softened with explicit panel-scope qualifier + ITI multi-head trap acknowledged + tuning-effort asymmetry surfaced (MAJ-N6). MAJ-N2 + MAJ-N4 + MIN-N3 + MIN-N4 + MIN-N6 receive targeted edits. MAJ-N7 / MAJ-N8 / MIN-N5 / MIN-N7 deferred or absorbed. Net body change: 828 → 826 lines.

## Decision summary table

| # | Reviewer point (verbatim) | Class | Section affected | Status |
|---|---|---|---|---|
| CRIT-N1 | "§6.5 Table 8 'uniquely passes 4-clause free-lunch' framing rests on a 5-row baseline panel that *structurally excludes* the two closest prior-method classes (CAA and ITI)" | EDIT | §6.5 sub-title, Table 8 verdict cell + caption, Insight Note 1 + Note 2 | done |
| CRIT-N2 | "Abstract's `correct-base df ratio ×12.7 point estimate` is not load-bearing for the four headline contributions but ... the hedge is *adjacent* but not *bound* to the number" | EDIT | Abstract auxiliary sentence | done — option (a) |
| CRIT-N3 | "§1.5 (3) 'routing vs integration synthesis' is labeled as the paper's *이론적 기여* but ... contribution is now closer to *useful framework* than *load-bearing theory*" | EDIT | §1.3, §1.5 (absorbed by MAJ-N1), §8.1 | done — relabel "이론적 기여" → "통합 설명 framework (사후 synthesis + §4.6 부분 prospective 검증)" |
| MAJ-N1 | "§1.5 four parallel contributions read as experiment catalog ... single-central-contribution restructure" | EDIT (substantive restructure) | §1.5 | done — one central + three supporting + one auxiliary, reviewer's drafted paragraph used as base |
| MAJ-N2 | "Chand et al. cross-axis positioning should move from §6.2.3 to §2 paragraph 4 for prominence" | EDIT | §2 paragraph 4 last sentence | done — Chand cross-axis sentence appended |
| MAJ-N3 | "§1.5 (1) over-anchors on first-ness ... independent + open-numeric load-bearing, rendered-digit not load-bearing" | PARTIAL EDIT | §1.5 | absorbed by MAJ-N1 — 평가 protocol now ends §1.5 paragraph as "complementary substrate" not "first-ness" claim |
| MAJ-N4 | "(a − m) digit-pixel paired-inpaint contrast is paper's strongest methodological move ... §1.5 (1) and §2 do not foreground enough" | EDIT | §1.5, §2 paragraph 4 | done — (a − m) elevated to supporting finding (iii) in §1.5; §2 paragraph 4 adds design-pattern sentence |
| MAJ-N5 | "§1.5 (3) routing-vs-integration synthesis overlaps with (4) multi-direction subspace mitigation in load-bearing function" | EDIT | §1.5 | absorbed by MAJ-N1 — synthesis now supporting finding (ii), mitigation is central |
| MAJ-N6 | "§6.5 baseline tuning effort asymmetric vs E6's 27-cell grid; what would change with fair-tuning" | EDIT | §6.5 Insight Note 2, Table 8 caption | done |
| MAJ-N7 | "γ-β plays dual role as auxiliary observation in §4.5 *and* prospective test in §4.6 — clarify" | EDIT | §1.5 auxiliary | done — auxiliary sentence now distinguishes "§4.5 behavioural / §4.6 residual-stream bridge as framework's prospective leg" |
| MAJ-N8 | "Dyslexify + E6 composition test — composable / interfere / substitute prediction" | DEFER | §8.4 | not added — would be 7th item; positioning opportunity not load-bearing for this round |
| MIN-N1 | "§8.2 line 470 still uses unmodified 'free-lunch'" | REBUT | §8.2 | rebut below — current §8.2 has only "E4 free-lunch" shorthand which Round 2 explicitly accepted, no unprefixed criterion-policy claim survives |
| MIN-N2 | "§1.5 (2) reorder so contribution noun = continuous-confidence reframe" | EDIT | §1.5 | absorbed by MAJ-N1 — supporting finding (i) leads with "anchoring을 wrong/correct binary projection이 아닌 continuous confidence gradient로 재해석" |
| MIN-N3 | "§2 paragraph 1 cites Wang 2025a as 4-class; should be 5-class" | EDIT | §4.5 Insight 2 (where the 4-class language actually lives) | done — fixed in §4.5; §2 paragraph 1 already says generic "judging bias가 reasoning trace를 통해 증폭" |
| MIN-N4 | "§2 paragraph 4 ITI sentence doesn't surface ITI was only evaluated on TruthfulQA" | EDIT | §2 paragraph 4 | done — TruthfulQA-only caveat appended in parenthetical |
| MIN-N5 | "§2 has no paragraph on visual representation steering / image-token attention pruning" | PARTIAL EDIT | §2 paragraph 4 | done — single-sentence positioning at end of paragraph 4 (vision-token pruning class) |
| MIN-N6 | "§4.5 Insight 2 line 233 'first-evidence' should match Round-2 hedging consistency" | EDIT | §4.5 Insight 2, §8.2 γ-β bullet | done — both sites now say "N=1 architecture × N=1 dataset existence proof" |
| MIN-N7 | "Goh et al. 2021 venue (Distill) inline" | DEFER | §2 paragraph 3 | not addressed — cosmetic; reference list already shows venue |

## Edit log

### Edit 1 — §1.5 single-central-contribution restructure (MAJ-N1 + MAJ-N3 + MAJ-N4 + MAJ-N5 + MAJ-N7 + MIN-N2)
**Reviewer point addressed:** MAJ-N1 (the highest-leverage edit). Subsumes MAJ-N3, MAJ-N4, MAJ-N5, MAJ-N7, MIN-N2.
**Reviewer reasoning:** Four parallel contributions of roughly equal weight = experiment-catalog fingerprint. Reviewer drafted a Korean paragraph (review line 176) collapsing to one central (E6 mitigation) + three supporting findings (cross-dataset evidence, mechanism synthesis, (a − m) substrate) + one auxiliary (γ-β). Adopting this restructure shifts venue posture from Findings to Main.

**Before (4 parallel contributions, ~360 words):**
> 본 논문은 네 contributions을 보고한다.
>
> (1) **Cross-modal anchoring 평가 프레임워크.** ...
>
> (2) **5 dataset × 6 model cross-dataset 증거 + 연속 confidence gradient.** ...
>
> (3) **Routing vs integration synthesis (이론적 기여).** ...
>
> (4) **Multi-direction subspace projection mitigation (E6).** ...
>
> *Auxiliary observation* — Reasoning-mode Qwen3-VL-8B-Thinking이 §4.4 continuous-confidence framework의 예측 방향으로 anchor pull을 amplify하는 N=1 architecture × N=1 dataset *existence proof* (×12.7 ratio §4.5, §8.2 한계).

**After (one central + three supporting + auxiliary, ~310 words):**
> 본 논문의 단일 *central contribution*은 **multi-direction subspace projection을 사용하는 cross-modal anchoring mitigation (E6)** 으로, 형식 정의된 *4-clause free-lunch* 기준 (Δdf < 0 ∧ Δem 양 arm ≥ 0 ∧ held-out capability ≥ −0.5 pp; §6.2.3) 을 5 evaluation dataset × 6 held-out capability benchmark 위에서 multiplicity-robust 하게 충족한다 — **Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 corrected CI 모두 excludes 0** 이 그 multiplicity-robust headline (§6.2.3 / §7). 본 mitigation은 세 *supporting finding* 위에 build된다 — (i) 5 dataset × 6 model cross-dataset 위에서 anchoring을 wrong/correct binary projection이 아닌 *continuous confidence gradient*로 재해석하는 세 직교 axis 증거 (L1 6-bin gradient, (a − m) digit-pixel causality, wrong/correct binary stratification; §4); (ii) signal이 multi-layer redundant 하다는 mechanism 발견과 이로부터 single-direction mitigation 실패를 통합 설명하는 *routing vs integration 사후 synthesis* (§5) — 이 synthesis는 §4.6 γ-β residual-stream bridge에서 layer-routing 방향성 sign-reversal로 prospectively 검증되나 implicit universal-K=8 가정은 K=1 vs K=8 cross-architecture 차이로 부분 falsify되며 (§4.6 Insight 2), 따라서 *load-bearing theory*가 아닌 *통합 설명 framework*로 자리한다; (iii) calibration substrate로서 (a − m) digit-pixel paired-inpaint contrast — CAA-style paired-contrast paradigm에 vision-modality specific *인과 통로 분리* 구조를 도입하는 generalisable design pattern (§6.2.1). 평가 protocol 자체 — independent-anchor open-numeric-estimation 4-condition (b/a/m/d) + gt-자유 direction-follow metric — 은 VLMBias [Vo, Nguyen et al., 2025]의 familiar-subject counting paradigm과 *cue independence × measurement type* 두 축에서 상보적이며 (§2 · §3), supporting finding (i)와 (iii)을 측정 가능하게 만드는 평가 substrate이다.
>
> *Auxiliary observation* — Reasoning-mode Qwen3-VL-8B-Thinking이 same continuous-confidence axis 위에서 anchor pull을 amplify하는 N=1 architecture × N=1 dataset *existence proof* (§4.5 behavioural; §4.6 residual-stream bridge가 framework의 prospective leg을 별도로 제공). §1.4 / §4.5 / §8.2 한계 참조.

**Rationale:** This single edit lands the user's central concern ("does this still read as experiment log?") at the contribution-framing axis. The restructure simultaneously (a) makes E6 mitigation the headline, (b) demotes synthesis to "supporting finding" + relabels framework explicitly (CRIT-N3), (c) elevates (a − m) substrate to its own supporting finding (MAJ-N4), (d) merges old (3) and (4) into one chain (MAJ-N5), (e) clarifies γ-β dual role between §4.5 behavioural and §4.6 residual-stream bridge (MAJ-N7), (f) drops "first-ness" framing on §1.5 (1) — protocol is now positioned as "complementary substrate" (MAJ-N3), (g) leads supporting finding (i) with continuous-confidence reframe noun (MIN-N2). The reviewer's drafted paragraph supplies the structural skeleton; tightened to remove the mid-sentence comma chains and the original "(i)/(ii)/(iii)" of supporting findings now also internally enumerates the three orthogonal axes inside (i).

### Edit 2 — §1.3 relabel "이론적 기여" → "통합 설명 framework" (CRIT-N3 propagation)
**Reviewer point addressed:** CRIT-N3 propagation site #1.

**Before:**
> 본 논문의 *이론적* 기여인 **routing vs integration framework**는 §5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak + §6.4 LEACE rank-1 ChartQA +56 % 역행을 단일 mechanism narrative로 묶는 *사후 synthesis*이다 (§5.4) ...

**After:**
> 본 논문의 **routing vs integration 통합 설명 framework** (사후 synthesis + §4.6 부분 prospective 검증)는 §5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak + §6.4 LEACE rank-1 ChartQA +56 % 역행을 단일 mechanism narrative로 묶는 *사후 synthesis*이다 (§5.4) ...

**Rationale:** Honest relabel. Intellectual content unchanged; the framework still does the same explanatory work. CRIT-N3 specifically asked for this exact one-word relabel; we extend with the parenthetical so the qualifier travels with the noun.

### Edit 3 — §8.1 종합 relabel "이론적 기여" → "통합 설명 framework" (CRIT-N3 propagation)
**Reviewer point addressed:** CRIT-N3 propagation site #2.

**Before:**
> 본 논문의 *이론적* 기여인 **routing vs integration synthesis** (§5.4) 가 ... 이 synthesis 로부터 multi-direction subspace projection이 single-direction failure mode를 우회하는 후보로 도출된다.

**After:**
> 본 논문의 **routing vs integration 통합 설명 framework** (사후 synthesis + §4.6 부분 prospective 검증; §5.4) 가 ... 이 framework로부터 multi-direction subspace projection이 single-direction failure mode를 우회하는 후보로 도출된다.

**Rationale:** §8.1 종합 must mirror §1 / Abstract framing. Both label changes + the "synthesis 로부터" → "framework로부터" cleanup keep §8.1 internally consistent with §1.5 + §1.3 post-restructure.

### Edit 4 — Abstract drop ×12.7 (CRIT-N2)
**Reviewer point addressed:** CRIT-N2.
**Reviewer reasoning:** ×12.7 is auxiliary observation but is the most quotable number with no CI; sitting next to the multiplicity-robust mitigation headline, it contaminates credibility.

**Before:**
> *Auxiliary observation*: Qwen3-VL-Thinking은 same continuous-confidence axis 위에서 anchor pull을 amplify (correct-base df ratio ×12.7 point estimate, N=1×N=1 existence proof; §4.5).

**After:**
> *Auxiliary observation*: Qwen3-VL-Thinking은 same continuous-confidence axis 위에서 anchor pull을 amplify하는 N=1 architecture × N=1 dataset *existence proof* (§4.5; ratio CI 미산출, denominator small).

**Rationale:** Reviewer's option (a) per task brief. The reader hitting Ctrl-F for "×12.7" now lands at §1.4 / §4.5 / §8.2 where the small-denominator hedge is bound to the number. Abstract is left with `existence proof` framing matching §1.5(aux) verbatim — Round-1 commit honoured. The "(ratio CI 미산출, denominator small)" parenthetical replaces the bare ratio with explicit acknowledgement of the deferred CI (§8.4 item 5).

### Edit 5 — §6.5 sub-title soften (CRIT-N1)
**Reviewer point addressed:** CRIT-N1.

**Before:** `### 6.5 부정적 결과 비교 — 4-clause free-lunch 통과 *유일* 후보`
**After:** `### 6.5 부정적 결과 비교 — 5-baseline panel 위 4-clause free-lunch 통과 후보`

**Rationale:** "유일" emphasis dropped; "5-baseline panel 위" makes the scope explicit upfront. Reviewer's option (i) per task brief.

### Edit 6 — §6.5 Table 8 verdict cell scope qualifier (CRIT-N1)
**Reviewer point addressed:** CRIT-N1.

**Before:**
> | **Multi-direction subspace (이 작업, K=8)** | ... | **권장 — 4-clause free-lunch 통과** |

**After:**
> | **Multi-direction subspace (이 작업, K=8)** | ... | **권장 — 4-clause 동시 충족 (이 5-baseline panel 위)** |

**Rationale:** Verdict cell now carries the panel-scope qualifier; reader reading just the table sees the scope without needing to read the Note paragraph.

### Edit 7 — §6.5 Insight Note 1 (CAA / ITI structural reduction with ITI multi-head trap addressed)
**Reviewer point addressed:** CRIT-N1 ITI multi-head trap, plus CRIT-N1 main "uniquely passes" softening.

**Before:** Single Note paragraph asserting CAA structural equivalence + ITI single-locus failure mode prediction; "5개 비교 방법 중 유일하게 통과" framing.

**After:** Split into Note 1 (CAA/ITI structural reduction, with explicit acknowledgement that "ITI는 multi-head cluster intervention이고 §5.2는 single-head ablation이다 — §5.2의 single-head single-layer null이 직접 cover하지 *않는다*", then carrying the §5.3 dataset-dependent peak as the *directional* prediction) + Note 2 (baseline tuning effort asymmetry, MAJ-N6).

**Rationale:** Round-3 reviewer's specific trap was that §5.2 single-head null does NOT cover ITI's multi-head intervention. We now acknowledge that gap explicitly. The structural prediction is now sourced from §5.3 (attention-locus dataset-dependence) rather than §5.2 (single-head null) — a cleaner load. Honest disclosure that two reductions are *structural* not *empirical*; CAA-at-K=1 + ITI multi-head empirical row deferred to §8.4 item 4.

### Edit 8 — §6.5 Insight Note 2 (baseline tuning effort asymmetry — MAJ-N6)
**Reviewer point addressed:** MAJ-N6.

**Before:** No corresponding text.

**After:** Added Note 2 paragraph stating each baseline evaluated at method-source default vs E6's 27-cell pilot grid (asymmetric tuning); fair-tuning predictions: (i) ActAdd / LEACE rank-1 cross-dataset failure quadrant likely persists (direction-mismatch is essence, not tunable), (ii) ITI multi-head fair-tuning could enable 4-clause partial-pass at some attention-head locus K-sweep cell. Both deferred to §8.4 item 4.

**Rationale:** Reviewer's MAJ-N6 explicitly asked for this — "what would change with fair-tuning increased". The two predictions distinguish CAA / LEACE-rank-1 (where fair-tuning probably won't rescue) from ITI (where it might). This is the honest position-taking the reviewer flagged as missing.

### Edit 9 — Table 8 caption tuning-asymmetry pointer
**Before:** "Multi-method 비교 ... Bold = 본 작업. *Note: CAA / ITI는 Insight 문단의 구조적 reduction으로 처리; 경험적 row는 §8.2 deferred.*"
**After:** "Multi-method 비교 ... Bold = 본 작업. *Note: 각 baseline은 method-source default 운영 모드로 평가되어 E6의 27-cell pilot grid 대비 *tuning effort가 비대칭*하다 (Insight 본문 Note 2 참조); fair-tuning 후 비교 + CAA / ITI empirical row는 §8.4 item 4 deferred.*"

**Rationale:** Caption now points to Note 2 + §8.4 item 4 so reader skimming Table 8 sees the asymmetry caveat without needing to read the full Insight prose.

### Edit 10 — §2 paragraph 4 Chand cross-axis + (a−m) design pattern + ITI TruthfulQA caveat + vision-token pruning positioning (MAJ-N2 + MAJ-N4 + MIN-N4 + MIN-N5)
**Reviewer point addressed:** MAJ-N2 (Chand to §2), MAJ-N4 ((a−m) design pattern in §2), MIN-N4 (ITI TruthfulQA caveat), MIN-N5 (vision-token pruning positioning).

**Before:** §2 paragraph 4 ended with "본 작업의 differentiator는 *기법 class의 신규성이 아니라 multi-direction × residual-stream × paired-inpaint 조합이 free-lunch 후보로 기능*한다는 점이다." No Chand positioning, no design-pattern sentence on (a − m), no ITI venue caveat, no vision-token pruning class.

**After:** Paragraph extended with (i) ITI TruthfulQA-only caveat in parenthetical, (ii) (a − m) paired-inpaint design-pattern sentence (calibration contrast must separate causal pathway from confounding variance), (iii) Chand cross-axis positioning sentence ("VLM × continuous numerical regression × inference-time activation projection에서 4-clause 동시 충족이 가능함" as cross-axis positive result vs Chand's LM × discrete social bias × weight space negative result), (iv) vision-token pruning class positioning (sentence demarcating site as LM post-cross-attention residual stream, not vision encoder / cross-modal attention).

**Rationale:** Four reviewer points addressed in one paragraph extension. MAJ-N2 specifically asked for §2 prominence of the cross-axis Chand framing; MAJ-N4 specifically asked for (a − m) design-pattern surfacing in §2; MIN-N4 and MIN-N5 piggyback on the same paragraph. Net §2 ~+90 words, all load-bearing.

### Edit 11 — §4.5 Insight 2: 4-class → 5-class (MIN-N3) + first-evidence → existence proof (MIN-N6)
**Before:** "텍스트 LRM에서 Wang et al. [2025a]이 LRM judging bias 4-class (bandwagon / authority / position / distraction) 전반에 보고한 *reasoning-amplifies* pattern ... 이 VLM에서 *first-evidence* 형태로 재현되며 ..."
**After:** "텍스트 LRM에서 Wang et al. [2025a]이 LRM judging bias 5-class (bandwagon / authority / position / distraction / superficial-reflection) 전반에 보고한 *reasoning-amplifies* pattern ... 이 VLM에서 *N=1 architecture × N=1 dataset existence proof* 형태로 재현되며 ..."

**Rationale:** MIN-N3 verified against arXiv:2504.09946 abstract — Wang 2025a is 5-class. MIN-N6 lifts the unhedged "first-evidence" to match Round-2 hedge consistency.

### Edit 12 — §8.2 first-evidence → N=1 × N=1 existence proof (MIN-N6 propagation)
**Before:** "§4.5의 reasoning-amplifies-anchoring 결과는 단일 architecture pair (Qwen3-VL-8B Instruct vs Thinking) × 단일 dataset (MathVista) × 단일 stratum의 *first-evidence* 결과로, ..."
**After:** "§4.5의 reasoning-amplifies-anchoring 결과는 단일 architecture pair (Qwen3-VL-8B Instruct vs Thinking) × 단일 dataset (MathVista) × 단일 stratum의 *N=1 × N=1 existence proof*로, ..."

**Rationale:** MIN-N6 propagation site #2. Same hedge consistency.

### Table edits

- **Table 8 verdict cell:** "권장 — 4-clause free-lunch 통과" → "권장 — 4-clause 동시 충족 (이 5-baseline panel 위)" (Edit 6).
- **Table 8 caption:** Note text rewritten for tuning asymmetry (Edit 9).

### Figure edits

None — all 16 inline figure embeds preserved.

## Rebuttals (DISAGREE class)

### Rebuttal 1 — MIN-N1 (§8.2 line 470 unmodified "free-lunch")

**Reviewer claim:** "§8.2 line 470 still uses unmodified `free-lunch` — paragraph reads `... 4-clause *free-lunch* 기준 위에서 충족 검증 ...`. This is the one terminology drift residual from MAJ-W6 cleanup."

**Our position:** Verified against current §8.2 (lines 462–476): the only remaining `free-lunch` instance in §8.2 is `**Mid-stack cluster 단일.** E4 free-lunch는 3 mid-stack cluster 모델에서 확립` — this is shorthand for "E4's free-lunch criterion result," not an unprefixed claim that the criterion itself is satisfied. The text the reviewer quoted (`4-clause *free-lunch* 기준 위에서 충족 검증`) does NOT appear in §8.2 as currently constituted — Round 2 already removed it. Round-3 reviewer flagged this as already-resolved (the review's own MAJ-N6 paragraph downgrades to "Minor residual"). We retain the `E4 free-lunch는 ... 확립` shorthand because (a) it is a back-reference to §6.1's E4 mitigation result, not a policy statement about the 4-clause criterion, (b) the full criterion is established earlier in the paper at §6.2.3 line 377 with explicit 4-clause prefix, (c) Round-2 explicitly considered this kind of shorthand acceptable.

**Why we believe the paper's position is correct:** Reviewer himself downgraded MAJ-N6 to "Minor residual; downgrade to MIN" after re-reading current draft. Current §8.2 language is shorthand for a back-reference, not unhedged criterion-policy claim. The grep verification was performed; no policy-tier site survives without prefix.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| MAJ-N8 (Dyslexify + E6 composition test) | Requires fresh experiment combining encoder-side typographic-circuit ablation with E6 LM residual hook on shared OneVision setup; ~1 H100-day. Not load-bearing for this round's positioning fixes. | Add to §8.4 as item 7 in next-revision pass; experiment not run yet |
| MIN-N7 (Goh et al. 2021 Distill venue inline) | Cosmetic; reference list already shows venue at line 506 | Style-pass at copy-edit phase |
| CRIT-N1 option (b) — CAA-at-K=1 + ITI multi-head empirical rows on PlotQA + InfoVQA pooled (a − m) calibration | Requires ~4–8 H200-hour for CAA-at-K=1 + ~1–2 day for ITI head-level adaptation; out of round-3 scope. We chose option (a) per task brief — soften framing, keep §8.4 item 4 as the deferral pointer. | §8.4 item 4 already lists this; commit to running before camera-ready if accepted |
| CRIT-N2 option (b) — paired-bootstrap CI on ×12.7 ratio | Requires γ-β raw per-cell numerator/denominator counts (~2 H100-hour). Round-1 deferred to §8.4 item 5; Round-3 chose option (a) per task brief — drop from Abstract instead of computing CI inline. | §8.4 item 5 retains this; commit to bootstrap CI before camera-ready |

## Open questions for next round

- **English compression target.** v11 Korean draft is 826 lines; English EMNLP target ~3,800–4,200 words main text. Compression hot zones: §4.4 6-bin detail (mostly to §B.1 already), §4.6 (load-bearing for CRIT-1 reframe; stays in main), §6.5 Note 1 + Note 2 (added this round, may compress at translation), §C.1 legacy VQAv2 (appendix already), §A.5 27-cell heatmap (appendix already). Pre-submission compression pass needs full English draft.
- **MAJ-N1 acceptance landed but venue verdict still depends on capability claims.** The reviewer's ranking ("EMNLP Main if §1.5 restructured + 3 CRITs fixed; Findings if not") is now anchored on the 3 CRITs being closed and §1.5 being restructured — both done in this round. Open question for round 4 (aggressive) and round 5 (bar-raiser): whether the substantive contribution holds up under hostile probing of the multiplicity-robust headline (Δem(b) Bonferroni-clean) + the 4-clause free-lunch on 5-baseline panel + the (a − m) substrate.
- **CAA-at-K=1 empirical row before submission.** The reviewer's option (b) closes CRIT-N1 *empirically*; we chose option (a) (rephrasing). If we have GPU budget pre-submission, running CAA-at-K=1 on the same PlotQA + InfoVQA pooled (a − m) calibration as E6 at L=26 would shift the verdict from "softened panel-scope statement" to "uniquely passes among all evaluated methods." Estimated cost ~4–8 H200-hour; recommend pre-camera-ready run.

## Internal consistency check

After all edits, verify:

- [x] **Abstract numbers** still match §4–§7 tables. Verified: 1.7-15.7 % (§4.1), +19.5-23.5 pp B6−B1 (§4.4), 5/5 null mech panel (§5.2), Δem(b) 5/5 cell × 95 % AND Bonferroni-20 CI excludes 0 (§6.2.3), +0.41 pp macro / +2.21 pp HallusionBench / POPE pinned (§7).
- [x] **§1.5 contributions still match §4–§7 deliveries.** Central contribution = E6 (§6.2 / §6.5 / §7); supporting (i) 5×6 cross-dataset + continuous gradient (§4.1 / §4.2 / §4.3 / §4.4); supporting (ii) routing-vs-integration synthesis with §4.6 prospective leg (§5.2 / §5.3 / §5.4 / §4.6 / §6.4); supporting (iii) (a − m) substrate (§6.2.1); auxiliary γ-β (§4.5 + §4.6 prospective leg). All sites still deliver the corresponding contribution.
- [x] **§8.1 종합 still consistent with body.** Reviewed: framework label updated to "통합 설명 framework" matching §1.3 + §1.5. ×12.7 in body-reference position (`(×12.7 ratio, §4.5)`) preserved. Δem(b) Bonferroni-robust headline preserved.
- [x] **All figure embeds still resolve.** 16 inline figures unchanged.
- [x] **No figure / table renumbering issues introduced.** Tables 1–9 + appendix tables intact.
- [x] **Canonical sources still cited where appropriate.** §A.5 reproducibility pointer retained; references list intact.
- [x] **"이론적 기여" appears 0 times in §1, Abstract, §8.1 in contribution-label role.** Verified via grep — only remaining "이론적" usages are *adverbial* ("이론적으로 예측") at §5.2 line 272 and §6.4 line 399, both meaning "predicted on logical/theoretical grounds" — these are not contribution labels and don't conflict with CRIT-N3.
- [x] **×12.7 not adjacent to multiplicity-robust mitigation headline.** Abstract auxiliary sentence no longer contains ×12.7. §1.2 / §8.1 mentions are body-reference paragraphs, not in the same paragraph as the Δem(b) headline.
- [x] **"uniquely passes" framing softened consistently.** §6.5 sub-title, Table 8 verdict cell, Insight Note 1 / Note 2 all carry "이 5-baseline panel 위" qualifier. No orphan "유일하게 통과" survives.
- [x] **No demoted claims resurrected.** Encoder-family-determines-archetype demotion (commit 549cf68) preserved at §D.1; not resurfaced.

## Diff stat

- **Lines:** 828 → 826 (net −2; §1.5 restructure compresses ~360 → ~310 words; §6.5 Note expansion adds ~200 words; §2 paragraph 4 extension adds ~90 words; net body line balance from ratio-of-prose).
- **Word delta:** §1.5 −50 words (compression); §2 +90 words (Chand + (a−m) + ITI + vision-token positioning); §6.5 +200 words (Note 1 split + Note 2 added + Table 8 caption rewritten); Abstract −10 words (×12.7 dropped); §1.3 / §8.1 / §4.5 / §8.2 +20 words combined (relabel + 4→5-class + first-evidence cleanup). **Net body: ~+250 words.**
- **Sections substantially rewritten:** §1.5 (4 contributions → 1 central + 3 supporting + auxiliary), §6.5 Insight (Note paragraph split into Note 1 + Note 2), §2 paragraph 4 (extended with four positioning sentences).
- **Files affected:**
  - `docs/paper/emnlp_draft_ko.md` (12 distinct edit sites, ~30+ cells in tables / captions changed)
  - `docs/paper/reviews/round3_response.md` (new — this file)
  - `docs/paper/CHANGELOG.md` (v11 entry to be appended in follow-up)
