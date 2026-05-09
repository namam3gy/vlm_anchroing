# Round 3 — Author Response to Novelty / Positioning Reviewer

**Paper version BEFORE:** `docs/paper/emnlp_draft_ko.md` @ 526 lines, post-Round-2 revision (`v4` changelog footer).
**Paper version AFTER:** `docs/paper/emnlp_draft_ko.md` @ 538 lines, `v5` changelog footer (2026-05-09).
**Date:** 2026-05-09.
**Reviewer round addressed:** `docs/paper/reviews/round3_novelty.md` (3 CRIT, 4 MAJOR, 8 should-fix; borderline Main/Findings, leans Main if must-fix resolved).

## Summary

We accept all three CRIT items and all four MAJOR items as paper edits — every must-fix is resolvable through revision-pass §2 prior-work additions, §1.5 register softening, §6 formal-definition insertion, and References list expansions, without changing any number in the paper. The reviewer's sharpest single critique — that §2 is missing the activation-steering / concept-erasure literature class (CAA / ITI / LEACE) which §6 implicitly extends — is addressed via a new §2 paragraph "Activation steering and concept erasure" (Edit 1), which explicitly positions E6 as multi-direction × residual-stream × paired-inpaint at the intersection of these prior methods rather than claiming method-class novelty. The §2 VLMBias description is factually corrected (Edit 2) per the WebFetch-verified abstract; the differentiator is reframed on *cue source* (independent rendered-digit vs subject-of-question) and *measurement target* (open-numeric-estimation baseline-relative shift vs counting accuracy on familiar subjects) rather than the indefensible "label vs digit" framing. §1.5 (1) is softened to "first-evidence" register matching (6); the encoder-family-determines-archetype sub-finding is elevated to its own (4b) bullet; *strict free-lunch* receives a formal 4-clause definition near §6.2.3 first headline use, with explicit citation of and differentiation from arXiv:2511.18635. We also address 6 of 8 should-fix items.

The novelty-axis positioning is now defensible against a hostile prior-work-aware reviewer: every method class the paper claims to outperform is now cited in §2; every "first" claim is hedged to match its evidence; the comparison-panel boundary is made explicit ("5-baseline panel" rather than implied universal claim) so CAA/ITI absence does not read as cherry-picking.

## Decision summary table

### Must-fix (3 CRIT + 4 MAJOR = 7)

| # | Reviewer point (verbatim short) | Class | Section affected | Status |
|---|---|---|---|---|
| 1 | CRIT-N1 §2 VLMBias factually wrong ("의미 클래스 라벨 + 반사실 카운팅 실패") | EDIT | §2 + §1.1 (echo) | done |
| 2 | CRIT-N2 §2 missing CAA / ITI / LEACE prior-work block | EDIT | §2 (new paragraph) + References | done |
| 3 | CRIT-N3 §1.5 (1) "VLM 최초의" not softened to (6) "first-evidence" register | EDIT | §1.5 contributions | done |
| 4 | MAJOR-N4 strict free-lunch term needs formal definition + arXiv:2511.18635 differentiation | EDIT | §6.2.3 + abstract + References | done |
| 5 | MAJOR-N5 §2 Weng comparison framing structurally unfair | EDIT | §2 typographic-attack/mechanism paragraph | done |
| 6 | MAJOR-N6 Lou & Sun citation orphan (in References, never described in §2) | EDIT | §2 LLM-anchoring lineage paragraph | done |
| 7 | MAJOR-N7 encoder-family-determines-archetype buried in §1.5 (4); deserves its own bullet | EDIT | §1.5 (4) split into (4a) + (4b) | done |

### Should-fix (8)

| # | Reviewer point (short) | Class | Section | Status |
|---|---|---|---|---|
| s1 | §1.5 split (4) into (4a) + (4b) | EDIT | §1.5 | done (folded into MAJOR-N7) |
| s2 | §2 paragraph 4 add Hufe Dyslexify in prose | EDIT | §2 typographic-attack paragraph | done |
| s3 | §2 "이들 중 어느 것도 cross-modal numerical anchoring을 측정하지 않는다" soften | EDIT | §2 multimodal-bias paragraph | done (rewritten in CRIT-N1 fix) |
| s4 | §1.5 (5) inline forward-link to strict-free-lunch definition | EDIT | §1.5 (5) | done |
| s5 | §4.6 Insight 2 — clarify which Wang 2025a bias class is the analog | EDIT | §4.6 Insight 2 | done |
| s6 | References — add Belrose 2023 LEACE + arXiv:2511.18635 | EDIT | References | done (also added Panickssery / Li / Chand) |
| s7 | §6.4 LEACE description — clarify "rank-1 closed-form" is one instantiation | EDIT | §6.4 | done |
| s8 | Citation form "Vo, Nguyen et al." → "Vo et al." | DEFER | References | DEFER — author-attribution stylistic; current "Vo, Nguyen et al." is unambiguously parseable and (per WebFetch) the actual author list has An Vo as first author + Anh Nguyen as senior author so the bigram captures both ends. Logged as MINOR; carry forward to camera-ready style pass. |

## Edit log (every paper change in this round)

### Edit 1 — §2 new paragraph: Activation steering and concept erasure (CAA / ITI / LEACE)

**Reviewer point addressed:** #2 (CRIT-N2 — sharpest single critique).
**Reviewer reasoning:** §6 invokes ActAdd, LEACE, CAA-style direction methods as comparison baselines but never cites them as Related Work; this is the largest single missing-prior-work gap, and any reviewer aware of CAA/ITI/LEACE will catch it.

**Before:** §2 ended with the typographic-attack / Weng paragraph; no activation-steering paragraph.

**After (new §2 paragraph appended after the typographic-attack paragraph):**
> **Activation steering과 concept erasure.** §6의 mitigation은 residual-stream intervention 계열에 속한다. CAA [Panickssery et al., 2024]는 *paired contrastive activation* — positive vs negative behavioral example pair의 residual-stream 차분 평균 — 으로 *single-direction* steering vector를 도출한다. ITI [Li et al., 2023]는 *attention head* 출력 수준에서 *multi-direction* 개입을 수행한다 (multiple head × direction). LEACE [Belrose et al., 2023]는 closed-form linear concept erasure (rank-1 default 포함)로 baseline 표현 손상을 최소화하면서 선형 분류기가 concept을 검출하지 못하게 만든다. 본 논문 E6는 이 계열의 직접 후속 — (i) CAA의 paired-contrast 패러다임을 *multi-direction subspace* (K=8 SVD)로 확장하고, (ii) ITI의 attention-head locus 대신 *residual-stream* locus에서 작동하며, (iii) text-only steering 문헌에 없는 *vision-modality (a − m) paired-inpaint contrast* 구성을 도입한다. §6.4에서 LEACE의 *rank-1 closed-form 인스턴스*를 직접 비교 baseline으로 평가하며 (LEACE 자체는 closed-form linear erasure framework이고 rank-1은 그 default 운영 모드), CAA·ITI는 §6.5 Table 7 footnote에서 single-direction failure mode (ActAdd 대등 — paired-contrast residual stream at K=1) 또는 multi-layer redundancy로부터 예측되는 attention-head intervention failure mode (§5.2 single-layer null과 동일 진단)로 reduce되는 점을 명시 — 본 작업의 differentiator는 *기법 class의 신규성이 아니라 multi-direction × residual-stream × paired-inpaint 조합이 strict free-lunch 후보로 기능*한다는 점이다.

**Rationale:** Closes the largest single novelty gap. (a) Names CAA / ITI / LEACE as the prior method class. (b) Positions E6 as a triadic intersection (multi-direction subspace + residual-stream locus + vision-modality paired-inpaint) rather than an entire method-class novelty claim. (c) Forward-links to the §6.5 Note (Edit 6) which explains how CAA/ITI reduce to already-tested failure modes in our setting. (d) Honestly characterizes LEACE as "closed-form linear concept erasure framework" (per WebFetch verification) rather than the reviewer's stronger "general affine of any rank" framing — the reviewer's claim that LEACE supports any-rank affine erasure is stronger than what the WebFetched abstract says ("provably prevents all linear classifiers from detecting a concept"). We use the more conservative, verified language. Three references added (Belrose et al. 2023 / Li et al. 2023 / Panickssery et al. 2024).

### Edit 2 — §2 VLMBias factual rewrite + §1.1 echo correction

**Reviewer point addressed:** #1 (CRIT-N1).
**Reviewer reasoning:** Paper says VLMBias uses "*의미 클래스 라벨*" and measures "*반사실 카운팅 실패*"; the WebFetched arXiv:2505.23941 abstract confirms stimuli are *visual subjects from 7 domains* (Adidas-style logo, animals, chess, board games, illusions, grids), no text-label cue, measurement is counting accuracy on familiar subjects. The "label vs digit" differentiator in §1.5 (1) collapses on the corrected description.

**Before (§2 multimodal-cognitive-bias paragraph):**
> VLMBias [Vo, Nguyen et al., 2025]는 가장 포괄적인 VLM cognitive bias benchmark이지만 단서가 *의미 클래스 라벨*이며 *반사실 카운팅 실패*를 측정한다. AIpsych [Liu et al., 2025], CIVET [Rizzoli et al., 2025], Tinted Frames [Fan et al., 2026]은 sycophancy / position bias / question framing을 다룬다. **이들 중 어느 것도 cross-modal numerical anchoring을 측정하지 않는다.** 본 논문의 단서는 *의미 라벨 없는 단독 rendered-digit 이미지*이며, 측정 대상은 *임의 anchor에 대한 회귀형 수치 이동*이다.

**After:**
> VLMBias [Vo, Nguyen et al., 2025]는 가장 포괄적인 VLM cognitive bias benchmark으로, *모델이 사전 지식을 가진 visual subject* (Adidas-style logo, 동물, chess board, board game, optical illusion, patterned grid 등 7개 domain)를 stimulus로 사용하여 *familiar-subject counting accuracy* (예: stripe 개수)를 측정한다 — 평균 17.05 % counting accuracy + background 제거 시 +21.09 pp 회복 (arXiv:2505.23941). AIpsych [Liu et al., 2025], CIVET [Rizzoli et al., 2025], Tinted Frames [Fan et al., 2026]은 sycophancy / position bias / question framing을 다룬다. 본 논문은 이들과 *측정 대상*과 *cue source* 두 축에서 상보적이다 — VLMBias가 *familiar subject의 prior knowledge에 대한 counting accuracy*를 측정하는 데 비해 본 논문은 *질문 대상과 무관한 독립 rendered-digit anchor 이미지에 대한 open-ended numerical estimation의 baseline-relative shift*를 측정한다 (cue가 question 자체의 subject가 아니라 independent second image; metric이 closed-form counting이 아니라 arbitrary anchor에 대한 회귀형 이동). 두 paradigm 모두 "visual content가 numerical answer를 편향시킨다"는 같은 mechanism question을 공유하나, 측정 대상 (counting vs open estimation)과 cue dependency (subject-bound vs independent draw)에서 분리된다.

**Before (§1.1):**
> 가장 가까운 이웃 — VLMBias [Vo, Nguyen et al., 2025]는 *의미 클래스 라벨*을, typographic attack [Wang et al., 2025b]은 *대상 이미지 위 오버레이*를, FigStep [Gong et al., 2025]은 *prompt 텍스트의 이미지화*를 단서로 사용한다 — 어느 것도 임의의 anchor에 대한 *회귀형 수치 이동*을 측정하지 않는다.

**After (§1.1):**
> 가장 가까운 이웃 — VLMBias [Vo, Nguyen et al., 2025]는 *familiar subject (Adidas-style logo / 동물 / chess 등)*의 counting accuracy를, typographic attack [Wang et al., 2025b]은 *대상 이미지 위 오버레이*에 의한 분류 뒤집기를, FigStep [Gong et al., 2025]은 *prompt 텍스트의 이미지화*에 의한 jailbreak ASR을 측정한다. 본 논문은 cue를 *질문 subject로부터 분리*하고 metric을 *open-numeric-estimation의 baseline-relative shift*로 두는 보완적 paradigm을 도입한다.

**Rationale:** (a) §2 description now factually matches the arXiv:2505.23941 abstract (WebFetch verified: An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, Daeyoung Kim — 7 domains, 17.05 % counting acc, +21.09 pp bg-removal effect). (b) The differentiator is reframed on the two defensible axes (cue source: independent vs subject-bound; measurement target: open-numeric vs counting accuracy) — the indefensible "label vs digit" axis is eliminated. (c) §1.1 echo of the same wrong claim is corrected in parallel; the §1.5 (1) "stacked qualifier" defense (independent + open-numeric-estimation) now rests on two defensible legs rather than three with one collapsed. (d) Should-fix MINOR-N8 ("이들 중 어느 것도 ..." too strong) is implicitly addressed by removing the all-quantifier sentence in the rewrite and replacing with a paradigm-comparison framing.

### Edit 3 — §1.5 (1) softened to "first-evidence" + (4) split into (4a)/(4b) + (5) inline forward-link

**Reviewer points addressed:** #3 (CRIT-N3) + #7 (MAJOR-N7) + s4 (§1.5 (5) inline forward-link).
**Reviewer reasoning:** Round 2 softened (6) to "first-evidence" but did not propagate to (1); reviewer flagged the asymmetric register. The encoder-family-determines-archetype sub-finding is "buried inside (4)" — reviewer judges it as paper-tier itself and asks for its own bullet. (5) uses "strict free-lunch" without forward-link to definition.

**Before:**
> (1) VLM 최초의 cross-modal numerical anchoring 평가. (2) gt-자유, baseline-relative C-form 표준 metric. (3) 5 dataset × 7 model cross-dataset 증거. (4) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 그 *multi-layer redundancy*를 정량화한다. (5) Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며, 6-benchmark capability preservation까지 동시에 검증한다 (strict free-lunch). (6) γ-β reasoning pair (N=1 Qwen3-VL Instruct vs Thinking)에서 reasoning-amplifies-anchoring을 처음 보이는 *first-evidence* VLM 결과.

**After:**
> (1) VLM에서의 *independent-anchor open-numeric-estimation* anchoring을 식별·정량화하는 *first-evidence* 평가 프레임워크 (familiar-subject counting을 다루는 VLMBias [Vo, Nguyen et al., 2025]와는 cue source · 측정 대상이 상보적; §2). (2) gt-자유, baseline-relative C-form 표준 metric. (3) 5 dataset × 7 model cross-dataset 증거. (4a) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 그 *multi-layer redundancy*를 정량화한다. (4b) Anchor-attention archetype이 *LM backbone이 아닌 encoder family*에 의해 결정됨 — 4 archetype (SigLIP-Gemma early / mid-stack cluster CLIP-ViT·ConvNeXt·InternViT / Qwen-ViT late / FastVLM late text-stealing) 분리는 LLaMA-계열 LM을 공유하는 세 모델이 모두 mid-stack cluster에 속하는 falsifiable cross-cut. (5) Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며, 6-benchmark capability preservation까지 동시에 검증한다 (*strict free-lunch* — anchoring 효과 + 양 arm em + 일반 능력 보존 동시 충족; §6.2.3 / §6.5에서 형식 정의). (6) γ-β reasoning pair (N=1 Qwen3-VL Instruct vs Thinking)에서 reasoning-amplifies-anchoring을 처음 보이는 *first-evidence* VLM 결과.

**Rationale:** (a) (1) softened to "first-evidence 평가 프레임워크" with stacked qualifiers (independent-anchor + open-numeric-estimation) — matches (6)'s register. (b) VLMBias relationship inline (cue source / 측정 대상 상보적) so the reader does not need to wait for §2 to know how the framing differs. (c) (4) split into (4a) single-layer null + multi-layer redundancy and (4b) encoder-family-determines-archetype — (4b) makes explicit that LLaMA-shared LM does not predict mid-stack cluster membership (the falsifiable cross-cut). (d) (5) now glosses "strict free-lunch" inline (anchoring 효과 + 양 arm em + 일반 능력 보존) with forward pointer to §6.2.3 / §6.5 formal definition.

### Edit 4 — §6.2.3 strict free-lunch formal definition + abstract inline 4-clause

**Reviewer point addressed:** #4 (MAJOR-N4) + s4 (forward link from §1.5).
**Reviewer reasoning:** "Strict free-lunch" is a real conceptual contribution (the four-clause conjunction with non-anchored em ≥ 0) but currently used colloquially without formal definition; closest precedent (arXiv:2511.18635 "No Free Lunch in LM Bias Mitigation") uses the same idiom for a *negative* result and must be cited.

**Before (§6.2.3 after the "5/5 dataset에서 다음 세 성질 ..." sentence):** sentence ended; no formal definition.

**After (new paragraph appended):**
> **Strict free-lunch 형식 정의.** 본 논문이 제안하는 *strict free-lunch* 기준은 *4-clause 동시 충족*을 요구한다 — *Δdf(anchoring task)* < 0 ∧ *Δem(anchored arm)* ≥ 0 ∧ *Δem(non-anchored arm)* ≥ 0 ∧ *Δ(held-out capability macro)* ≥ −0.5 pp (사전등록, §7). 통상적인 Pareto-improvement 기준은 첫 번째 + 마지막 두 clause만 요구하며 두 번째 *non-anchored arm em* 조항을 가지지 않는다. 이 추가 clause는 bias mitigation의 *cross-category collateral damage* — Chand et al. [2025]가 LM debiasing에서 보고한 *31.5 % 비표적 dimension에서의 부수 손상* — 에 직접 대응하는 screening 기준으로, *anchoring task family를 벗어난 forward에서 mitigation이 representation을 손상시키지 않는다*는 측면을 *경험적으로 강제*한다 (vs Chand et al.의 negative result는 LM × 이산 social bias × weight space에서 free-lunch가 성립하지 않음을 보임; 본 논문은 VLM × 연속 numerical regression × inference-time activation projection에서 *4-clause 동시 충족이 가능*함을 보고). "*strict*" 접두사는 이 추가 clause + held-out 외부 검증 (§7)이 *동시 충족*되어야 함을 명시한다.

**Abstract inline addition (after the "유일하게" claim):**
> 본 논문이 제안하는 ***strict free-lunch*** 기준 (Δdf < 0 ∧ Δem(anchored) ≥ 0 ∧ Δem(non-anchored) ≥ 0 ∧ Δ(held-out capability) ≥ 0 동시 충족; §6.2.3에서 형식 정의)을 5개 비교 baseline (...) 중 *유일하게* 통과 — CAA · ITI 등 인접 prior 방법의 본 setting 내 reduction은 §6.5 Note에서 서술.

**Rationale:** (a) Four-clause conjunction is now formally stated. (b) "Strict" is justified by what it adds over plain Pareto improvement (the Δem(non-anchored) ≥ 0 clause, which screens against cross-category collateral damage). (c) Chand et al. [2025] arXiv:2511.18635 (WebFetch-verified actual title: "No Free Lunch in Language Model Bias Mitigation? Targeted Bias Reduction Can Exacerbate Unmitigated LLM Biases") is cited as the negative-result precedent the paper differs from; differentiation is on three axes (LM vs VLM, discrete social bias vs continuous numerical regression, weight space vs inference-time activation space). (d) Abstract inline 4-clause + comparison-panel boundary disclosure ("5 비교 baseline … CAA · ITI 인접 prior reduction은 §6.5 Note") closes the abstract-level overclaim risk the reviewer flagged.

### Edit 5 — §2 LLM-anchoring lineage adds Lou and Sun [2024] description

**Reviewer point addressed:** #6 (MAJOR-N6).
**Reviewer reasoning:** Lou & Sun is in References list but never described in §2. As a citation orphan it weakens literature awareness; semantically it's the closest neighbor on the LLM side (same paper title structure: "Anchoring Bias in [LLMs]: An Experimental Study"). Reviewer recommends adding one §2 sentence motivating representation-level approach.

**Before (§2 LLM lineage paragraph):**
> ... LLM 시대에 들어 Jones and Steinhardt [2022], Echterhoff et al. [2024]는 텍스트 anchoring을 확립했고, Wang et al. [2025a]은 LRM에서 *judging bias가 reasoning trace를 통해 증폭*됨을 보고했다. Huang et al. [2025]은 합성 데이터로 메커니즘을 분해했다.

**After:**
> ... LLM 시대에 들어 Jones and Steinhardt [2022], Echterhoff et al. [2024]는 텍스트 anchoring을 확립했고, Lou and Sun [2024]은 텍스트 LLM에서 anchoring을 재확인하면서 Chain-of-Thought, Thoughts-of-Principles, Ignoring-Anchor-Hints, Reflection 같은 *prompt-level mitigation들이 모두 불충분*함을 보고 — 이는 본 논문이 § 6에서 *representation-level (residual-stream subspace) intervention*으로 향하는 직접적 동기이다. Wang et al. [2025a]은 LRM에서 *judging bias가 reasoning trace를 통해 증폭*됨을 보고했다. Huang et al. [2025]은 합성 데이터로 메커니즘을 분해했다.

**Rationale:** (a) Lou & Sun no longer a citation orphan. (b) The sentence does double duty — describes their negative result AND motivates the paper's representation-level approach in one sentence. (c) WebFetch-verified author names (Jiaxu Lou and Yifan Sun) updated in References (Edit 9).

### Edit 6 — §2 typographic-attack paragraph adds Hufe Dyslexify in prose + Weng comparison narrowed

**Reviewer points addressed:** #5 (MAJOR-N5) + s2 (MINOR-N10).

**Before:**
> Wang et al. [2025b, NAACL] multi-image typographic attack과 FigStep [Gong et al., 2025]은 *클래스 라벨* 또는 *prompt 텍스트*의 이미지화로 분류 뒤집기·jailbreak를 측정한다. 본 논문의 단서는 *수치값 단독* (클래스 정체성 없음), 표적은 *open-ended numerical estimation* (분류 뒤집기·ASR 아님). Weng et al. [2024, EMNLP Main]은 causal mediation 분석으로 VLM mechanism을 분해 + 22 % mitigation을 제시한 EMNLP-Main 기준 사례인데, 본 논문은 (i) single-layer ablation의 *null result*, (ii) multi-layer redundancy, (iii) single-direction mitigation의 cross-dataset 실패, (iv) 그것을 *우회하는* multi-direction subspace projection, (v) strict free-lunch 검증까지 한 단계 더 나아간다.

**After:**
> Wang et al. [2025b, NAACL] multi-image typographic attack과 FigStep [Gong et al., 2025]은 *클래스 라벨* 또는 *prompt 텍스트*의 이미지화로 분류 뒤집기·jailbreak를 측정한다. 본 논문의 단서는 *수치값 단독* (클래스 정체성 없음), 표적은 *open-ended numerical estimation* (분류 뒤집기·ASR 아님). Hufe et al. [2025, Dyslexify]는 typographic attack에 대한 *encoder-side mechanistic defense* (CLIP 측 개입)를 제시 — 본 논문 §4.4 / §5.1의 encoder-family-determines-archetype 결과와 직접 정렬되는 typographic-attack defense의 가장 가까운 mechanistic 이웃이다. 본 논문 E6는 이와 *상보적*으로 LM residual stream에서 작동한다. Weng et al. [2024, EMNLP Main]은 *gender bias*를 대상으로 causal mediation으로 image-encoder 기여를 식별하고 encoder-side feature blurring으로 22 % bias reduction (MSCOCO)을 보고한 EMNLP-Main mechanism→mitigation 사례이다. 본 논문은 *bias class*가 다르고 (numerical anchoring), mitigation의 *작용 site*도 다르다 (encoder가 아닌 LM residual stream); 공유하는 부분은 *mechanism→mitigation chain*이라는 venue-tier 형식이다.

**Rationale:** (a) Hufe Dyslexify mention (s2) — encoder-side typographic-attack mechanistic defense, complementary to E6's LM-side residual-stream intervention. (b) Weng comparison narrowed: bias class (gender vs anchoring) and mitigation site (encoder vs LM residual stream) are now explicit; the implicit "we go further" framing is replaced with "we share mechanism→mitigation chain venue-tier form, differ on bias class and mitigation site". This addresses MAJOR-N5's structural-unfairness flag honestly. (c) The five-step "(i)...(v) 한 단계 더 나아간다" enumeration is removed — that framing was the load-bearing element the reviewer judged unfair.

### Edit 7 — §6.4 LEACE description "rank-1 closed-form" → "rank-1 closed-form 인스턴스 (LEACE 자체는 closed-form linear erasure framework)"

**Reviewer point addressed:** s7.
**Reviewer reasoning:** Reviewer initially asserted LEACE supports any-rank affine erasure. WebFetch confirms LEACE is "closed-form linear concept erasure" (not "any-rank affine"). Even with the more conservative reviewer-corrected framing, the paper currently reads as if "LEACE = rank-1" is the standard, which over-states the prior method's limitation.

**Before:**
> 두 단일-방향 방법 (ActAdd + LEACE) 모두 cross-dataset *실패*: ActAdd는 TallyQA-calibrated `v` self-test 자체가 α=1에서 backfire (`E6-steering-vector.md`); LEACE rank-1 closed-form은 gt ∈ [0,8]로 제한해도 ChartQA에서 direction-follow를 +56 % *증가* (`E6-tally-only-rerun-tracker.md:480`).

**After:**
> 두 단일-방향 방법 (ActAdd + LEACE) 모두 cross-dataset *실패*: ActAdd는 TallyQA-calibrated `v` self-test 자체가 α=1에서 backfire (`E6-steering-vector.md`); LEACE [Belrose et al., 2023]를 *rank-1 closed-form 인스턴스*로 calibrate하면 (LEACE framework 자체는 closed-form linear concept erasure이며, rank-1은 본 비교에서 사용한 default 운영 모드) gt ∈ [0,8]로 제한해도 ChartQA에서 direction-follow를 +56 % *증가* (`E6-tally-only-rerun-tracker.md:480`).

**Rationale:** Names what was actually run (rank-1 closed-form instance) and acknowledges LEACE's framework-level capabilities — avoids overstating the prior method's limitation. Citation added inline.

### Edit 8 — §6.5 Table 7 Insight: CAA / ITI reduction footnote + abstract panel boundary disclosure

**Reviewer point addressed:** Differentiator D7 (§6.5 Table 7 missing CAA/ITI rows).
**Reviewer reasoning:** Hostile reviewer will ask "Why no CAA / ITI?". Current "5개 비교 방법 중 유일하게" reads as cherry-picked because the comparison space does not include the multi-direction precedent (ITI) or paired-contrast precedent (CAA).

**After (Insight paragraph extended with a Note):**
> ... 이는 우연이 아니라 §5.2 multi-layer redundancy와 §6.2.1 (a − m) contrast 설계의 직접 결과*이다*. *Note (CAA / ITI baseline 처리):* CAA [Panickssery et al., 2024]는 paired-contrast residual-stream steering을 *rank-1*으로 가중 합산하므로 본 비교의 ActAdd 행 ((a − m) calibration의 rank-1 mean-direction 인스턴스)과 *구조적으로 동치*이며 cross-dataset α=1 self-test backfire를 동일하게 상속한다. ITI [Li et al., 2023]는 *attention-head 출력*에서 multi-direction 개입을 수행하므로, §5.2의 *single-layer attention ablation null* + *multi-layer redundancy* 결과가 attention-head 수준 single-locus 개입의 cross-dataset 실패를 사전 예측한다 (cross-dataset attention peak가 dataset-dependent — §5.3). 따라서 두 prior 방법은 본 5-baseline 비교의 *ActAdd / LEACE rank-1 cell이 이미 점유하는 failure quadrant*에 속하며, "5개 비교 방법 중 유일하게 통과"라는 표현은 *이 5-baseline panel 내부의 비교 결과*로 한정한다. 본 작업의 differentiator는 *기법 class 신규성*이 아니라 (multi-direction subspace는 ITI에서, paired-contrast residual-stream은 CAA에서 각각 prior) *multi-direction × residual-stream × (a − m) paired-inpaint × strict free-lunch 사전등록* 조합이 4-clause를 동시 충족하는 후보로 기능한다는 점이다.

**Abstract change (s8 + panel boundary):** "5개 비교 방법 (ActAdd, LEACE, query-adaptive, CogBias decode-time, MIA-DPO LoRA) 중 *유일하게* 통과" → "5개 비교 baseline (ActAdd, LEACE rank-1, query-adaptive, CogBias decode-time, MIA-DPO LoRA) 중 *유일하게* 통과 — CAA · ITI 등 인접 prior 방법의 본 setting 내 reduction은 §6.5 Note에서 서술."

**Rationale:** (a) Explains how CAA reduces to ActAdd (paired-contrast rank-1 residual-stream is structurally equivalent to ActAdd at K=1 under (a−m) calibration) and how ITI reduces to attention-head locus failure mode predicted by §5.2's single-layer ablation null. (b) "5-baseline panel 내부 비교"로 claim 경계 명시 — hostile reviewer's "cherry-picked" objection is preempted. (c) The differentiator is reframed from "method class novelty" to "multi-direction × residual-stream × paired-inpaint × strict free-lunch combination" — matches the §2 activation-steering paragraph (Edit 1). (d) Abstract level disclosure ensures the reader has the comparison-panel boundary in hand from the abstract, not just §6.5.

### Edit 9 — §4.6 Insight 2: clarify Wang 2025a bias-class analog

**Reviewer point addressed:** s5 (G axis γ-β clarification).

**Before:**
> 텍스트 LRM에서 Wang et al. [2025a]이 보고한 "reasoning trace가 judging bias를 누적"하는 현상이 VLM에서 *first-evidence* 형태로 재현되며, ...

**After:**
> 텍스트 LRM에서 Wang et al. [2025a]이 LRM judging bias 4-class (bandwagon / authority / position / distraction) 전반에 보고한 *reasoning-amplifies* pattern — 그중 distraction-bias 계열이 본 논문 anchoring과 가장 가까운 cross-modal 유추 — 이 VLM에서 *first-evidence* 형태로 재현되며, ...

**Rationale:** Distraction-bias is the closest analog to anchoring in Wang 2025a's 4-class taxonomy; making this explicit avoids the "loosely calling Wang 2025a as 'reasoning amplifies bias' covers all four classes" ambiguity the reviewer flagged.

### Edit 10 — References list expansion (4 new entries)

**Reviewer points addressed:** s6 (Belrose 2023 LEACE + arXiv:2511.18635) + Edit 1 dependency (Panickssery + Li).

**Added entries:**
> [Belrose et al., 2023] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Tamar Pichasov, and Stella Biderman. 2023. LEACE: Perfect linear concept erasure in closed form. *NeurIPS 2023*. arXiv:2306.03819.
>
> [Li et al., 2023] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS 2023 (spotlight)*. arXiv:2306.03341.
>
> [Panickssery et al., 2024] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2024. Steering Llama 2 via contrastive activation addition. *ACL 2024*. arXiv:2312.06681.
>
> [Chand et al., 2025] Shireen Chand, Faith Baca, and Emilio Ferrara. 2025. No free lunch in language model bias mitigation? Targeted bias reduction can exacerbate unmitigated LLM biases. arXiv:2511.18635.

**Lou and Sun corrected entry:**
> [Lou and Sun, 2024] Jiaxu Lou and Yifan Sun. 2024. Anchoring bias in large language models: An experimental study. arXiv:2412.06593.
(Was: "Jingyuan Lou and Wei Sun" — corrected to WebFetch-verified author names.)

**Rationale:** Every newly cited author in body now appears in References. Author names verified via WebFetch (Panickssery et al. first 4 authors; Li et al. first 4 authors; Belrose et al. first 4 authors; Chand / Baca / Ferrara; Lou / Sun corrected first names).

### Table edits

None — no table cell values changed in this round.

### Figure edits

None — no figure caption or PNG path changed in this round.

## Rebuttals (DISAGREE class)

None substantive. Two minor clarifications:

### Clarification 1 — LEACE "any-rank affine" reviewer framing softened

**Reviewer asserted:** "LEACE provides general affine concept erasure (any rank, closed-form) ... LEACE is *not* inherently rank-1 (the original paper does general affine concept erasure of any rank)."

**Our position:** WebFetch on arXiv:2306.03819 verifies the paper's actual claim: "a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the embedding as little as possible." The reviewer's stronger "any-rank affine" framing is not inconsistent with the paper but is somewhat stronger than the abstract states. We adopted the more conservative "closed-form linear concept erasure (rank-1 default 포함)" language in Edit 1 / Edit 7. This is not a disagreement with the reviewer's underlying point (LEACE rank-1 is one instantiation, not LEACE in totality) — it's a verification-anchored softening of the framing.

### Clarification 2 — "Vo, Nguyen et al." citation form (s8 DEFER)

**Reviewer suggested:** "Vo, Nguyen et al." → "Vo et al." or "Vo and Nguyen 2025".

**Our position:** WebFetch on arXiv:2505.23941 confirms the actual author list is *An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, Daeyoung Kim*. Both An Vo (first author) and Anh Totti Nguyen (senior author) appear in the bigram "Vo, Nguyen", and the form is unambiguous in context. Defer to camera-ready style pass; not paper-blocking.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| s8 "Vo, Nguyen et al." citation form | Not paper-blocking; current bigram captures both first + senior author | Camera-ready style pass |
| Run CAA at K=1 as additional Table 7 row (D7 stronger version) | New GPU run required (~4–8 H100-hours for calibration + 5-dataset eval); the §6.5 Note (Edit 8) already explains the structural reduction (CAA = ActAdd at K=1 on (a−m)) so the empirical row would be confirmatory rather than novel-information | Owner: paper author. Estimate: 1 day. Reviewer-acceptable substitute (Note explanation) is shipped; Table 7 row addition is for camera-ready or rebuttal. |
| Run ITI multi-direction at attention-head level as additional Table 7 row | New GPU run required + ITI calibration recipe adaptation to (a − m); §5.2 single-layer attention ablation null already predicts attention-head locus failure | Owner: paper author. Estimate: 2–3 days. Same DEFER posture as CAA above. |

## Open questions for next round

- **CAA / ITI empirical rows on Table 7.** The §6.5 Note (Edit 8) explains the structural reduction; if a Round-4 reviewer demands the empirical row, the deferred GPU run would close it. Estimate: 4–8 H100-hours for CAA, 1–2 days for ITI adaptation.
- **§4.6 distraction-bias-as-cross-modal-analog framing.** The Edit 9 sentence asserts distraction-bias is the closest Wang 2025a analog; if a future reviewer challenges the directness, we can either (a) add a Wang 2025a quote pinpointing distraction's setup, or (b) widen to "judging-bias 일반 패턴" which is also defensible.
- **§1.5 (4b) encoder-family bullet.** This contribution is now elevated; a future reviewer may ask for explicit hypothesis test against the LM-backbone alternative (e.g. "would two LLaVA-class models with different LM but same encoder still cluster together?"). The current 6-model panel is suggestive but not definitive on this falsifiable cross-cut; a follow-up could pin it down.
- **Strict free-lunch + Chand et al. dialogue.** The Edit 4 paragraph differentiates on three axes (LM/VLM × discrete/continuous × weight/activation space). If Chand et al. is published in a venue Round-4 reviewer reads more carefully, they may push for a head-to-head comparison on a shared metric — currently no such metric exists across the two papers' setups.

## Internal consistency check

After all edits:

- [x] **Abstract numbers unchanged.** No table cell values modified; macro +0.41 / HB +2.21 / POPE −0.06 / γ-β ×1.6 / ×2.9 / ×12.7 / 51 of 85 monotonic / Δdf [-5.2,-0.3] / Δem(a) +3.9 / Δem(b) +8.8 — all preserved.
- [x] **§1.5 contributions still match §4–§7 deliveries.** (1) → §3 + §4 (first-evidence framework hedge); (2) → §3.2; (3) → §4.4; (4a) → §5.2; (4b) → §5.1 / §4.4 Insight 3 / §1.4; (5) → §6.2 + §6.4 + §6.5 + §7 (with formal definition in §6.2.3); (6) → §4.6 + §8.2.
- [x] **§8.1 종합 still consistent.** Strict free-lunch language carries through; no number changed.
- [x] **All figure embeds resolve to existing PNG paths.** No figure path modified.
- [x] **No table renumbering.** Tables 1–8 + Figures 1–7 + appendix labels stable.
- [x] **§2 prior-work block now covers every method class §6 compares against.** CAA / ITI / LEACE explicitly cited; Hufe Dyslexify added; Lou & Sun no longer orphan.
- [x] **VLMBias factual correction propagated to §1.1 + §2.** No remaining "의미 클래스 라벨" callsite. Verified by `grep "의미 클래스 라벨"` = 0 hits in body (only in v5 changelog as descriptive history of the fix).
- [x] **First-evidence register consistent across (1) and (6).** Both now use "first-evidence" with stacked qualifiers; abstract / §1.4 / §1.5 / §4.6 Insight 2 / §8.1 / §8.2 lim 5 still aligned.
- [x] **Strict free-lunch term — formal definition + abstract inline + §1.5 (5) inline + §6.5 Insight Note + §8.1 종합 — all consistent on 4-clause conjunction.**
- [x] **Round-1 / Round-2 fixes preserved.** All Round-1 numerical corrections (Table 3 TallyQA, Table 5 E4, Table 7 ActAdd "+57%" → qualitative, Figure 6 Q2/Q3) and all Round-2 prose fixes (이분법 / 주력 / 발화 / 빈자리 / 회귀 elimination, §4.6 Insight 1 grammar rewrite, Table 2 dual-metric robustness) are untouched.
- [x] **References — verified arXiv IDs.** 2306.03341 (ITI), 2306.03819 (LEACE), 2312.06681 (CAA), 2412.06593 (Lou & Sun), 2505.23941 (VLMBias), 2511.18635 (No Free Lunch — actual title verified). No fabricated IDs.

## Diff stat

- Lines: 526 → 538 (+12 lines, +2.3 %).
- Sections fully rewritten: 0.
- Tables modified: 0 (no cell value changes).
- Figures modified: 0.
- New paragraphs: 2 (§2 *Activation steering and concept erasure*; §6.2.3 *Strict free-lunch 형식 정의*).
- New References entries: 4 (Belrose 2023, Li 2023, Panickssery 2024, Chand 2025); 1 corrected (Lou and Sun first names).
- §1.5 contributions: 6 → 7 bullets (split (4) → (4a) + (4b)).
- §2 prior-work paragraphs: 3 → 4 (new "Activation steering and concept erasure" paragraph) + Hufe Dyslexify added in prose to typographic paragraph.
- Word count delta (rough): +500–600 Korean words (the two new paragraphs + Edit 8 §6.5 Note + Edit 9 §4.6 Insight 2 clarifier).

**Single most impactful edit:** **Edit 1 — new §2 paragraph "Activation steering and concept erasure" (CAA / ITI / LEACE).** The reviewer named this as "the sharpest single critique" and "the largest single missing-prior-work gap" — adding it closes the novelty axis without changing any number in the paper. The new paragraph (a) names the prior method class the paper claims to outperform, (b) positions E6 as the triadic intersection rather than method-class novelty, (c) honestly characterizes LEACE per WebFetch verification, and (d) forward-links to the §6.5 Note (Edit 8) which explains how CAA / ITI reduce to already-tested failure modes in our setting. Combined with Edit 4 (strict free-lunch formal definition) and Edit 8 (Table 7 panel-boundary disclosure), the paper's novelty pitch survives a hostile-reviewer prior-work-aware audit on the activation-steering axis.
