# Round 4 — Aggressive Adversarial Review

**Reviewer persona:** Adversarial. Wants to reject. Attacks every weakness. Hostile PC member who has read the activation-steering literature and the post-Round-3 draft.
**Paper version reviewed:** `docs/paper/emnlp_draft_ko.md` (538 lines, post-Round-3 revision, footer changelog `v5`).
**Date:** 2026-05-09.
**Prior rounds:** round1_methodology + response, round2_writing + response, round3_novelty + response.
**Posture:** Three constructive rounds covered surface-level critiques. My job is to attack what survived. Round-1 fixed numbers, Round-2 fixed prose, Round-3 fixed novelty positioning. None of them attacked the *load-bearing experimental design* of the headline mitigation chain.

## Recommendation

**Recommend reject.** The single most damaging issue is: **every paper-tier mitigation claim — E6 calibration, the 27-cell hyperparameter selection, the 5-dataset cross-evaluation, the strict-free-lunch criterion verification, and the 6-benchmark capability preservation — is on a single model (`llava-onevision-qwen2-7b-ov`). The title says "Vision-Language Models" plural, the abstract says "deployable", and §1.3 / §1.5 (5) imply universality, but the N is 1. The paper acknowledges "Mid-stack cluster 단일" for E4 in §8.2 but never lists "headline E6 mitigation N=1 model" as a limitation.** This is a transparency failure that survives three review rounds because Rounds 1–3 each verified their own axis (numbers / prose / positioning) and never asked "for how many models does this generalize?"

## Strengths begrudgingly acknowledged

1. **The (a − m) paired-inpaint contrast design (§6.2.1 Insight) is genuinely clean.** Same scene, digit pixel removed, captures digit-pixel-specific variance with generic distraction auto-subtracted. I cannot find a way to attack the design itself; it is the strongest single methodological move in the paper.
2. **§5.2 → §6.4 prediction-then-verify chain is tight.** Single-layer ablation null → multi-layer redundancy → single-direction mitigation will fail cross-dataset → empirically verified +56 % LEACE backfire on ChartQA. This is a genuine mechanism-grounded prediction with empirical follow-through, and Rounds 1–3 verified the numbers.
3. **§4.5 L1 monotonicity panel-mean numbers** (43/85 cross_entropy, 51/85 log_prob_sum, +15.6 / +19.1 pp Q4-Q1 gap) survived Round 1's table-by-table audit and are the kind of reproducible finding that a Findings-tier paper rests on.

I will not attack these. Save the fire for the rest.

## Critical attacks (any one of these justifies rejection)

### CRIT-1. **§6 / §7 / §8.1 / abstract — all paper-tier mitigation claims are N=1 model.**

Section after section, the deployable mitigation rests on `llava-onevision-qwen2-7b-ov`:
- §6.2.2: "Main 모델 `llava-onevision-qwen2-7b-ov`을 PlotQA + InfoVQA pooled wrong-base set (N=5,000)으로 calibrate."
- §6.2.2: "27-cell pilot grid... **선택 cell: L* = 26, K = 8, α = 1.0**." All 27 cells on the same model.
- §6.2.3 Table 6: 5 datasets × 1 model.
- §7 Table 8: 6 benchmarks × 1 model.
- §8.1 종합: "5/5 cross-evaluation dataset에서 direction-follow를 줄이는 동시에..." — "5/5" refers to *datasets*, not models. The implicit "deployable mitigation" recommendation is across all VLMs.

The title says **Vision-Language Models** (plural). The abstract says "*배포 가능한* subspace projection mitigation." §1.3 says "**E6 (residual-stream subspace projection, deployable)**". §1.5 (5) "Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며" implies the mitigation generalizes. §8.2 limitation list includes:
- "단일 prompt." (paraphrase)
- "Open-weight 모델만." (closed-weight defuse)
- "Human baseline 부재."
- "Mid-stack cluster 단일." — *for E4* (3 mid-stack models)
- "γ-β N=1 reasoning pair."

It does NOT include "Headline E6 mitigation N=1 model." A reviewer sees "Mid-stack cluster 단일" called out for the *demo* mitigation E4 but no comparable hedge for the *deployable* mitigation E6 even though E6 is N=1 and E4 is N=3.

Why this is paper-killing rather than revision-required: the §6.4 *predict→verify* argument explicitly invokes single-direction methods failing because "per-dataset mean-anchor direction은 측정 가능하게 다른 곳을 가리킨다." The same argument applies between *models*: the mid-stack cluster's L=14–16 peak vs Qwen-ViT's L=22/28 peak vs SigLIP-Gemma's L=5/42 peak (§5.1) means the L=26 hyperparameter chosen for OneVision *cannot* be the right choice for any other archetype's residual stream geometry. The paper's own §5.3 says OneVision peak is dataset-dependent (L=27 on Plot/Tally vs L=14 on Info/VQAv2). If even *within OneVision* the peak migrates by 13 layers across datasets, the L=26 + K=8 + α=1.0 cell *cannot* be expected to work on Gemma3-27b or Qwen2.5-VL-7b without re-tuning. The paper has shipped a single-model proof-of-concept and called it deployable. A hostile reviewer would write: "The deployable mitigation in this paper is a single-model finding; the cross-VLM generalization claim implicit in the title and §1.5 is unsupported."

### CRIT-2. **§6.2.2 27-cell pilot grid — two rounds of reviewers asked, two rounds of authors deferred. The chosen cell is post-hoc selected on a hidden grid.**

Round 1 (should-fix #4): "27-cell pilot grid appendix table — only the chosen cell is shown in body, 26 rejected cells deferred to appendix... Owners + estimates per Round-1 response document. Estimate: 1–2 days."

Round 1 response: DEFER.

Round 3 (no separate flag, but the comparison-baseline gap in §6.5 is the related concern about cherry-picking).

Round 3 response: still deferred. Paper §8.2 deferred list:
> "**§6.2.2 27-cell pilot grid 부록.** 선택 규칙 (em-deal-breaker ≤ −6 pp)은 본문에 명시되어 있으나, 거부된 26 cell의 per-cell `Δdf(a)` / `Δem(a)` heatmap은 후속 부록에 추가."

Three problems:
1. **The "em-deal-breaker ≤ −6 pp" rule is not pre-registered.** §6.2.2 uses the past tense ("rejected"). Was the −6 pp threshold set *before* observing the 27 cells, or *after*? In the absence of a pre-registration document or commit hash, a hostile reviewer assumes post-hoc.
2. **27 cells × 4 metrics (Δdf, Δem(a), Δem(b), Δacc(b)) = 108 observations the paper has not shown.** The chosen cell could be the *only* cell that passes strict free-lunch, or one of 5 that do, or one of 25 with no other cell within 1 pp — the reader cannot tell. The strict-free-lunch claim's Bayesian update from "27 cells produced one survivor" vs "27 cells produced 6 survivors" is qualitatively different.
3. **The selection criterion "em(a) ≤ −6 pp deal-breaker"** is *anchored arm only*. The strict-free-lunch criterion (defined in §6.2.3) requires four clauses including Δem(b) ≥ 0. So the *selection rule* is weaker than the *evaluation criterion* — i.e., the chosen cell could have failed the Δem(b) ≥ 0 clause and the paper would still have selected it (and presumably called it strict free-lunch failure). That the chosen cell *also* passes Δem(b) ≥ 0 across all 5 datasets is then post-hoc luck or post-hoc filtering. Either way the strict-free-lunch criterion was not what selected the cell.

I am not asking for new experiments. I am asking for the 27-cell heatmap that already exists in `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_*` (per Round 1 response). Two rounds of reviewers asked; two rounds of authors deferred. That is no longer "deferred" — it is suppression of evidence that bears on the headline claim.

### CRIT-3. **§6.3 b-arm em +8.8 pp — the post-hoc explanation is one of three equally consistent stories.**

§6.3 reports b-arm (target_only, no anchor) exact-match rises +8.8 pp on average across 5 datasets after the L=26 / K=8 / α=1.0 hook is applied. The paper's explanation:
> "(a − m) calibration contrast는 wrong-base 부분집합에서 *digit-anchor arm을 visual-matched no-digit control과 구별하는 모든 variance direction*을 capture한다. 정의상 digit의 representational signature가 포함되지만, *wrong-base의 digit-anchor failure와 co-aligned된 어떤 error mode*도 함께 포함된다. K = 8 leading subspace projection은 둘 다 제거하며, b-arm em 이득은 그 방향들 중 일부가 *target_only arm에서 amplitude가 em을 억제하던 generic wrong-base error mode*를 운반했음을 드러낸다."

This is a story constructed to explain a positive surprise. Three competing stories the paper has not ruled out:

**Alternative 1 — General regularization.** Removing any K=8 dim subspace from a 28-layer / 4096-dim residual stream is, with K/d ≈ 0.002, plausibly a mild regularizer that shifts the answer-token distribution toward the modal correct digit. The paper does not control with a *random K=8 subspace* (e.g., K=8 SVD on the residual stream of a non-anchor calibration set, or K=8 random orthogonal subspace) at L=26. If the b-arm em gain reproduces with random projections, then the (a − m) contrast is not load-bearing for the b-arm gain.

**Alternative 2 — Mode-collapse to most-frequent-digit.** Removing 8 dims at L=26 may collapse the answer-token logit distribution toward the dataset-modal answer (0, 1, 2, 3 are all common ground-truths in counting / chart / math VQA). If the b-arm gain is concentrated on the *low-information* questions whose modal answer is correct by chance, this is not "wrong-base error mode debiasing" — it is statistical regression-to-mode. Test: report b-arm em gain decomposed by GT-mode-frequency tertile.

**Alternative 3 — Implicit hyperparameter overfitting on the deferred 27 cells.** If the L=26 / K=8 / α=1.0 cell was selected (per CRIT-2) on PlotQA + InfoVQA pooled with em-deal-breaker = −6 pp, and the b-arm em was *implicitly observed* during selection, then the chosen cell is selected partly on b-arm em — making the 5-dataset b-arm em gain a partial training-set artifact, not an independent observation. Test: pre-register the cell, then test on held-out datasets. The paper's "1회 보정 후 5 dataset 보편 적용" framing implies pre-registration; the deferred 27-cell grid says it was post-hoc.

§6.3 Insight 2 says "b-arm 양 결과가 §7 검증의 *사전 신호*였고, §7에서 ... *확인*된다." This is circular: §6.3 explains the b-arm em bonus by appealing to a story; §7 then "confirms" the story by showing the same hook also helps HallusionBench. But HallusionBench helping (Alternative 1: random K=8 helps illusion) is also consistent with general regularization. The mechanism story has *not* been distinguished from the regularization story.

This matters because the **abstract** sells the +8.8 pp b-arm em gain as part of the strict-free-lunch claim ("anchored arm + non-anchored arm 양쪽 모두에서 exact-match 상승"). If the b-arm gain is a regularization artifact, the strict-free-lunch criterion is met for trivial reasons; if it is anchor-mechanism-specific, it is the surprising finding the paper sells. The paper has not run the experiment that distinguishes these.

## Major attacks (multiple of these compound to rejection)

### MAJ-4. **§6.2.3 paired-sids small-n cells — Round 1 flagged, Round 1 response DEFER, Rounds 2–3 silent. The "5/5 dataset df reduction" claim does not survive paired-bootstrap CI.**

Round 1 (MAJOR-9): "§6.2.3 paired-sids small-denominator cells (n=170 MathVista, n=224 ChartQA) report no CIs. With n=170 paired and the underlying df rate ~0.20, the 95 % bootstrap CI half-width on Δ df is approximately ±0.06."

Round 1 response: DEFER ("Owner: paper author. Estimate: <1 day.").

Round 2 / Round 3: did not re-raise.

The paper's Round-3 §8.2 deferred list:
> "**§6.2.3 small-n cell CI 미보고.** 5/5 paired-sids 표 중 2 cell (n=170 MathVista, n=224 ChartQA)이 점추정 (point estimate)만 보고."

But it is not 2 cells. Look at Table 6:
- TallyQA n=4,978 → CI plausibly tight, Δdf=−0.3 pp (but is this ≠ 0?)
- PlotQA n=2,306 → Δdf=−5.2 pp, plausibly significant
- **InfoVQA n=443 → Δdf=−0.7 pp** — at p ≈ 0.20, half-width on Δ df ≈ ±0.04 (Wilson-style paired). Δdf=−0.7 pp is *within* the CI half-width — i.e., **indistinguishable from zero.**
- ChartQA n=224 → Δdf=−4.0 pp, half-width ≈ ±0.06; borderline.
- MathVista n=170 → Δdf=−4.1 pp, half-width ≈ ±0.07; borderline.

Realistic claim under bootstrap CI: **PlotQA significant, TallyQA inconclusive (small magnitude), MathVista + ChartQA borderline, InfoVQA noise.** That is 1/5 with strong evidence, 2/5 borderline, 2/5 inconclusive — not the "5/5 dataset df reduction" headline. The paper sells "5/5 dataset에서 다음 세 성질이 동시에 유지된다 — (1) df 5/5 감소" as the §6.2.3 headline (line 299); under paired bootstrap CI this is not defensible.

Worse: the paper *itself* uses paired bootstrap CI for §7 capability preservation (Round-1 response added the methodology). Why is the same procedure run for §7 (where the threshold is met) and not for §6.2.3 (the headline mitigation result)? The reader's read: cherry-picked statistical rigor.

### MAJ-5. **§6.5 CAA/ITI as a Round-3 Note instead of Table 7 rows — claiming victory without empirical comparison.**

Round 3 reviewer offered: "either run CAA / ITI as additional Table 7 rows, or add §6.5 footnote explaining the reduction." Round 3 response chose footnote:
> "*Note (CAA / ITI baseline 처리):* CAA [Panickssery et al., 2024]는 paired-contrast residual-stream steering을 *rank-1*으로 가중 합산하므로 본 비교의 ActAdd 행 ((a − m) calibration의 rank-1 mean-direction 인스턴스)과 *구조적으로 동치*이며 cross-dataset α=1 self-test backfire를 동일하게 상속한다. ITI [Li et al., 2023]는 *attention-head 출력*에서 multi-direction 개입을 수행하므로, §5.2의 *single-layer attention ablation null* + *multi-layer redundancy* 결과가 attention-head 수준 single-locus 개입의 cross-dataset 실패를 사전 예측한다."

The "CAA = ActAdd at K=1 on (a − m)" reduction is hand-wavy:
1. **CAA** (Panickssery 2024) computes the steering vector as the *mean of paired contrastive activation differences*, then adds α · v to the residual stream. Specifically: `v = mean_i (h_pos[i] − h_neg[i])`, applied as `h ← h + α v`.
2. **ActAdd** (Turner 2023) computes the steering vector as the *single-pair* difference at one layer, *or* averaged over multiple pairs, then adds with a coefficient.

The paper's "(a − m) ActAdd row" presumably uses some calibration on (a − m) pairs. CAA's distinguishing characteristic is the *paired* contrast across many examples; ActAdd's original formulation was single-pair. Whether the paper's ActAdd implementation is in fact "CAA at K=1" depends on the implementation details, which the paper does not describe (Round 1 axis B Baselines: "Insufficient hyperparameters. No specific α, layer, K, calibration-set descriptions for any of the 5 baselines"). Round 1 response did not fix this.

So the §6.5 Note's claim "CAA = ActAdd at K=1 on (a − m), structurally equivalent" is a reduction asserted without (a) implementation detail of the ActAdd baseline, or (b) empirical verification that they produce the same Δdf curves. This is "claim victory without comparison." A hostile reviewer who has actually implemented CAA and ActAdd will know the activation spaces, calibration counts, and per-layer aggregations are *different operations* even when the API is similar.

For ITI the reduction is even weaker. ITI operates on attention-head outputs (per-head direction selection by truth-classifier accuracy). The §6.5 Note says "§5.2 single-layer attention ablation null predicts ITI's failure." But §5.2's ablation *zeros* attention; ITI *adds* a direction to attention output. These are different operations with different mechanisms. §5.2's null-ablation result is consistent with ITI working *or* failing.

The right move is to run the rows. Round 3 estimated 4–8 H100-hours for CAA, 1–2 days for ITI. That is not "deferred to next revision" — that is a one-week delay before submission. The paper's headline "**multi-direction subspace projection** is the only candidate that clears strict free-lunch" is a comparative claim made against an incomplete baseline panel.

### MAJ-6. **§7 multiple-comparisons correction not applied. "Sub-pre-registration" is asserted but the registry is not cited.**

§7 Table 8 reports 6 benchmarks. The strict-free-lunch criterion in §6.2.3 requires Δ ≥ −0.5 pp on macro and ≥ −1.0 pp per-benchmark. §7 reports HallusionBench Δ=+2.21 pp 95 % CI [+1.14, +3.28] *excludes zero*; POPE Δ=−0.06 pp 95 % CI [−0.21, +0.09] *pinned to zero*. Both are presented as significance claims.

Issues:
1. **No multiple-comparisons correction.** 6 benchmarks × {baseline drift, mitigation effect} = 12 paired tests; just for the mitigation effect direction, 6 tests. Bonferroni at α=0.05 demands per-test α=0.0083, two-sided z≈2.64 instead of 1.96, which on the HB SE of (3.28−1.14)/3.92 = 0.546 pp gives a Bonferroni-corrected CI of [+0.77, +3.65] — still excludes zero, so HB *survives* correction, but the paper does not show this. POPE's CI [−0.21, +0.09] would widen to roughly [−0.25, +0.13] — still pinned to zero, but again the paper does not show. Reviewer will not give the paper the benefit of *not running* the correction even when it would survive.
2. **The "사전등록" claim is not backed by a registry.** §6.2.3 Strict free-lunch 형식 정의 says "*Δ(held-out capability macro)* ≥ −0.5 pp (사전등록, §7)." §7 says "사전등록 임계: 벤치마크별 Δ ≥ −1.0 pp, 매크로 Δ ≥ −0.5 pp." A pre-registration is a document — OSF, AsPredicted, GitHub commit before the analysis — with a *timestamp* before the result was observed. The paper does not cite such a document. "사전등록" without a registry is just "we picked this threshold." A hostile reviewer reads this as either a misuse of the term *pre-registration* or hidden flexibility.
3. **Strict free-lunch's "Δ(held-out capability macro) ≥ 0" was tightened to "≥ −0.5 pp" between abstract and §6.2.3.** Abstract says "Δ(held-out capability) ≥ 0." §6.2.3 formal definition says "Δ(held-out capability macro) ≥ −0.5 pp." That is a 0.5 pp loosening. With macro Δ = +0.41 pp, the chosen cell would *fail* the abstract's stricter "≥ 0" only if held-out gain were below 0, but the *form* of the criterion is inconsistent between callsites.

### MAJ-7. **§4.6 γ-β still presented as paper-tier in §1.5 (6) and §8.1 despite N=1 architecture × N=1 dataset.**

Round 2 softened the abstract closing from "입증" to "first-evidence VLM 결과 (단일 architecture pair, cross-architecture 일반화는 §8.2 한계로 명시)." Good. But:
- §1.5 (6) still treats γ-β as a contribution: "(6) γ-β reasoning pair (N=1 Qwen3-VL Instruct vs Thinking)에서 reasoning-amplifies-anchoring을 처음 보이는 *first-evidence* VLM 결과."
- §8.1 종합 still treats it as a finding: "reasoning mode는 효과를 *증폭*한다 (§4.6) — reasoning trace에서 bias가 *축적*된다는 것을 시사하는 first-evidence VLM 결과이다."
- §4.6 itself uses MathVista only (one dataset), with Insights 1, 2, 3 each generalizing across H2 asymmetry mechanism, LRM literature alignment, and accuracy trade-off.

The data: 1 architecture pair × 1 dataset × 1 stratum (single-stratum γ-β setup per §G appendix). N=1 in all three of (model, dataset, stratum) is *not* a paper-tier finding; it is an *existence proof* that warrants a workshop paper or a Findings short paper. The §4.6 *Insight 1* claim "메커니즘 결합 — confidence-modulated anchor pull 가설과 일치" rests on this single-cell ratio (×12.7) — and Round 2 softened "직접 입증" to "강하게 뒷받침" but did not remove the framing.

The §8.2 limitation list says "γ-β N=1 reasoning pair." The paper hedges *for* N=1 but treats the result as a contribution. A hostile reviewer's read: the paper is borrowing tier from §6 (mitigation chain) to elevate γ-β from existence-proof to paper-tier. If §6 is N=1 on the model axis (CRIT-1), then γ-β is the second N=1 finding in the same paper. Two N=1 findings stacked into one Main paper.

## Minor attacks (would be revision-required even individually)

### MIN-8. **"Strict free-lunch" is a marketing term whose Δem(non-anchored) ≥ 0 clause is suspicious.**

§6.2.3 Strict free-lunch 형식 정의 says:
> "통상적인 Pareto-improvement 기준은 첫 번째 + 마지막 두 clause만 요구하며 두 번째 *non-anchored arm em* 조항을 가지지 않는다. 이 추가 clause는 bias mitigation의 *cross-category collateral damage* — Chand et al. [2025]가 LM debiasing에서 보고한 *31.5 % 비표적 dimension에서의 부수 손상* — 에 직접 대응하는 screening 기준으로..."

Question: why does anchor mitigation require *improving* baseline (no-anchor) accuracy? The motivating cross-category collateral damage from Chand et al. would be addressed by Δem(non-anchored) **≥ 0** (i.e., no harm), not by Δem(non-anchored) > 0. The paper's actual data shows Δem(b) = +8.8 pp average — not "no harm" but positive. The "*strict*" framing is built around a celebration criterion: the paper found a cell where Δem(b) was positive, then defined "strict free-lunch" to include "Δem(non-anchored) ≥ 0" as a clause that the chosen cell trivially passes. The criterion is post-hoc shaped to fit the result.

Compare to the genuine no-harm criterion: Δem(b) ≥ −ε for some pre-registered ε. That would be the operationally meaningful screening rule. The paper's choice — Δem(b) ≥ 0 — is rhetorically stronger and selectively favorable to the chosen cell.

### MIN-9. **§4.4 Insight 2 "Anti-scaling" claim from one model pair × 3/4 datasets.**

§4.4 Insight 2 says Gemma3-4b > Gemma3-27b on PlotQA, ChartQA, MathVista (anti-scaling, 4B more anchor-pulled than 27B), but reverses on InfoVQA (4B < 27B). Round 2 added the qualifier "anti-scaling이 chart/plot/math 3개 dataset에 한정되며 InfoVQA에서는 표준 scaling 회복."

The data is one model-family pair (Gemma3 4B vs 27B) on 3/4 directional. "Anti-scaling" is a strong term that originated in language modeling (e.g., inverse scaling tasks, McKenzie et al. 2023) where multiple model families show the same trend. Calling 4B-vs-27B-on-3-of-4-datasets "anti-scaling" is inflated. A hostile reviewer:
- One Gemma3 family-pair is N=1 architecture-pair on the scaling axis.
- 3/4 directional consistency at N=1 is suggestive, not established.
- Other model families in the panel (Qwen2.5-VL 7b vs 32b; OneVision 7B alone) are not analyzed for anti-scaling. Why? Either show the cross-family analysis or drop "anti-scaling."

### MIN-10. **§A.2 FLUX seed not reported. Round 1 should-fix #9, deferred. Stimulus inventory conditioned on unspecified seed.**

Round 1 noted: "Seeds for FLUX rendering: reported? ... §A.1 reports greedy decoding (no seed needed for inference) but does *not* state seeds for FLUX rendering. The 128-image inventory in §A.2 was generated by `scripts/generate_irrelevant_number_images.py` — what seed? This affects the (a) inventory specifically, which is the load-bearing stimulus for every paper claim."

Round 1 response: "DEFER ... seed lookup pending."

Round 3 still does not report it. Three rounds of reviewers, no FLUX seed. *Every* paper claim is conditional on this 128-image inventory. A hostile reviewer who tries to reproduce the paper cannot regenerate the inventory.

### MIN-11. **"다층 중복 (multi-layer redundancy)" reframable as "single-cluster mechanism with three archetype outliers."**

§1.4 abstract: "**single-layer mask ablation은 6-model 메커니즘 panel ... 6/6 null** — signal은 multi-layer redundant이다."

The 6-model panel:
- gemma4-e4b → SigLIP-Gemma early (L5/42)
- llava-1.5 → CLIP-ViT mid-stack (L14-16)
- ConvLLaVA → ConvNeXt mid-stack (L14-16)
- InternVL3 → InternViT mid-stack (L14-16)
- qwen2.5-vl → Qwen-ViT late (L22/28)
- fastvlm → FastVLM late text-stealing (L22)

That is 3 models in the mid-stack cluster + 3 archetype outliers (1 SigLIP-Gemma early, 1 Qwen-ViT late, 1 FastVLM late). The panel is *not* uniformly multi-layer redundant; it is one cluster of three + three singletons.

The "multi-layer redundancy" framing covers the case where you ablate one layer and signal stays. But the *reason* it stays differs across archetypes:
- Mid-stack cluster: signal is genuinely distributed across L14-16 layers (multi-layer in the sense of "redundant copies").
- SigLIP-Gemma early: signal is at L5 *and* L42 — bimodal, not "redundant copies" but two distinct loci.
- Qwen-ViT late: L22/28 bimodal (verified via cross_dataset_peaks.csv).
- FastVLM: cross-dataset L17/22/23/27 — peak migration, not redundancy.

A hostile reviewer reframes: "The 6/6 null result is consistent with three different mechanisms — multi-layer redundancy in mid-stack, bimodal peaks in SigLIP-Gemma and Qwen-ViT, and per-dataset peak migration in FastVLM. The paper's 'multi-layer redundancy' framing assumes a single mechanism explains all six." This is not paper-killing, but it weakens §5.2 → §6.4 "predict single-direction failure" because the prediction's mechanism story differs across archetypes.

## Attack-by-axis

### A. Overclaim hunt

| Section | Claim | Closest contesting prior work or correction |
|---|---|---|
| Title | "Vision-Language Models" (plural) | Headline mitigation is on **one** model. Should say "in LLaVA-OneVision" or "(case study on one architecture)." |
| Abstract / §1.3 | "deployable" | N=1 model is not deployable across architectures without re-tuning. |
| Abstract | "5/5 evaluation dataset의 paired-sids wrong-base 부분집합에서 Δdf(a) ∈ [−5.2, −0.3] pp" | Without paired bootstrap CI, InfoVQA Δdf=−0.7 pp is noise floor (MAJ-4). Realistic: 1/5 strong, 2/5 borderline, 2/5 inconclusive. |
| Abstract | "***strict free-lunch*** ... 5개 비교 baseline ... 중 *유일하게* 통과" | Within 5-baseline panel only; CAA/ITI not run (MAJ-5). The "유일" bound is panel-internal and is not the universal claim the headline implies. |
| §1.5 (1) | "*first-evidence* 평가 프레임워크" | Defensible after Round-3 correction; no attack here. |
| §1.5 (4b) | "encoder family에 의해 결정됨" | Defensible at the panel level; cross-family validation (e.g., 2 SigLIP-Gemma models, 2 Qwen-ViT models) absent. Suggestive, not established. |
| §1.5 (5) | "*예측한 뒤 multi-direction subspace projection으로 우회*" | Predict-verify chain on N=1 model. Generalization claim implicit in "deployable" is unverified. |
| §1.5 (6) | "처음 보이는 *first-evidence* VLM 결과" | Hedge present but contribution-tier framing inappropriate for N=1 architecture × N=1 dataset (MAJ-7). |
| §1.4 | "다층 중복 (multi-layer redundancy)" | Mixed mechanism across 6 archetypes; "multi-layer redundancy" cleanly applies to mid-stack cluster only (MIN-11). |
| §6.2.3 | "5/5 dataset df 5/5 감소" | InfoVQA Δdf=−0.7 pp on n=443 indistinguishable from zero without CI (MAJ-4). |
| §6.5 Insight | "유일 후보" (unique candidate) | Within 5-baseline panel only; CAA/ITI reduction asserted not run. Round-3 response added "5-baseline panel 내부의 비교" qualifier — good — but the §6.5 Insight closing line says "유일 후보 — *dataset 간 shared variance direction* + *inference 시 weight 보존*이 동시 충족된다" without re-asserting the panel boundary. |
| §7 Insight | "VLM hallucination의 일부가 본 논문 anchoring mechanism과 representation space를 공유" | Hedged with "시사" — appropriate. No attack. |
| §8.1 | "**세 gate의 conjunction**" | OK as a behavioral synthesis claim. |

### B. N=1 / cherry-pick hunt

- **§6.2.2 chosen cell L=26 K=8 α=1.0 from 27-cell grid.** See CRIT-2.
- **§6 / §7 / §8.1 entire mitigation chain on N=1 model.** See CRIT-1. This is the deepest cherry-pick: not just one cell from a grid but one model from a population of 6 (the main panel).
- **§4.5 worked example E5c VQAv2 LLaVA-Interleave S1 cross_entropy.** Round 1 caught the Q2/Q3 numerical errors and Round 1 response fixed them. The remaining N=1 issue: this is one specific (experiment × dataset × model × stratum × proxy) cell. Round 1 axis D noted: "Generalisability is claimed via the 85-cell mean +15.6 / +19.1 pp, which is the right move, but readers should be cued that the 23 pp number is one cell, not the panel mean." Paper still uses the 23 pp single-cell number in Figure 6 caption to dramatize the gradient.
- **§4.6 γ-β single architecture × single dataset.** See MAJ-7.
- **§4.4 anti-scaling claim from Gemma3 family-pair only.** See MIN-9.
- **§3.3 dataset n** — Round 1 fixed the raw-vs-per-cell n issue, but per-cell n on ChartQA goes as low as 129 (qwen2.5-vl-32b) and TallyQA as low as 6,934. Effective N for the headline 5-dataset result depends on which model's per-cell denominator was smallest in any computation; not currently surfaced.

### C. Confound hunt

- **§6.3 b-arm em +8.8 pp post-hoc explanation vs general regularization.** See CRIT-3 (three alternative stories).
- **§7 HallusionBench Δ=+2.21 pp.** Same general-regularization confound. The paper's §6.3 Insight 2 forward-points to §7 as confirmation of the b-arm error mode story; but if K=8 random subspace projection at L=26 also helps HallusionBench (an empirical question), then HB gain is regularization, not anchor-mechanism-specific.
- **§4.3 PlotQA un-mitigated free-lunch (em(a) > em(b) for 6/7 models).** Paper §4.3 Insight 3 frames as positive finding ("이 패턴은 InfoVQA로 일반화하지 *않으며*") and §6.2.3 picks PlotQA as part of the calibration set. The S1 cutoff (anchor within ±10 % of GT for stratified data) is the design choice that *creates* this artifact. PlotQA's S1 stratum is anchors at ≤10 % of GT — i.e., the anchor *is* a plausible answer. Of course em(a) > em(b) on S1: the model is being shown a plausible answer hint. The paper acknowledges this somewhat ("S1 cutoff가 anchor를 GT의 ±10 % 안에 두므로 anchor를 '그럴듯한 추측 단서'로 픽업") but then *uses this PlotQA strength* in the calibration set for §6.2 SVD. Calibration on a stratum where anchor is the right answer to subspace-extract "anchor variance" risks confounding "GT-amplitude" variance with "anchor-presence" variance. The (a − m) contrast subtracts this only if masking the digit pixel removes the GT-amplitude signal too (which it should), but the paper does not verify on a no-anchor-near-GT calibration alternative.
- **§5.2 single-layer ablation null — were the right layers ablated?** Per E.2 table, modes are `ablate_peak / ablate_peak_window / ablate_lower_half / ablate_upper_half / ablate_all`. The peak ± 1 window is 3 layers. What about a *random* 3-layer ablation as a null comparison? If random 3-layer ablation also gives 6/6 null (because any 3 out of 28 layers are individually ablation-tolerant), then the null result is *not* about peak vs non-peak but about LM resilience to single-layer dropout. The paper's "single-layer null → multi-layer redundant" inference depends on the specific peak ablation being non-trivial; a random-layer null comparison would tighten this.

### D. Negative result suppression hunt

- **27-cell pilot grid: only chosen cell shown (CRIT-2).** Two rounds asked, two rounds deferred. Suspect there are runner-up cells in the grid that *also* pass strict free-lunch but were not chosen for some other reason (smaller magnitude? worse on a particular dataset?). The paper presents one cell as if it were the unique survivor; the 26 rejected cells determine whether this is true.
- **CAA / ITI not run (MAJ-5).** Round 3 deferred, response asserts structural reduction. If a hostile reviewer ran CAA at K=1 on (a − m) and found it failed for the same reason as ActAdd (which the §6.5 Note predicts), this would support the paper. But no such verification exists; the reduction is hypothesized.
- **OneVision E1d analyzer fix pending.** The 6-model mech panel single-layer ablation is 6/6 null on the canonical models, but OneVision Main (the model the entire mitigation chain runs on) is excluded from §5.2 because the analyzer has a 0.000 baseline bug. So the paper's mechanism claim ("single-layer null → predict single-direction failure → motivates §6.2 multi-direction subspace") *does not include the model the §6.2 mitigation runs on.* §5.3 partially addresses by reporting OneVision peak is dataset-dependent, but the full single-layer ablation result on OneVision is missing. A hostile reviewer: "Has the analyzer bug been fixed yet? If so, what does the OneVision single-layer ablation actually show? If it shows partial reduction (not 6/6 null), the multi-layer redundancy claim weakens for the very model the mitigation runs on."
- **E4 attention re-weighting on Main `llava-onevision`** is implied by the paper's "two complementary mitigations" framing (E4 + E6) but §6.1 reports E4 only on the 3-mid-stack-cluster panel (LLaVA-1.5 / ConvLLaVA / InternVL3). E4 was *not* run on OneVision Main. So the "two complementary mitigations" narrative compares an E4 panel-of-3 (not including OneVision) with an E6 panel-of-1 (only OneVision). They are not run on the same model. A hostile reviewer: "If E4 doesn't work on OneVision, the 'complementary' story is a single-model E6 plus a different-model-family E4. If you ran E4 on OneVision and it did work, why not show it? If it didn't work, why isn't that in the paper?"

### E. Statistical rigor

- **§7 multiple-comparisons correction not applied.** See MAJ-6.
- **§6.2.3 paired-bootstrap CI not applied.** See MAJ-4.
- **§4.5 monotonicity 51/85 reported as count, not as proportion with CI.** Round 1 axis A noted: "With 85 trials and a per-cell pass rate of ~60 %, a 95 % bootstrap CI on the 'fraction monotone' estimate would be approximately [50 %, 70 %]." The paper still does not report this. The 60 % headline is likely between 50 % and 70 % — i.e., the paper's "majority of cells monotone" claim is right but at the borderline of the CI lower bound.
- **§4.1 Table 2 7-model panel n=17,730 each.** Paired sample-level CIs on the wrong-correct gap +6.9–19.6 pp range not reported. With per-model n ≈ 17,730 the CIs would be tight, but absent reporting, a reviewer cannot verify "7/7 모델" claim CIs.
- **§4.6 γ-β ratios ×1.6 / ×2.9 / ×12.7.** No CI on the ratios; ×12.7 on n=385 with denominator (correct-base subset) of unknown size could have a wide CI. The MathVista per-cell CSV would show the underlying counts; paper does not surface them.

### F. Reproducibility

- **§A.2 FLUX rendering seed not reported (MIN-10).** Three rounds, three deferrals.
- **HF model commit hashes not pinned.** Round 1 Reproducibility F: "EMNLP camera-ready will need this." §A / §3.3 lists HF model IDs (`llava-onevision-qwen2-7b-ov`, etc.) but no commit hashes / revision SHAs.
- **PlotQA + InfoVQA pooled was the 4th-or-later method-development run.** Round 1 noted: "Were the *earlier* runs ablated?" Round 1 response did not address. The paper presents PlotQA + InfoVQA pooled as the *one* calibration choice; in fact the choice was iterated over (per `E6-tally-only-rerun-tracker.md` indicating multiple calibration attempts). A hostile reviewer: "Why this calibration set? What happened on Tally-only / Chart-only / Tally + Chart calibration?" The §6.2.3 cross-evaluation claim's strength depends on whether PlotQA + InfoVQA pooled was *the first* calibration tried (one shot) or *the best* of several (cherry-picked).
- **27-cell grid not surfaced (CRIT-2).** Rejected cells unavailable.
- **Bootstrap CI procedure for §7 cited Round-1-added paragraph.** Good; one of the few clean reproducibility wins.

### G. Missing comparisons

- **CAA / ITI empirical rows.** See MAJ-5.
- **Random K=8 subspace as null baseline for §6.3 b-arm em gain.** See CRIT-3 Alternative 1.
- **K=8 subspace from non-anchor-task calibration (e.g., POPE, RealWorldQA).** Tests whether the (a − m) contrast specifically is what cleans up b-arm em or whether any K=8 SVD on residual stream at L=26 helps. CRIT-3 Alternative 1 stronger version.
- **Finetuning on debiased data.** Not run, not mentioned. A reasonable baseline.
- **CFG (classifier-free guidance) pruning.** Not run, not mentioned. Active VLM intervention literature uses this.
- **CoT prompting "ignore the second image."** Lou & Sun 2024 reports CoT-style mitigations are *insufficient* for LLM anchoring; the paper cites this in §2 as motivation for representation-level approach but does not run CoT-VLM as an explicit baseline. A reviewer would want to see the CoT-VLM number to confirm it's also insufficient on numerical anchoring.
- **RLHF rejection / instruction-tuning on anchor-resistance.** Not run. Out of scope but should be acknowledged.

### H. Theoretical depth

- **§6.4 K=8 sweet spot — empirical sweet spot, no theory.** Insight 2 says "더 작은 K (2, 4)는 anchor signal을 더 누출, 더 큰 K (16)는 non-anchor variance를 제거 시작." This is empirical observation across the 27-cell grid (which is not shown). No theory predicts K=8. No connection to the residual stream's intrinsic dimension or the (a − m) signal's effective rank. A Main-tier paper on subspace projection would include either (i) eigenvalue spectrum of D[:, L*, :] showing rank-8 elbow, or (ii) ablation across α with fixed K showing the bias-variance trade-off curve. Neither is in the paper.
- **§5.2 multi-layer redundancy not formalized.** "Signal은 multi-layer redundant" is asserted prose. What is the formal claim? Distributed in the sense of multiple layers each carrying ≥ ε mass? Or compositional, requiring the conjunction of layers to produce the effect? The paper's §6.4 prediction "single direction fails because signal lives in different layers across datasets" is a *cross-dataset* mechanism statement, not a multi-layer redundancy statement. The two are conflated.
- **Subspace projection assumes linear anchor representation.** No nonlinear (kernel SVD, autoencoder bottleneck) attempt. Defensible scope choice but not theoretically motivated.

### I. Framing

- **"Strict free-lunch" is rhetorical inflation (MIN-8).** The Δem(non-anchored) ≥ 0 clause is a celebration criterion shaped around the chosen cell's positive surprise.
- **"Three-gate signature" / "세 gate의 conjunction" (§1.2 / §8.1).** Reframe attack: are the three gates (uncertainty / plausibility / digit-pixel) genuinely *one* signature or three independent observations? The paper does not test the *conjunction* — i.e., does removing any one gate eliminate the effect, or do the gates each provide a separable contribution? §4.2 Insight 1 (adopt vs df separation) suggests adopt and df measure different gates, but the "conjunction" framing implies all three must hold. The data supports each gate individually; "conjunction" is a stronger claim.
- **"VLM-first reasoning amplification" — is this paper-tier or follow-up bait? (MAJ-7).** N=1 architecture × N=1 dataset is an existence proof.
- **"Multi-direction × residual-stream × paired-inpaint × strict free-lunch combination" (§6.5 Insight closing).** Round 3 reframed differentiator from method-class novelty to a four-axis combination. Defensible, but the *uniqueness* of this combination depends on (a) actually running CAA / ITI to verify they don't accidentally also hit this combination (MAJ-5), and (b) the "strict free-lunch" being a real criterion not a marketing label (MIN-8).
- **"Capability preservation" from 6 benchmarks.** §7 macro Δ=+0.41 pp. The 6 benchmarks were chosen as held-out. But "capability preservation" implies the mitigation does not harm the model on *any* downstream task; 6 benchmarks is a sample. RealWorldQA / OCRBench / HallusionBench / MMStar / MMBench-DEV-EN / POPE skews toward perception-/illusion-/grounding-style tasks. Reasoning-heavy benchmarks (MMMU, MathVista — which is *in* the headline panel — and AI2D) are not in the held-out set. The "capability preservation" framing is partially supported; reframe to "perception-style benchmark preservation."

### J. Venue fit (Main vs Findings)

The paper currently positions for EMNLP Main. After Round 3, the novelty positioning is defensible at the Main bar *if* the experimental design holds up. My read after stress-testing:

- **§5 mechanism vs Weng et al. EMNLP 2024 Main.** Round 3 reframed Weng comparison to "mechanism→mitigation chain" similarity rather than "we go further." Honest. But Weng's mechanism analysis used causal mediation with quantified mediation fractions; §5 here uses ablation null + per-model archetype labeling. Weng's mediation analysis is more quantitatively grounded; §5 is more taxonomic. *Comparable* mechanism depth but different style.
- **N=1 model on the headline mitigation.** A Main-bar "deployable mitigation" paper either (a) demonstrates on multiple architectures, or (b) explicitly scopes to one architecture and titles accordingly. This paper does neither. Title pluralizes "Vision-Language Models" while shipping N=1 deployable.
- **Statistical rigor below Main bar.** No multiple-comparisons correction (MAJ-6), no paired bootstrap CI on the headline 5-dataset deltas (MAJ-4), 27-cell grid suppressed (CRIT-2). Findings papers can ship with these gaps; Main papers cannot.
- **Genuine surprises.** Encoder-family-determines-archetype (§4.4 Insight 3 / §1.5 (4b)), reasoning-amplifies-anchoring (γ-β), mid-stack cluster's robustness scaling. These are Main-tier surprising claims *if* the experimental support for each holds; per the analysis above, each is N=1 or cherry-picked to varying degrees.

**Honest verdict:** This paper is **strong Findings**, not Main. Main rejection is highly likely if reviewed by anyone with mitigation literature awareness; Findings acceptance is plausible if the deferred items (27-cell grid + paired CIs + CAA/ITI rows + N=1 model acknowledgment) are surfaced. The paper has a real story and clean (a − m) design; it does not have Main-tier experimental scale on the headline mitigation chain.

## What would the authors need to do for me to switch to weak accept?

Realistically: this is reject as-is. To move to weak accept (with revisions), the four blocking items:

1. **Surface the 27-cell pilot grid** as an appendix table or heatmap (CRIT-2). Pre-commit the em-deal-breaker selection rule with a dated commit hash showing the rule was set before observing the grid. Without this, the chosen-cell finding is post-hoc.
2. **Add paired bootstrap CI to §6.2.3 Table 6** (MAJ-4). Use the same procedure §7 already has. Re-state the headline: "5/5 dataset df reduction (PlotQA significant, ChartQA + MathVista borderline-significant, TallyQA + InfoVQA inconclusive)" or similar.
3. **Run CAA at K=1 on (a − m) and ITI on attention-head outputs as additional Table 7 rows** (MAJ-5). Round 3 estimated 4–8 H100-hours for CAA, 1–2 days for ITI. The paper's "유일 후보" claim demands these be empirical, not asserted-by-reduction.
4. **Acknowledge the N=1 model on headline E6 mitigation in §8.2 limitations** (CRIT-1). Either retitle to "case study on LLaVA-OneVision" or add a §8.2 bullet explicit about model cardinality. Currently the paper hedges every other axis but not this one.

Even with these four, I would still want to see CRIT-3 (random-subspace null comparison for b-arm em) before going to Main accept. Findings accept is achievable with the four above.

If the authors do *none* of these and refile with the §1.5 (1) softening + §2 activation-steering paragraph + Round 1/2 corrections only: hard reject from me.

## Final

**Decision:** **reject**.
**Confidence:** **high**.
**One-line:** *The paper's deployable mitigation, capability-preservation, and 'strict free-lunch' headline are all on N=1 model with 27 hyperparameter cells of which only the chosen one is shown and 2/5 evaluation cells reported without confidence intervals — this is not a deployable mitigation, it is a single-cell finding with three concurrent transparency deferrals.*
