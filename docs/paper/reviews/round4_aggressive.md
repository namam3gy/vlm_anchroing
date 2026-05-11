# Round 4 — Aggressive Adversarial Review

**Reviewer persona:** Adversarial. Wants to reject. Attacks every weakness that survived three constructive rounds. Hostile PC member with mechanism-interpretability + steering-vector + EMNLP-Main calibration.
**Paper version reviewed:** `docs/paper/emnlp_draft_ko.md` @ 826 lines (post Round-1 methodology + Round-2 writing + Round-3 novelty surgery, v11).
**Date:** 2026-05-11.
**Prior rounds:** round1_methodology + response (CRIT-1 framework prospectivity, CRIT-2 abstract free-lunch overclaim, CRIT-3 experiment-log surgery, MAJ-1..MAJ-10), round2_writing + response (CRIT-W1/W2/W3 title-strip + Abstract compress + §6.6 chain), round3_novelty + response (CRIT-N1/N2/N3 + MAJ-N1 single-central-contribution restructure).
**Posture:** Three constructive rounds genuinely improved the form. They did not test the substance under hostile probing. The question I am answering is: *if I am a PC member who reads to reject, can I find a kill shot?* The answer is yes — three of them.

---

## Decision

**REJECT for EMNLP / NeurIPS Main. Borderline-acceptable for EMNLP Findings only after the FATAL items below are surfaced honestly in §1 / Abstract; otherwise reject for Findings as well on transparency grounds.**

The single most damaging issue: **the paper sells a "deployable" *4-clause free-lunch* mitigation (E6) whose Δdf headline survives multiplicity correction on exactly 1/5 datasets (PlotQA n=2,306) — 4/5 cells fail their own bootstrap CI test, three of them on n ≤ 443 — and the entire mitigation chain (calibration, hyperparameter selection, evaluation, capability preservation) is run on a single architecture (`llava-onevision-qwen2-7b-ov`). The §1.5 "central contribution" is therefore ONE point in a (model × dataset × hyperparameter) space whose generalization to either neighbouring axis is structurally unverified.** Round-1 surfaced part of this (CRIT-2 → Δem(b) Bonferroni-clean is the multiplicity-robust headline); Round-2/Round-3 left the reframe in place but did not address the underlying single-architecture single-headline-cell problem. Add to this the §4.6 framework "prospective verification" which tests at K=1 the framework that was operationalised at K=8 (i.e. tests a *different* dimensionality than the one the framework's paper-§6 instantiation uses), and the residual experiment-log signature is that **the strongest cleaned-up version of this paper still rests on a chain of single cells that each survive their own narrow test but do not interlock**.

What three rounds of constructive review papered over but did not fix:

1. **(FATAL-A)** N=1 model on the headline mitigation is *acknowledged* in §3.3 panel-scope hedge but the Abstract / §1.5 contribution sentence still reads as a method-of-the-paper. The §3.3 disclaimer ("once") is form-correct but the headline does not honour it.
2. **(FATAL-B)** Δdf "5/5 sign-clean" headline collapses to "1/5 CI-clean" under the paper's own paired-bootstrap procedure. The §1.5 (4) prose pivots to *Δem(b) Bonferroni-clean* — but the 4-clause free-lunch criterion's *Δdf < 0 clause* is the one that matters for the *anchoring mitigation* claim, and it is 1/5 CI-clean.
3. **(FATAL-C)** §4.6 γ-β bridge is positioned as the framework's *prospective* test, but it tests the framework at K=1 (different from K=8 used in §6) and concludes "qualitative bridge / quantitative interlock deferred" while implicitly partial-falsifying the universal-K=8 assumption that the paper *actually deployed*. This is not prospective verification — it is *partial post-hoc rescue at a different point* of the framework's parameter space.

If any one of these were addressed by surfacing it as a hedged limitation tied to the *contribution sentence*, this paper sits at borderline-Findings. As currently written, the §1.5 central contribution sentence is structurally at variance with the §6.2.3 Δdf 1/5 reality and the §3.3 N=1 reality. **Reject.**

---

## Recommended action by the area chair

**Reject for Main. Accept for Findings only conditional on:**

- (a) Abstract's "central contribution" sentence rewritten so the *single-model case study* register propagates into the contribution language, not just §3.3.
- (b) §1.5 (4) Δem(b) headline disambiguated — the *anchoring task Δdf* (the criterion's first clause) is 1/5 CI-clean; the Δem(b) (the criterion's third clause) is 5/5 Bonferroni-clean. These are not the same thing and the body / abstract continue to slide between them.
- (c) §6.5 "유일 cell" / "권장 — 4-clause 동시 충족" verdict cell stripped of headline status until either CAA-at-K=1 (~1 H100-day per §8.4 item 4) or random-K=8 (~1 H100-day per §8.4 item 2) is run. As-currently-positioned, both gating baselines are deferred, and the paper still ships the verdict.

These are not new experiments; they are honest framing of what the paper actually demonstrates.

---

## Stress-test verdict — what survived three rounds, and does it hold under aggressive read?

| Surviving claim (what Rounds 1-3 left in place) | Section | Holds under aggressive read? | Reasoning |
|---|---|---|---|
| Multi-layer redundancy (single-layer 5/5 null on mech panel + OneVision 5/5 null) | §5.2, §5.3 | **Yes (substantively)** | Lab-grade evidence; OneVision extension is a real cross-validation. *But*: §5.2 Insight 2 "single-layer null *predicts* single-direction failure" overreaches — see MAJ-3. |
| (a − m) digit-pixel paired-inpaint design | §6.2.1 Insight | **Yes (design)** | Genuinely clean isolation if Telea-inpaint counterfactual is exact. *But*: the inpainting confound (FATAL-D below) is not surfaced. |
| L1 6-bin gradient (continuous-confidence reframe of wrong/correct asymmetry) | §4.4 | **Partially** | 51-57/80 strict-monotonic is real; *fully* strict 5/5 only 21-24/80 — body has the hedge but the §1.5 "세 직교 axis 증거" prose smears point-estimate sign with strict monotonicity. |
| 6-benchmark capability preservation macro Δ +0.41 pp | §7 | **Yes (single-arch)** | Numbers reproduce. *But* macro-Δ averaging across benchmark types disguises the OCRBench/MMBench 1-2 pp drops that are within the paper's own ±1 pp band but cluster on the negative side of zero (3/6 negative point estimates). |
| Δem(b) 5/5 Bonferroni-clean | §6.2.3, Abstract | **Yes (numerically) / No (interpretation)** | Numbers are CI-clean. The interpretation as *"4-clause free-lunch passed"* slides from "Δem(b) clean" to "free-lunch passed" — the criterion has 4 clauses, only one is multiplicity-robust. |
| Δdf 5/5 sign-clean | §6.2.3, body | **No (in headline form)** | Sign-clean ≠ CI-clean; 1/5 CI-clean. Round 1 CRIT-2 surfaced this in §6.2.3 self-reframe; Abstract still leads with 5/5 sign-clean framing. |
| §5.4 framework (post-hoc synthesis labeled honestly post-Round-1) | §5.4 | **Surface honesty / substance trouble** | Round 1 forced "사후 synthesis" relabel — good. *But* §1.5 (3) supporting finding still puts the framework on the load-bearing path between §5.2 mechanism and §6.2 mitigation, while the framework's only prospective leg (§4.6) operates at K=1 and the deployed mitigation at K=8 — i.e., the framework's "prospective verification" doesn't verify the *operative* parameterisation. See FATAL-C. |
| γ-β reasoning amplification (auxiliary, ×12.7 ratio) | §4.5, §1.5 aux | **No (substance)** | N=1 architecture × N=1 dataset × no CI on a 3-sig-fig ratio computed from 0.021/0.267 with denominator 249/385 (i.e., post-stratification ≤200 numeric pairs per arm). Round 1 MAJOR-6 asked for paired-bootstrap CI; deferred. ×12.7 is unfit for any conference reporting. |
| 27-cell pilot grid winner cell #17 vs runner-up #8 within-1-SE disclosure | §6.2.2 | **Surface honesty / substance trouble** | Round 1 MAJOR-10 forced the 1.2 pp gap surfaced. *But*: the within-1-SE language treats the 27-cell *grid* as the only multiple-comparisons family; the *Bonferroni-20 across 5 datasets × 4 metrics* in §6.2.3 *does not* correct for the 27-cell grid selection. See MAJ-2. |
| 4-clause free-lunch criterion as positive cross-axis result vs Chand et al. | §2 + §6.2.3 + §6.5 | **Yes (positioning) / No (load-bearing)** | The criterion is well-defined and Chand et al. positioning is correct. *But* the criterion's first clause (Δdf < 0) is 1/5 CI-clean — the cross-axis "VLM × continuous numerical regression × inference-time activation projection achieves 4-clause" claim is *overall point-estimate-consistent + Δem(b) clean + Δdf single-dataset clean*, which is a much weaker positive than the paper's positioning carries. |

Bottom line on stress-test: 4-5 of the surviving claims hold up substantively; 3-4 hold only as written but contain unresolved kill points the constructive rounds didn't reach.

---

## Severity-graded attacks

### FATAL (paper must be rejected unless rebuilt)

#### **[FATAL-A] §1.5 / Abstract: the central-contribution sentence still reads as a method-of-the-paper claim while the §3.3 hedge says it is a single-model case study. The honest contribution is single-arch existence proof; the framing markets it as method.**

Verbatim, §1.5 line 39:

> 본 논문의 단일 *central contribution*은 **multi-direction subspace projection을 사용하는 cross-modal anchoring mitigation (E6)** 으로 ... 5 evaluation dataset × 6 held-out capability benchmark 위에서 multiplicity-robust 하게 충족한다 ...

Abstract (line 9), end:

> Mitigation chain은 단일 모델 case study이며 cross-architecture 일반화는 §8.2.

Two problems compound:

1. **The contribution sentence does not contain "case study"** — it says "central contribution" and lists *multi-direction subspace projection mitigation* as the noun. The 11-line sentence ends without scope qualifier; the case-study admission is in the *next* paragraph (Auxiliary observation), structurally separated from the central contribution.

2. **The §3.3 panel-scope hedge** ("Cross-architecture E6 재calibration은 §8.2 후속 작업; 이 panel-scope 분리는 본 절에서 단 1회 명시하며, 후속 절은 reference 이외에 반복하지 않는다") is form-correct, but the "once" rule means the *contribution sentence in §1.5 — the most-cited surface in the paper — does not carry the hedge*. A reader Ctrl-F'ing for "central contribution" in §1.5 finds a method-of-paper claim with no scope qualifier; a different reader Ctrl-F'ing for "single-model" in §3.3 finds the hedge. The §1.5 reader does not necessarily reach §3.3.

3. **The contribution falls apart the moment a different architecture is calibrated.** §5.3 documents OneVision dataset-dependent peak migration L=14 ↔ L=27 *within the same model*. §5.1 documents the mechanism panel cluster split: SigLIP-Gemma early (L=5), mid-stack cluster (L=14), Qwen-ViT late (L=22), FastVLM late (L=17). The §6.2 chosen cell L=26 + K=8 + α=1.0 is calibrated for OneVision Qwen2-backbone with cross-dataset peak around L=27; on Gemma3-4b (peak L=5) or Qwen2.5-VL-7b (peak L=22), L=26 is structurally *not* the integration site. The paper knows this — §8.4 item 3 explicitly defers cross-architecture E6 replication. **A method whose method-of-paper hyperparameter is known not to transfer to other architectures is not a method.** It is a single-model proof-of-concept marketed as a method.

What a hostile PC member writes: *"The title is plural ('Vision-Language Models'); the central contribution is N=1. The transparent action is to retitle to 'on LLaVA-OneVision' or to label E6 as 'a calibration recipe to test on each model individually'. Currently the paper is at variance with itself."*

**Why the constructive rounds missed this**: Round 1 surfaced the issue as MAJOR-1 panel scope; Round 1 reviser response added the §3.3 canonical hedge and removed 4 of the 5 duplicate hedge sentences from later sections. The fix consolidated *form* (one canonical statement) but left the *function* (the central contribution sentence is the most-read surface and does not carry the hedge). Round 3 reviewer reorganised §1.5 from 4 contributions to 1 + 3 supporting + 1 auxiliary — strengthening the headline form *without* tying it to scope.

**What would change my mind**: §1.5 (4) reframed as *"central contribution: a calibration recipe for cross-modal anchoring mitigation, demonstrated as a single-model case study on `llava-onevision-qwen2-7b-ov` (Δem(b) Bonferroni-clean across 5 anchor-task datasets and 6 held-out capability benchmarks); cross-architecture transfer is the immediate follow-up question (§8.4)"*. This is honest about scope and preserves the substance.

---

#### **[FATAL-B] §6.2.3 Δdf (the 4-clause free-lunch's *first* clause — the actual *anchoring mitigation* clause) is 1/5 CI-clean. The paper has been re-engineered to lead with Δem(b) Bonferroni-clean as the headline, but Δem(b) is the *non-anchored arm* effect — it is *not* the anchoring-task improvement.**

Per `docs/insights/_data/stage4_final_per_dataset_ci.md` (canonical source) Sign-clean count table:

| Metric | 95 % CI excludes 0 (matching dir) | Bonferroni-20 CI excludes 0 |
|---|:---:|:---:|
| Δ adopt(a) | 2/5 | 2/5 |
| **Δ df(a)** | **1/5** | **1/5** |
| Δ em(a) | 3/5 | 2/5 |
| Δ em(b) | 5/5 | 5/5 |

The 4-clause free-lunch criterion (paper §6.2.3 line 379):
> *Δdf(anchoring task)* < 0 ∧ *Δem(anchored arm)* ≥ 0 ∧ *Δem(non-anchored arm)* ≥ 0 ∧ *Δ(held-out capability macro)* ≥ −0.5 pp

The first clause is *anchoring task Δdf < 0*. CI-clean on 1/5 datasets (PlotQA only). The fourth clause (capability) is met by macro +0.41 pp (HallusionBench excludes zero, POPE pinned). The third clause (Δem(b)) is met 5/5 Bonferroni-20-clean.

**The paper's pivot is to emphasise the third clause as headline.** Verbatim Abstract: *"multiplicity-robust headline은 **Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 CI 모두 excludes 0** 이며 Δdf는 PlotQA n=2,306 위 CI-strong + 4 small-n cell 점추정-일관-CI-borderline."*

This is structurally a reframe-by-relabelling. The criterion is "4-clause"; the paper's headline pivots to the clause where the data is cleanest, not the clause that defines the *anchoring mitigation* function. **Δem(b) is improvement on the non-anchored arm. Δem(b) > 0 means the projection makes the model better at target_only forward passes — i.e., target_only forward passes that don't have an anchor.** That is *capability preservation on the non-anchored arm*. Calling that the "headline" of an anchoring mitigation sells a side-effect as the deliverable.

The actual *anchoring* deliverable is **Δdf(a) < 0**. CI-clean evidence: 1 out of 5 datasets, PlotQA, n=2,306. The 4 small-n cells with Δdf:
- TallyQA: Δdf = −0.3 pp [−1.3, +0.6] (CI overlaps zero by half its width)
- InfoVQA: Δdf = −0.7 pp [−4.7, +3.4] (CI half-width 4 pp; "fence" in §6.2.3 prose)
- ChartQA: Δdf = −4.0 pp [−9.8, +1.8] (CI overlaps zero)
- MathVista: Δdf = −4.1 pp [−11.8, +3.5] (CI overlaps zero by huge margin)

A hostile reviewer's read: "*the anchoring mitigation reduces direction-follow on PlotQA (the calibration domain) and is statistically indistinguishable from no effect on the four held-out evaluation datasets. The headline finding (Δem(b)) is on the non-anchored arm and is a side-effect of the projection. The paper rebrands the side-effect as the headline because the anchoring effect is single-dataset.*"

**Why the constructive rounds missed this:**

Round 1 CRIT-2 explicitly raised the abstract-vs-body §6.2.3 mismatch on the Δdf clause. Round 1 reviser response surfaced "Δem(b) is multiplicity-robust headline" — i.e., conceded the headline pivot. The reviser's edit reframes the abstract to lead with Δem(b). **This is not the fix CRIT-2 asked for** — CRIT-2 asked for the abstract to mirror §6.2.3's self-reframe; the reviser instead lifted §6.2.3's reframe up to the abstract and made it the *new* headline. The reframe was rhetorically smart but moved the headline to the side-effect axis.

Round 3 novelty reviewer noticed (CRIT-N1 verdict) but framed it as "uniquely passes" softening, not as "the headline clause is the *non-anchoring* clause."

**What this paper actually demonstrates** (honest reading): On a single-model calibration, a K=8 residual subspace projection at L=26 (a) reduces direction-follow on the calibration-domain dataset (PlotQA) with CI-clean evidence, (b) point-estimate-reduces direction-follow on 4 small-n held-out datasets within a sample-size-bound noise floor, (c) raises target_only exact-match consistently across all 5 datasets (Bonferroni-clean), (d) does not measurably damage 6 held-out capability benchmarks. The *anchoring* claim is single-dataset CI-clean. The *capability-preservation-with-bonus* claim is multi-dataset CI-clean. These are different findings; the paper marketing fuses them under the "free-lunch" idiom and labels the fusion the central contribution.

**What would change my mind**: Either (i) acknowledge in §1.5 (4) that the *anchoring task Δdf clause is single-dataset CI-clean; the multi-dataset CI-clean clause is non-anchored arm em*, or (ii) rerun on full-n cross-evaluation datasets (PlotQA-n=2,306-equivalent for ChartQA / MathVista / InfoVQA) so the small-n CI bands close. Option (ii) is the right scientific move; option (i) is the right rhetorical move.

---

#### **[FATAL-C] §4.6 γ-β residual-stream bridge is positioned as the framework's "prospective verification" but tests the framework at K=1 while the framework's operational deployment uses K=8. This is not prospective verification — it is partial post-hoc rescue at a different point in the framework's parameter space.**

Per the paper §4.6 (line 248) Insight 1:
> 동일 L = 33 + 동일 data + K = 1 vs K = 8 비교에서 bridge가 null (−0.05)에서 Bonferroni-positive (+0.28 [+0.19, +0.38])로 9× 차이 — §6의 K = 8 OneVision sweet spot이 cross-architecture universal이 아니다.

And §1.5 (3) supporting finding (line 39):
> 이 synthesis는 §4.6 γ-β residual-stream bridge에서 layer-routing 방향성 sign-reversal로 prospectively 검증되나 implicit universal-K=8 가정은 K=1 vs K=8 cross-architecture 차이로 부분 falsify되며 (§4.6 Insight 2), 따라서 *load-bearing theory*가 아닌 *통합 설명 framework*로 자리한다

What §4.6 actually shows:
- **At K=1**, mid-stack negative ↔ late-stack positive sign-reversal: 14/84 Bonferroni-corrected cells exclude 0.
- **At K=8 same L=33**: bridge is null (−0.05).
- **The framework's §6 instantiation uses K=8** at L=26. So the *parameterisation that the framework operationalises in the deployed mitigation* has *null evidence on the prospective leg*.
- **The 14/84 cells that pass Bonferroni** are 14 of 84 = 16.7 %. With Bonferroni at α=0.05/84 ≈ 0.0006, false-positive rate per cell is 0.0006 — under null we expect 0.05 cells out of 84. 14 is much greater than 0.05. So the *direction* finding survives Bonferroni. *But*: the framework's prediction does not specify *which 14 cells*; the framework predicts *direction*, not *which K × layer cells specifically realise the direction*. So the 14/84 statistic is *post-hoc cell selection* dressed as Bonferroni-clean.

What this means for the contribution:

1. The framework's *direction* prediction (mid-stack vs late-stack sign-reversal) is verified — but this is also what a generic "residual stream accumulates information" prior predicts. The framework does not make a *unique* prediction here.
2. The framework's *operational* prediction (K=8 universal sweet spot) is partially falsified. The paper acknowledges this. But the *deployed mitigation* uses K=8.
3. So the framework's prospective leg (a) confirms the generic prediction at K=1 (where the paper's deployed mitigation does not operate), and (b) *contradicts* the framework's specific prediction at K=8 (where the deployed mitigation does operate).

This is not a prospective verification of the framework. **This is a finding that on a different model the framework's universal-K=8 assumption fails**, recharacterised as "framework's layer-routing prediction confirmed."

§5.4 (line 290) explicitly says this:
> framework의 implicit *universal K=8 sweet spot* 가정은 부분 falsify — Qwen3-VL은 sv7/sv8 elbow가 1.026 gradual decay라 K=2..7 noise가 K=1 anchor direction을 dilute, 동일 L=33에서 K=1 vs K=8 ratio가 9× — *layer-routing 방향성은 framework-confirmed, dimensionality 보편성은 framework-partial-falsified*. 본 honest disclosure는 framework의 falsifiability를 보장하는 핵심 element이다.

The "honest disclosure" is form-correct. But it is positioned as a *strength* of the framework (falsifiable). What the paper does not surface: **the K=8 partial-falsification means the framework's only prospective leg fails the test at the deployed K**. This is much closer to "framework partially falsified at K=8, rescued at K=1" than to "framework prospectively verified."

A hostile reviewer's read: *"The framework's prospective leg verifies the generic part of the prediction (residual stream accumulates information across layers; obvious) and falsifies the specific part (universal K=8); the framework is then re-fit to K=1 on Qwen3-VL while K=8 stays operational on OneVision. This is framework retrofitting, not framework verification."*

§4.6 also has Insight 2 (line 250): *"within-Thinking magnitude (+0.5 ~ +0.9 amplitude units, baseline 위 ~0.2 ~ 0.4 % 상대 변화)는 §4.5 Table 4의 correct-base df ratio 큰 폭 증가와 *정량적*으로 정렬되지 않는다 ... *Qualitative bridge established / quantitative interlock deferred* (§8.2)"*. The framework's prospective test fails the quantitative interlock and the paper labels it "qualitative bridge established."

**Why the constructive rounds missed this:**
Round 1 CRIT-1 forced "predict-then-verify chain" reframe to "post-hoc synthesis + prospective leg at §4.6." Round 1 reviser response retained §4.6 as the prospective leg "with K-falsification surfaced." The reviser explicitly chose to keep the §4.6 leg *as* the prospective verification despite the K=1 vs K=8 mismatch. **The reviser papered over the mismatch by relabelling the framework partial-falsification as a strength.**

Round 3 reviewer (CRIT-N3) demanded "이론적 기여" → "통합 설명 framework" relabel. The reviser did this. But the relabel does not address the K=1/K=8 mismatch — it only softens the contribution language. The framework is still on the load-bearing path between §5.2 mechanism and §6.2 mitigation in §1.5 (3) and §8.1.

**What would change my mind**: Run the §6.2 calibration at K=1 + K=4 + K=12 + K=16 on OneVision (~1 H100-day per K-cell) and report the L=26 × {K} × α=1.0 sweep. If K=8 is the empirical sweet spot among {1, 4, 8, 12, 16} on OneVision, the framework's prediction at K=8 has data behind it. If K=8 is not the empirical sweet spot when broader K is searched, the framework's K=8 instantiation is grid-search-artefact. Currently the OneVision K-grid is K ∈ {2, 4, 8} per §6.2.2 line 347 — three points, K=8 is the largest. The Qwen3-VL bridge sweeps K ∈ {1, 2, 4, 8, 12, 16} and finds K=1 is right. The OneVision grid does not include K=1 or K ≥ 12. **The K=8 sweet spot claim on OneVision is therefore "K=8 is the largest in {2, 4, 8}", not "K=8 is the local optimum in a wide K range."**

This is the same kind of grid-coverage-as-optimum issue Round 1 MAJOR-7 surfaced and Round 1 reviser response soft-deferred. Three rounds later, the K=8 sweet spot is still not from a K-grid that includes K=1 or K ≥ 12.

---

#### **[FATAL-D] (a − m) Telea-inpaint counterfactual leaves a texture signature; "digit pixel causality" should read "digit-pixel-or-Telea-residue causality." OCR-verified digit absence does not control for representation-level texture cues.**

§3.1 + §A.2 + §6.2.1 establish the (a − m) contrast: same scene, OCR-verified digit absence after Telea inpainting. §4.2 line 169 reframes:
> `adopt(m)`이 양수일 수 있는 원인은 (i) 모델 prediction noise가 우연히 anchor_value와 일치 (~0.5 × P(numeric pair) baseline), (ii) anchor scene background에서 잔존하는 미세한 cue (Telea inpaint이 픽셀 레벨에서 완전 무 잔여 OCR 검증되었으나 representation level에서의 잔여 가능성), ...

The paper acknowledges (in passing) that representation-level residue is plausible. But the *(a − m) calibration substrate* — the SVD basis for the K=8 subspace projection — is computed from `D[i, L, :] = h(x_i^a, L) − h(x_i^m, L)` and the §6.2.1 Insight makes the design principle explicit:
> calibration contrast는 인과 통로 (causal pathway) 를 confounding variance로부터 *분리*하는 paired difference여야 한다

If the Telea inpaint residue contains *texture cues distinguishable to the representation* (e.g., frequency-domain artefacts where the digit was blurred out, color-bleeding around the inpaint boundary, edge artefacts), then `D[i, L, :]` captures *(digit content + Telea-residue) − Telea-residue + epsilon*, which is not the *digit-pixel-only* axis the paper claims. The K=8 subspace then includes Telea-residue-related directions. The "digit-pixel causality" claim becomes *digit-pixel-or-Telea-residue* causality.

Why this matters for the headline:
1. **§4.2 Insight 1** (line 171) ties the (a − m) gap magnitude to model anchor pull strength. If the (a − m) gap also reflects how much the model representation distinguishes Telea texture from clean pixels, then anchor-strong models also have strong texture-distinguishing representations — confounded.
2. **§6.2 E6 mitigation** removes the K=8 subspace at inference. If part of the subspace is Telea-residue-related, the deployed mitigation removes a generic *image-modality-noise direction* on every forward pass. The b-arm em +8.8 pp gain is then easier to explain as "removed a generic noise direction" — i.e., Alt-1 (general regularization) of §6.3 Insight 1.5, which the paper acknowledges is unfalsified within the round.

The paper's only control for this is the OCR digit-absence verification — pixel-level absence, not representation-level absence. The right control is comparing K=8 SVD on (a − m) with K=8 SVD on (m − m') where m and m' are two different inpaint-passes of the same scene with no digit ever present (so D is all-Telea-residue and any subspace recovered should be Telea-only). The paper does not report such a control.

**§4.2 (m, b, d) controls** address scene-background causality (paragraph at line 169: *"masked와 neutral이 correct-base 정확도에 끼치는 손실은 1-2 pp 안에서 구별 불가"*). Scene background ≠ Telea inpaint texture. Scene background is the *anchor scene* shared with `a` and `m`; Telea inpaint texture is *distinct to `m` only*. So (m, b, d) controls do not cover the Telea texture axis.

**Why the constructive rounds missed this:**
Round 1 methodology focused on the canonical CSV reproduction; the (a − m) Insight is treated as the paper's strongest move and not attacked. Round 2 writing was prose-axis. Round 3 novelty positioned (a − m) as a generalisable design pattern. Three rounds in, the inpaint-counterfactual confound has not been raised because each reviewer focused on a different axis.

**What would change my mind**: Either (i) report the (m − m') "inpaint-noise-only" SVD baseline (~2 H100-hour: regenerate 128 inpainted-twice neutrals, run SVD, compare K=8 subspace cosine similarity to the original (a − m) subspace), or (ii) explicitly acknowledge in §6.2.1 Insight that "(a − m) captures *digit-pixel-or-Telea-texture-correlated* directions; a control for inpaint-only texture is deferred to §8.4." Currently §6.2.1 frames (a − m) as cleanly isolating the causal pathway; a hostile reviewer reads "Telea inpaint" as a confound. (Round 4 v1-archive review (CRIT-3 in archive) raised the b-arm regularization story; this round adds the Telea texture confound as the *upstream* explanation.)

---

### MAJOR (multiple of these compound to rejection)

#### **[MAJ-1] §4.6 14/84 Bonferroni-clean cells is post-hoc cell-selection. The framework predicts *direction*; the paper reports *which* cells are clean, treating Bonferroni as if it disciplined the selection. It does not.**

Per §4.6 Setup (line 229):
> 7 layer × 6 K (K ∈ {1, 2, 4, 8, 12, 16}) × 2 statistic = 84 cells L×K sweep으로 측정. Within-Thinking paired (T_a − T_d per sid) bootstrap B = 10,000, Bonferroni-corrected k = 84.

Bonferroni at k=84, α=0.05: per-test α = 0.05/84 ≈ 0.0006, two-sided z ≈ 3.43, 99.94% CI bound. 14 cells clean of zero.

Two issues compound:

1. **The framework's prediction is qualitative-directional**: late-stack positive, mid-stack negative. The framework does not predict *which K ∈ {1, 2, 4, 8, 12, 16}* nor *which layer ∈ {late stack}* should be Bonferroni-clean. The paper *finds* that 14 cells survive, then reports those 14 cells, then claims the surviving cells "confirm" the framework. But under the framework's directional prediction, *any* combination of (some K, some late layer) being non-zero is consistent. The 14/84 fraction is "fraction of (K, layer, statistic) cells where the directional effect is large enough to survive Bonferroni" — not "fraction of framework-predicted cells that survive."

2. **The framework's *implicit* universal-K=8 prediction is then partial-falsified by inspection of the 14 cells.** If §4.6's framework prediction *had been* "K=8, late layer, mean statistic ⇒ positive amplitude," and Bonferroni had then tested 1 cell, a single-cell pre-registered test would discipline the comparison. Instead the paper sweeps the entire (K, L, statistic) lattice, post-hoc selects the cells that are clean, and then reads off "K=8 was null at L=33; K=1 was clean — therefore universal-K=8 is partial-falsified." The K=1 finding is *post-hoc cell selection*, not pre-registered prediction.

A hostile reviewer's read: *"This is a fishing expedition with Bonferroni correction labelled as if it were predictive verification. The framework's prospective leg is post-hoc cell-selection. The 14/84 number is 'how many cells happened to be large enough'; it is not 'how many framework-predicted cells held up.' If the framework had no predictive content, you would still see 4-5 cells survive Bonferroni in a 84-cell sweep at α=0.05/84 if the underlying signal is weakly correlated across cells (which late-stack residual amplitudes are, since they're highly correlated across L within Thinking)."*

**Why the constructive rounds missed this**: Round 1 raised "84-cell sweep is multiple-comparisons inside Bonferroni" as MAJ-9 (statistical rigor); Round 1 reviser surfaced the column header "Bonferroni 99.94 % CI" but did not address the cell-selection issue. The structural critique — that 14/84 is post-hoc selection — was not made.

**What would change my mind**: Pre-register a specific (K, layer, statistic) prediction *before* running the sweep, then test that single cell. The current 14/84 is exploratory analysis labelled as confirmatory.

---

#### **[MAJ-2] §6.2.2 cell #17 selection — Round 1's MAJOR-10 within-1-SE disclosure addressed *grid* uncertainty but not the larger cherry-pick problem: the paper applies Bonferroni-20 across the §6.2.3 evaluation but does NOT correct for the 27-cell grid pre-selection. The Bonferroni-20 family-wise error rate is therefore mis-stated.**

Round 1 MAJOR-10 forced the paper to disclose that cell #17 vs #8 is within ~1 SE on the calibration-set ranking (§6.2.2 line 351-353). This is a partial fix.

The deeper issue: §6.2.3's Bonferroni-20 (5 datasets × 4 metrics) treats the 5×4=20 paired tests as the multiple-comparisons family. **But the cell #17 (L=26, K=8, α=1.0) was selected from a 27-cell pilot grid on the calibration set.** The selection-on-calibration → evaluation-on-held-out paradigm requires the held-out evaluation to be *single-cell-pre-registered*. The Bonferroni-20 is then correct for *one specific cell's* multi-dataset / multi-metric evaluation. **It is not correct if the 27-cell selection itself is a multiple-comparisons family.**

The paper's argument (Round 1 reviser MAJOR-10 fix) is that the selection rule was ex ante fixed (em-deal-breaker ≤ −6 pp). But:
- The em-deal-breaker rule is non-binding: §A.5 line 596 says "27 cell 중 *어느 cell도* 이 −6 pp 임계값을 위반하지 *않는다*" — the rule pruned no cells.
- The actual selection criterion is therefore "argmax over 27 cells of mean Δdf(a) on PlotQA + InfoVQA pooled."
- A 27-cell argmax is a 27-fold multiple-comparisons selection.
- The §6.2.3 evaluation then runs the chosen cell on 5 datasets × 4 metrics = 20 tests, with Bonferroni-20 applied.
- The combined family is therefore more like 27 × 20 = 540 comparisons or 27 + 20 = 47 (under conservative counting).

A hostile reviewer's read: *"Bonferroni-20 is correct for the held-out evaluation conditional on the cell being pre-registered. Pre-registration requires the cell to have been chosen without any look at the held-out data and without any look at any other plausible selection criterion. The 27-cell argmax violates the latter — it is data-driven cell selection from a grid even if the grid was specified ex ante. The correct correction is Bonferroni-540 or at minimum Bonferroni-47. Under Bonferroni-540, only PlotQA Δem(b) (the single largest effect) would survive."*

This is more aggressive than the paper's "1.2 pp within ~1 SE" admission. The within-1-SE admission is *cell #17 vs cell #8 ranking instability*. The cherry-pick attack is *cell #17 vs the null hypothesis of no signal anywhere in the 27 cells*.

The 1/5 Bonferroni-20 CI-clean for Δdf would shift to 0/5 under Bonferroni-540, leaving only Δem(b) (which has effect sizes of +4.7 to +13.8 pp, large enough to survive even Bonferroni-540 on n=2,306 PlotQA and n=4,978 TallyQA).

**Why the constructive rounds missed this**: Round 1 MAJOR-10 was framed as "is the within-grid ranking stable" and the response (reviser Edit 11) addressed within-grid SE. The cross-family multiplicity (grid × evaluation) was not raised.

**What would change my mind**: Either (i) re-run §6.2.3 with a Bonferroni correction that includes the 27-cell selection (Bonferroni-540), or (ii) explicitly discuss in §6.2.3 that the multiplicity correction is conditional on cell pre-registration and acknowledge the 27-cell selection as a separate multiplicity layer.

---

#### **[MAJ-3] §5.2 Insight 2 "Multi-layer redundancy *predicts* single-direction failure" overreaches. The paper presents §6.4 LEACE rank-1 ChartQA backfire as *empirical verification* of the §5.2 prediction. But §6.4 was observed *before* §5.4 framework was authored (Round 1 CRIT-1 admission). So §6.4 is not a verification of a prediction — it is one of the observations the framework was synthesised from.**

Verbatim §5.2 Insight 2 (line 272):
> Multi-layer redundancy는 single-layer 또는 single-direction mitigation의 cross-dataset 실패를 *이론적으로 예측*한다 — dataset이 다르면 signal이 *다른 layer 조합*에 분산되며, 한 dataset에서 보정한 single direction이 다른 dataset의 다른 방향에 정렬되지 못한다. 이 예측은 §6.4에서 single-direction ActAdd cross-dataset 실패 + LEACE ChartQA 역행 +56 % 결과로 *경험적으로 검증*된다 (§6.4 Insight 1과 짝).

§5.4 (line 282) acknowledges:
> §5.2의 multi-layer redundancy 결과 ... + §6.4의 LEACE rank-1 ChartQA +56 % 역행 — 이 세 mechanism finding은 본 framework 작성 *이전*에 모두 관찰되었으며, 본 절은 이 세 결과를 *사후*에 단일 mechanism narrative로 묶는 **synthesis**이다.

**The two statements are at variance.**
- §5.2 Insight 2: multi-layer redundancy *predicts* single-direction failure; §6.4 *verifies* it.
- §5.4: §5.2 + §5.3 + §6.4 all observed *before* framework writeup; framework is *post-hoc* synthesis.

Both cannot be true. If §5.4 framework is post-hoc to §5.2 + §5.3 + §6.4, then §5.2 Insight 2 cannot claim §6.4 *verifies* a §5.2 prediction unless that prediction was authored *between* §5.2 (single-layer null observation) and §6.4 (LEACE backfire observation). Per Round 1 CRIT-1 admission (in reviser response Edit 10): *§5.4 was authored in v7 (Round-5 bar-raiser response), pre-Phase 5; §6.4 result is in `docs/experiments/E6-tally-only-rerun-tracker.md:480` predating §5.4.* So:
- §5.2 result observed (call this T1).
- §6.4 LEACE backfire observed (call this T2).
- §5.4 framework authored (call this T3).
- T3 > T2 > T1.

The §5.2 Insight 2 prediction language ("이론적으로 예측") implies T2 > prediction > T1. But the paper documents T3 > T2: the framework that *generates* the prediction is post-T2. So the prediction language is retrospective fitting.

The honest reading: §5.2 multi-layer redundancy *and* §6.4 LEACE backfire are two empirical observations that §5.4 framework explains together. §6.4 does not verify a §5.2 prediction; it is one of two observations from which the framework was constructed.

**Why the constructive rounds missed this**: Round 1 CRIT-1 was the deepest attack on this — and the reviser fix was to relabel §5.4 framing to "post-hoc synthesis." But Round 1's fix did not propagate to §5.2 Insight 2's "predict-then-verify" language. The §5.2 → §6.4 chain language survives, structurally inconsistent with §5.4's post-hoc admission.

**What would change my mind**: Edit §5.2 Insight 2 to remove "이론적으로 예측" / "경험적으로 검증" language; replace with "*post-hoc consistent with*" or "*together accommodated by §5.4 routing-vs-integration synthesis*". The intellectual content is unchanged; the prediction-verification language is what's wrong.

---

#### **[MAJ-4] §1.5 supporting finding (i) "anchoring을 wrong/correct binary projection이 아닌 continuous confidence gradient로 재해석" — the strict 5/5 monotonicity is on 21-24/80 cells (26-30%); ≥ 4/5 strict is 51-57/80 (64-71%). The contribution sentence reads as "5×6=80 cells confirm continuous gradient"; the data says 30% strict, 70% relaxed.**

§4.4 line 193 honest:
> *fully strict 5/5 pairs* 기준은 21 / 80 ~ 24 / 80 cell로 더 엄격하게 잡힌다

§4.4 line 203 surfaces non-monotonic cells:
> ≥ 4/5 strict pair criterion을 통과하지 못하는 23-29 / 80 cell (29-36 %; cross_entropy 29 / 80, log_prob_sum 23 / 80)

So:
- Strict 5/5 monotonic: 26-30% of cells.
- ≥4/5 strict: 64-71%.
- Fail ≥4/5: 29-36%.

§1.5 (i) supporting finding language:
> 5 dataset × 6 model cross-dataset 위에서 anchoring을 wrong/correct binary projection이 아닌 *continuous confidence gradient*로 재해석하는 세 직교 axis 증거 (L1 6-bin gradient, (a − m) digit-pixel causality, wrong/correct binary stratification; §4)

The "세 직교 axis 증거" language treats L1 6-bin gradient as load-bearing for the *continuous gradient* reframe. But strict-monotonic L1 only holds on 30% of cells. The body's load-bearing claim is the panel-mean B6 − B1 gap (+19.5 to +23.5 pp), which is *aggregate-level monotonicity* — different from cell-level monotonicity.

A hostile reviewer's read: *"The 'continuous gradient' reframe holds for the panel mean and on a majority (≥4/5 strict) of cells, but not on a majority (5/5 strict) of cells. The §1.5 contribution language elides the strict-vs-relaxed distinction. A 30% strict-monotonic cell rate is consistent with the binary projection being the *true* underlying structure with measurement-noise injecting cell-level non-monotonicity. The paper has not falsified that null."*

**Why the constructive rounds missed this**: Round 2 MAJ-W4 / MAJ-W5 addressed the run-on sentences in §4.4 but did not attack the strict-vs-relaxed framing. Round 3 MIN-N2 reordered §1.5 (i) to lead with the reframe but did not address the strict-cell-fraction.

**What would change my mind**: Either (i) compute panel-level monotonic-trend test (Mann-Kendall on the bin-rank correlation per cell, Bonferroni-corrected across 80 cells) and report what fraction is significantly monotonic *as a statistical test*, not as a strict-pair-counting heuristic; or (ii) acknowledge in §4.4 that 70% relaxed monotonic is the headline finding but 30% strict-monotonic is the more conservative count.

---

#### **[MAJ-5] §6.2.3 InfoVQA Δdf = −0.7 pp on n=443 with 95 % CI [−4.7, +3.4] is *symmetrically distributed around zero*. Calling this "fence" hedges the noun but the data is consistent with no effect at all. The "5/5 sign-clean" headline relies on +0.001 pp positive point estimates that the paper itself admits are within noise.**

InfoVQA Δdf:
- Point estimate: −0.7 pp
- 95 % CI: [−4.7, +3.4] (CI half-width 4 pp on a 0.7 pp point estimate)
- §6.2.3 prose: "InfoVQA n=443 [−4.7, +3.4] fence"

For Δdf = −0.7 pp to count as "sign-clean negative" the point estimate needs to be negative; under the paper's own bootstrap procedure with B=10,000 paired draws, the fraction of bootstrap samples where Δdf < 0 is approximately the lower-CI-mass of the symmetric distribution. With CI [−4.7, +3.4] symmetric around −0.65, roughly 60% of bootstrap draws are negative — i.e., the *direction* itself is barely better than coin-flip on this dataset.

§6.2.3 line 376 prose:
> Δdf(a)는 sample-size에 묶여 있다 ... InfoVQA n=443 [−4.7, +3.4] fence

"Fence" is a verbal hedge that does *not* admit the data is consistent with zero effect. A hostile reviewer reads: *"This dataset shows no effect. The paper reports it as 'sign-clean negative' because the point estimate is −0.7 pp; under bootstrap that's 60-40 negative. That is not sign-clean evidence; that is no evidence."*

The same structural problem: TallyQA Δdf = −0.3 pp [−1.3, +0.6] is reported as "TallyQA baseline df floor" — i.e., the paper acknowledges TallyQA has so little baseline anchoring effect that the projection cannot reduce it further. TallyQA contributes nothing to the cross-dataset Δdf claim. So of the "5/5 sign-clean" Δdf, in honest reading:
- PlotQA: CI-clean strong evidence.
- ChartQA: CI [−9.8, +1.8], point estimate −4.0 pp, CI overlaps zero by 1.8 pp half-width.
- MathVista: CI [−11.8, +3.5], point estimate −4.1 pp, CI half-width 3.5 pp.
- InfoVQA: CI [−4.7, +3.4], point estimate −0.7 pp, fence.
- TallyQA: CI [−1.3, +0.6], point estimate −0.3 pp, baseline floor.

Of these, PlotQA is the only one with statistical evidence. ChartQA and MathVista point estimates suggest a real effect but CI half-widths are 1.8 and 3.5 pp (close to and exceeding the point estimate magnitude). InfoVQA is null. TallyQA is floor.

**The "5/5 sign-clean" framing in the body is technically true (5/5 datasets have negative point estimates) but is misleading as evidence**. Honest framing: 1/5 CI-clean, 2/5 point-estimate-strong-but-CI-wide, 1/5 fence, 1/5 floor.

**Why the constructive rounds missed this**: Round 1 CRIT-2 surfaced the multiplicity issue and the 1/5 CI-clean count. Round 1 reviser response (Edit 5) compressed the §6.2.3 self-reframe to 3 sentences carrying the headline-vs-fence distinction. The fence language is preserved but the "5/5 sign-clean" framing is also preserved in §1.5, Abstract, §8.1. The two framings co-exist; the latter is the marketed framing.

**What would change my mind**: Re-run InfoVQA / ChartQA / MathVista at PlotQA-equivalent n (~2,300 each) and report whether the Δdf CIs close. Currently the paper has run §6.2.3 on stratified small-n cells; the natural cross-evaluation is on full-n cells. (Per §3.3 raw n: ChartQA 5,390; MathVista 385; InfoVQA 1,147 — so MathVista is fundamentally bounded; ChartQA and InfoVQA could be run at full n.)

---

#### **[MAJ-6] §7 Table 9 macro Δ +0.41 pp masks heterogeneity. 3 of 6 benchmarks have negative point estimates (OCRBench −0.80, MMBench −0.34, POPE −0.06); only HallusionBench excludes zero on the positive side. The macro is weighted by per-benchmark n, not by signal magnitude — under any reasonable per-benchmark weighting the macro Δ is statistically distinguishable from null only via HallusionBench.**

Per the data:
- RealWorldQA: +1.31 [−0.27, +2.89] — CI overlaps zero on the positive side; point estimate positive, CI half-width 1.6 pp.
- OCRBench: −0.80 [−1.68, +0.08] — CI overlaps zero on the negative side; point estimate negative.
- HallusionBench: +2.21 [+1.14, +3.28] — CI excludes zero positive.
- MMStar: +0.13 [−0.77, +1.04] — CI symmetric around zero.
- MMBench-DEV-EN: −0.34 [−0.82, +0.13] — point estimate negative, CI overlaps zero.
- POPE: −0.06 [−0.21, +0.09] — pinned to zero.
- Macro: +0.41 pp.

3/6 negative point estimates. 1/6 CI-excludes-zero (HallusionBench). 5/6 within ±1.0 pp pre-registered band. Macro +0.41 pp is dominated by RealWorldQA (+1.31) and HallusionBench (+2.21); the negative drift on OCRBench / MMBench / POPE / MMStar is real but small.

**This is a 4-clause free-lunch's *fourth* clause.** The clause is "Δ(held-out capability macro) ≥ −0.5 pp." The paper meets this clause. But the *qualitative* picture is not "free-lunch positive across capabilities" — it is "free-lunch positive on hallucination axis (HallusionBench), zero on the rest, with 3 small negative point estimates."

§7 Insight (line 452) frames HallusionBench positively as evidence the projection helps both anchoring and hallucination. But the negative point estimates on OCRBench / MMBench / POPE / MMStar are *consistent with* the projection mildly damaging capability on those tasks while helping HallusionBench. The macro averaging hides this.

A hostile reviewer's read: *"Half the benchmarks have negative point estimates. The HallusionBench gain is real and load-bearing. But the paper's '4-clause free-lunch passed' rests on macro averaging that disguises the fact that the projection has *mixed* effects across capabilities — strongly positive on hallucination/illusion, weakly negative on some VQA / multimodal-bench tasks. Calling this a free-lunch is an oversell."*

**Why the constructive rounds missed this**: Round 1 MAJOR-9 surfaced the 6-bench vs 8-bench issue (8-bench macro is +0.31 not +0.41); reviser Note added at line 446. The negative-point-estimates issue was not raised. Round 2/3 stayed on novelty and writing.

**What would change my mind**: Per-benchmark per-task-class breakdown (e.g., "anchoring-adjacent benchmarks: HallusionBench +2.21 (strongly positive); broad VLM benchmarks: −0.5 pp average (mildly negative); the projection helps where image-language hallucination is the failure mode, neutral or mildly negative where it is not"). This is the honest interpretive frame. The current "4-clause free-lunch passed" is the rhetorical frame.

---

#### **[MAJ-7] §1.4 ×12.7 ratio is back-cited from Abstract (Round 3 dropped it from Abstract per CRIT-N2). But §1.4 is part of §1 (Introduction) and is the FIRST place a reader sees the ratio. The ratio is computed from Instruct correct df = 0.021 with n_correct ≈ 249/385, so the post-stratification numerator denominator on the *correct-base* anchor pairs is ≤ 249 across all anchor levels. With the typical observed numerator < 10, the binomial SE on Instruct correct df is ~0.01 — i.e., the ratio is ×8.9 to ×26.7 under a 1-σ swing on the *denominator* alone. No CI reported.**

§1.4 line 35:
> Qwen3-VL-8B-Thinking은 같은 자극에서 Instruct 변형 대비 anchor pull을 amplify하며 (adopt ×1.6, df ×2.9; correct-base subset df ratio ×12.7, §4.5), ...

§4.5 Table 4 (line 217):
> | qwen3-vl-8b-instruct | 0.256 | 0.021 | +0.235 |
> | qwen3-vl-8b-thinking | **0.327** | **0.267** | **+0.060** |
> | 비율 | ×1.28 | **×12.7** | — |

Round 1 MAJOR-6 explicitly asked for paired-bootstrap CI on the ratio; Round 1 reviser response soft-deferred. Round 3 CRIT-N2 dropped the ratio from the *Abstract* but it survives in §1.4.

The numerical issue:
- Instruct correct df = 0.021. With n_correct ≈ 249, binomial point-estimate SE ≈ √(0.021 × 0.979 / 249) ≈ 0.0091 ≈ 0.91 pp.
- 1-σ band on Instruct correct df: [0.012, 0.030].
- Ratio = 0.267 / Instruct = 8.9 (at 0.030) to 22.3 (at 0.012).
- Ratio CI under just denominator noise: roughly ×9 to ×22.

Three sig figs (×12.7) is fundamentally over-reported. The ratio is ≥ ×8.9 directional under a 1-σ swing; ×12.7 ± large.

**Why the constructive rounds missed this**: Round 3 dropped from Abstract; §1.4 was treated as the body and not pulled. But §1.4 is in §1 — the introduction. A reader skims §1 before §4. The hedge ("§4.5", "denominator small") in §1.4 is one parenthetical away from the number; a Ctrl-F'er finds ×12.7 in §1.4 with the hedge two lines later. The deeper issue: even with the hedge, ×12.7 is reported to 3 sig figs without CI.

**What would change my mind**: Either compute the paired-bootstrap CI (Round 1 MAJOR-6 task; ~2 H100-hour) or drop the ratio entirely from §1.4 in favor of "≥5× directional." The paper has had three rounds to do this and has not.

---

#### **[MAJ-8] §6.5 "5-baseline panel" CAA / ITI exclusion. The Round-3 CRIT-N1 fix added Note 2 acknowledging tuning asymmetry, but this still leaves the central exhibit table (Table 8) with 5 rows (ActAdd, LEACE rank-1, Query-adaptive, CogBias, MIA-DPO LoRA) — none of which are actually CAA or ITI as evaluated by their authors. ActAdd is described as a CAA proxy; CogBias and Query-adaptive are obscure prior baselines. The 5-baseline panel is a *constructed* baseline panel that excludes the most direct competitors.**

Verbatim §6.5 Table 8 baselines:
- Single-direction ActAdd
- LEACE closed-form (rank-1)
- Query-adaptive offset (PCA + Ridge)
- CogBias decode-time
- MIA-DPO LoRA (weight space)

CAA [Panickssery et al., 2024] is *not* in the table; CAA is *reduced structurally* to "ActAdd cell" via Note 1. ITI [Li et al., 2023] is *not* in the table; ITI is reduced structurally via Note 1's §5.3 attention-locus argument.

The Note 1 reduction has the gap that Round 3 CRIT-N1 raised: ITI is multi-head, §5.2 single-head null does not cover it. The Round 3 reviser response (Edit 7) acknowledged this in the Note prose — but the *table itself* still calls the chosen cell "권장 — 4-clause 동시 충족 (이 5-baseline panel 위)". The "5-baseline panel" is therefore a panel of *non-canonical* baselines (ActAdd, ChatGPT-style decode-time methods, weight-space DPO) that *does not include CAA and ITI*. The most prominent prior methods to which this paper's E6 should be compared are *deferred to §8.4 item 4*.

A hostile reviewer's read: *"This table compares E6 against five baselines that are not the closest competitors. CAA and ITI — the two papers any informed reviewer expects to see — are reduced via thought experiment. The 'panel' is constructed to make E6 look distinctive against weaker baselines while declining to test against the strongest ones. Note 2's tuning-asymmetry caveat is more honest about ActAdd / LEACE than the paper's headline framing — but Note 2 itself surfaces that 'ITI multi-head fair-tuning could enable 4-clause partial-pass at some attention-head locus K-sweep cell' (paraphrased), which means ITI might pass the criterion under proper tuning, which means E6 is not uniquely-passing under proper tuning, which means the headline 'this 5-baseline panel' is hiding."*

**Why the constructive rounds caught this but the fix wasn't enough**: Round 3 CRIT-N1 demanded option (a) "rephrase to 5-baseline panel" or option (b) "run CAA-at-K=1 + ITI multi-head head-cluster at L=26." The reviser chose option (a) (cheaper). The fix is *form-correct* but leaves the table doing rhetorical work the substance does not support.

**What would change my mind**: Run CAA-at-K=1 with the same PlotQA + InfoVQA pooled (a − m) calibration as E6, on OneVision Main, at L=26. Report Δdf and Δem(b) for the 5 datasets. If CAA-at-K=1 backfires as predicted by §6.5 Note 1 prose, the structural reduction has empirical content. ~4-8 H100-hours per §8.4 item 4 estimate. Three rounds of review and this is still deferred.

---

#### **[MAJ-9] §6.3 b-arm em +8.8 pp interpretation: §6.3 Insight 1.5 surfaces three competing explanations and explicitly says "head-to-head 비교하지 *않았다*". §6.3 Insight 2 then walks the +8.8 pp into "사후 일관성 (post-hoc consistency) 형태의 신호" with HallusionBench. POPE is offered as ruling out yes/no general regularization — but POPE is binary, while the b-arm em is on numeric tokens. The Alt-1 (general regularization → mode-collapse on numeric digits) is not falsified by POPE.**

§6.3 Insight 1.5 (line 391) honestly lists Alt-1 (general regularization) and Alt-2 (numeric mode-collapse to 0/1/2/3 frequent ground-truths) as alternative explanations and notes that POPE pinned-to-zero "yes/no general-regularization을 어느 정도 약화시키나, *numeric* token logit 위에서의 mode collapse (Alt-2) 또는 anchor-task-specific subspace 정렬 (본 가설) 중 어느 것인지를 분리하지 *않는다*." The honest disclosure is in §6.3 Insight 1.5; the headline (§1.5 (4) + Abstract) does not carry it.

The b-arm em +8.8 pp is the multiplicity-robust headline (5/5 Bonferroni-clean). If the +8.8 pp is *anchor-task-specific* (i.e., the (a − m) subspace removed cleans up a generic wrong-base error mode), the paper's interpretation is correct. If the +8.8 pp is *generic regularization* (any K=8 subspace removal at L=26 has this effect), the paper's interpretation is wrong and the "free-lunch" reading is statistical artefact.

The within-paper falsification baseline is Random-K=8 + non-anchor calibration — explicitly deferred to §8.4 item 2 per Round 1 MAJOR-8 acknowledgement.

A hostile reviewer's read: *"The headline of the paper is +8.8 pp on b-arm. The paper says it doesn't know whether this is (a) anchor-task-specific subspace alignment, or (b) generic regularization at K/d ≈ 0.002, or (c) numeric mode-collapse to frequent answers. The Alt-1 falsification baseline (random-K=8) is one calibration run and one inference pass — perhaps half a day on the existing infrastructure. After three rounds of revision, this baseline is still deferred. The headline is on an unfalsified clause."*

**Why the constructive rounds missed this**: Round 1 MAJOR-8 raised it; Round 1 reviser deferred to §8.4 item 2. Round 2/3 didn't return to it. The paper's own honest disclosure (§6.3 Insight 1.5) is in the body but the Abstract / §1.5 headline relies on the +8.8 pp without the contingency.

**What would change my mind**: Run random-K=8 on OneVision at L=26 with non-anchor calibration set. ~1 H100-day per §8.4. The paper's headline rests on a clause whose competing explanations the paper itself cannot distinguish.

---

### MINOR (would be revision-required even individually)

#### **[MIN-1] §1.5 supporting finding (i): the L1 6-bin gradient claim is treated as one of "three orthogonal axes" alongside (a − m) digit-pixel causality and wrong/correct binary stratification. But (a − m) and wrong/correct stratification are *not orthogonal*: §4.2 Slice B explicitly orders dataset-level (a − m) magnitudes against §6.2.3 Δdf magnitudes, demonstrating they're correlated, not orthogonal.** §4.2 line 175: "*§6.2.3 5-dataset paired-sids Δdf 표 ordering ... 은 본 절 Slice B의 (a − m) gap magnitude ordering ... 과 *개별 dataset 부호 일관*하며*". Calling these "orthogonal" is a rhetorical move; the data shows they are correlated.

#### **[MIN-2] §4.6 Insight 1's 9× ratio (K=1 vs K=8 same L=33) is reported with 3 sig figs as "+0.28 [+0.19, +0.38]" vs "−0.05" — but the ratio is undefined (division by negative or zero) and the report mixes "9× difference" verbal with arithmetic on signed amplitudes. A 9× ratio claim on amplitudes [+0.28] vs [−0.05] is geometrically nonsensical (different signs); what the paper means is "K=1 is Bonferroni-positive while K=8 is null." Edit prose to remove "9×" framing.

#### **[MIN-3] §3.3 panel-scope hedge ends with "이 panel-scope 분리는 본 절에서 단 1회 명시하며, 후속 절은 reference 이외에 반복하지 않는다." This is *meta-instruction to the reader*; in conference paper form, the rule is implicit. The instruction-to-reader format reads as project-management residue.

#### **[MIN-4] §7 Note on benchmark coverage (line 446) discloses 6-bench → 8-bench macro shift +0.41 → +0.31 pp (drag of 0.10 pp from MME + AMBER). But the disclosure language ("contamination-resistant floor 강화 측면에서 4-clause free-lunch 판정은 두 panel에서 모두 유지") slides past the question of which is the canonical macro. Both can be technically free-lunch (≥ −0.5 pp); but 8-bench is closer to the threshold than 6-bench. Be specific: which is the canonical headline — 6-bench or 8-bench?

#### **[MIN-5] §A.3 distance cutoff table now has eligibility cutoff (≤ 5) vs S1 stratum (≤ 1) split per Round 1 MIN-9 fix. But §6.2.3 is on S1-stratified data; §6.2.2 calibration is on S1-stratified data. So the entire §6.2 / §6.2.3 mitigation chain is calibrated and evaluated on the *anchor-close* subset only. The §6.2.3 "5-dataset cross-evaluation" is therefore "5-dataset S1-stratum cross-evaluation" — not a representative test across the cutoff range. This needs surfacing in §6.2.3 caption.

#### **[MIN-6] §3.3 raw n vs per-cell stratified n: ChartQA 129 cells exist in the data. ChartQA Δdf in §6.2.3 is on n=224 (paired-sids intersection on a-S1 + b). 224 < 250 = pilot calibration n. Under the paired-sids intersection, the held-out evaluation set on ChartQA is *smaller than the calibration pilot*. This is a structural problem with the cross-evaluation design that no round has surfaced.

#### **[MIN-7] §4.5 Table 4 row "비율 | ×1.28 | **×12.7** | —" — the bold on ×12.7 is the only point-estimate-only number bolded in the paper's main tables (Tables 2, 3, 4, 5, 6, 7, 8, 9). Bold convention is "CI excludes 0 in headline direction" elsewhere; ×12.7 has no CI. The bold is rhetorical.

#### **[MIN-8] §8.4 item 1 (eigenvalue spectrum study, "1 spectral plot") is described as "가장 cheap한 rigor 향상" — a half-day's work. Three rounds in, the spectral plot is still item 1 of follow-up. If it's that cheap and that load-bearing for K=8 sweet spot defensibility, why has it not been done?

#### **[MIN-9] §A.4 stimulus seed: FLUX seed pinned per `seed_base 1729 + number`. But: the Telea inpaint (m) inventory is "deterministic, no random seed" — except OpenCV Telea inpaint *can* depend on input image floating-point precision and platform. Reproducibility on an H200 vs A100 vs H100 should be cross-validated; it is not.

#### **[MIN-10] §6.5 Table 8 "Cross-dataset 감소" column for E6 reads "**−0.3 ~ −5.2 pp on 5/5**" — this is the same range as §6.2.3 Table 7. But Table 7 has paired-bootstrap CI on each cell; Table 8 strips the CI and reports the point-estimate range. Reader misreads "−0.3 to −5.2 pp" as Δdf reduction across all 5 datasets when 4/5 have CI overlapping zero. Table 8 should carry the per-CI summary or at minimum point to Table 7.

---

## Per-attack-angle pass

### A. Overclaim hunt

Words I searched for in the body: "first", "uniquely", "유일", "novel", "deployable", "robust", "load-bearing", "predict", "free-lunch", "strict".

| Word/phrase | Location | Closest prior work | Hedge required | Status |
|---|---|---|---|---|
| "Vision-Language Models" (plural in title) | Title | — | "on `llava-onevision-qwen2-7b-ov` Main; cross-architecture follow-up" | **Survives — overclaim** |
| "deployable subspace projection mitigation" | Title, Abstract | CAA (LM-only deployment) | "single-architecture case study" | **Survives — overclaim** |
| "first VLM cross-modal numerical anchoring" | §1.1 (line 19) | VLMBias [Vo et al. 2025] (cue is subject-bound but is *visual content biases numerical answer*) | "first *independent-anchor open-numeric*" — the "first" is conditional on stacked qualifiers | **Surface honest, marginal** |
| "central contribution" | §1.5 (line 39) | — | "single-architecture case study" | **Survives — overclaim**, missing scope qualifier |
| "유일 후보 (only candidate)" | §6.5 sub-title (round-3-softened to "후보") | CAA / ITI not run | Round 3 changed to "5-baseline panel 위 ... 후보"; Note 1 still reduces CAA / ITI structurally | **Soften retained — but Table 8 verdict cell still claims unique passage** |
| "권장 — 4-clause 동시 충족 (이 5-baseline panel 위)" | §6.5 Table 8 verdict cell | CAA / ITI absent | Round 3 added "이 5-baseline panel 위" — but the table is still the central exhibit | **Survives — Table verdict reads as recommendation without the structural-baseline-exclusion hedge** |
| "free-lunch" without prefix | scattered (§8.2 "E4 free-lunch") | Chand et al. negative result | Round 2 mostly converged to "4-clause free-lunch"; one residual at §8.2 | **Mostly fixed** |
| "predict-then-verify" / "predicts" / "예측" | §5.2 Insight 2 ("이론적으로 예측"), §6.4 Insight 1 ("예측 → 검증") | Round 1 forced §5.4 to "post-hoc synthesis" | The §5.2/§6.4 "predict-verify" language was *not* updated when §5.4 was relabeled | **Survives — internally inconsistent** (see MAJ-3) |
| "load-bearing theory" → "통합 설명 framework" | §1.5 (3) | Round 3 forced relabel | Round-3 fix correct | **Fixed** |
| "novel" / "신규" | §6.5 Insight ("기법 class의 신규성이 아니라") | — | Already self-disclaimed | **Honest** |
| "robust" / "robustness" | scattered | — | Used contextually; OK | **OK** |
| "Δem(b) is the multiplicity-robust headline" | §1.5 / Abstract | — | Headline is on the *non-anchored arm*; the *anchoring task* clause is 1/5 CI-clean (FATAL-B) | **Survives — headline pivot is itself a kind of overclaim by relabel** |

### B. N=1 / cherry-pick hunt

- **§1.5 / §6.2 / §6.5 / §7 — entire E6 chain on N=1 model (`llava-onevision-qwen2-7b-ov`).** §3.3 hedge in place; §1.5 contribution language not. **FATAL-A.**
- **§4.5 / §4.6 — γ-β on N=1 architecture pair × N=1 dataset (Qwen3-VL Instruct vs Thinking, MathVista).** §1.5 auxiliary observation correctly demoted. ×12.7 ratio still in §1.4 with no CI (MAJ-7).
- **§4.6 14/84 cells — post-hoc cell selection from 84-cell sweep, dressed as Bonferroni verification (MAJ-1).**
- **§6.2.2 cell #17 selection from 27-cell grid (MAJ-2).** Bonferroni-20 corrects within-evaluation; does not correct grid-selection.
- **§4.5 PlotQA "single-dataset depth panel" — selected as Main matrix because "paper-wide pattern이 가장 또렷이 분리되는" (per §4.1 line 132). Other datasets (TallyQA, InfoVQA) per §C.3 show the same pattern but with smaller magnitude. PlotQA selection itself is paper-pattern-conditional.**

### C. Confound hunt

- **§6.3 b-arm em +8.8 pp**: Alt-1 general regularization, Alt-2 numeric mode-collapse, both surfaced in §6.3 Insight 1.5 as competing explanations, both unfalsified. **MAJ-9.**
- **§7 HallusionBench +2.21 pp**: Paper interprets as confirming wrong-base error mode hypothesis. Alt: the projection reduces the model's overall confidence on perception-illusion items; HallusionBench scoring favors lower confidence on hallucination prompts. Has not been controlled.
- **§4.3 PlotQA un-mitigated free-lunch**: §4.2 Insight 3 (line 173) reports em(a) > em(b) on 5/6 PlotQA models — and explicitly notes this is *because S1 cutoff puts anchor within ±10% of GT*. So the "free-lunch baseline pattern" is an *artefact of the S1 cutoff design*, not a discovery. The paper frames it as a mitigation domain ("§6.2의 free-lunch mitigation이 이 PlotQA baseline 패턴을 5-dataset에 일반화 가능한 복구 메커니즘으로 변환"); this is positive framing of an S1-cutoff design choice.
- **§5.2 single-layer null on qwen2.5-vl** uses VQAv2 reference layer L=22, not PlotQA-calibrated. Round 1 MIN-7 raised; Round 1 MIN-5 acknowledged. The 5/5 single-layer null is therefore 4/5 PlotQA-calibrated + 1/5 VQAv2-cross-calibrated. Mild confound.
- **(a − m) Telea inpaint texture**: representation-level texture cues that OCR pixel-level absence does not control for. **FATAL-D.**

### D. Negative result suppression hunt

- **CAA-at-K=1 on (a − m) calibration**: deferred to §8.4 item 4. Three rounds. (Round 4 archive aggressive raised this.) **MAJ-8.**
- **ITI multi-head**: deferred. (Round 3 CRIT-N1 raised the multi-head trap; Round 3 reviser added Note 1 acknowledgement; empirical row deferred.)
- **Random-K=8 baseline** (Alt-1 falsification): deferred to §8.4 item 2. Round 1 MAJOR-8 raised. Three rounds. **MAJ-9.**
- **Numeric mode-collapse** (Alt-2 falsification): deferred to §8.4. Round 1.
- **Eigenvalue spectrum study on (a − m) at L=26**: deferred to §8.4 item 1 ("가장 cheap한 rigor 향상"). Three rounds.
- **Non-perfect-square / AnyRes mechanism panel**: §D.1 says OneVision routed via AnyRes; the 4-model perfect-square mechanism panel restricts to gemma4-e4b / llava-1.5 / convllava / qwen2.5-vl. The non-perfect-square cluster is partly suppressed by panel-scope choices.
- **CAA / ITI fair-tuning-effort cells**: §6.5 Note 2 explicitly predicts ITI fair-tuned could enable 4-clause partial-pass at some attention-head locus K-sweep cell. Yet the empirical row is deferred. **The paper has predicted that ITI under fair-tuning might pass the criterion but has not run the experiment.** This is a near-admission of overclaim that survives because the experiment is deferred.

### E. Statistical rigor

- **Bonferroni-20 on §6.2.3** is correct conditional on §6.2.2 cell pre-registration. If 27-cell grid is part of the multiplicity family, correct correction is Bonferroni-540 not Bonferroni-20. **MAJ-2.**
- **§4.6 Bonferroni-84** counts each (K, layer, statistic) cell as a test; framework prediction does not specify which cell. Post-hoc cell selection labeled as Bonferroni-clean. **MAJ-1.**
- **§4.5 ×12.7** has no CI. Three rounds requested; deferred. **MAJ-7.**
- **§4.2 (a − m) gaps** on n_wb = 152 / 211 / 403 have no CIs. Round 1 MAJOR-5 partially addressed via "load-bearing magnitude restricted to PlotQA + MathVista". (a − m) "5/5 dataset (a − m) > 0" headline relies on three point-estimate-only cells.
- **Multiple comparisons across 5 datasets × 4 metrics**: Bonferroni-20 applied — but: across-§6.2.2-grid (×27) not corrected; across-§4.6-sweep (×84) post-hoc-selected; across-§7 (6 benchmarks) Bonferroni-6 applied post-hoc.

### F. Reproducibility

- **Random seeds**: FLUX `seed_base 1729 + digit/scene_offset` documented in §A.4. (m) inventory is "no random seed" — Telea inpaint is deterministic but float-precision sensitive.
- **HF model commit hashes**: not pinned in §3.3. The model panel is named (`llava-hf/llava-interleave-qwen-7b-hf`, `Qwen/Qwen2.5-VL-7B-Instruct`, etc.) but no commit hashes. Reproducibility against HF model updates is at risk.
- **Inference seeds**: greedy decoding (temperature 0); reproducible in principle. But model rollouts can vary across hardware (FP16 vs BF16 vs FP32 reduction order).
- **§A.5 reproducibility appendix**: pointer to canonical CSVs; but per Round-1's stated workflow, those CSVs are gitignored and only available in `outputs/` of the local repo. Public release pre-submission is not committed in the paper.

### G. Missing comparisons

- **Finetuning on debiased data** (LoRA-DPO): MIA-DPO LoRA is in Table 8 but with em side-effect; no other finetuning baseline.
- **CFG (classifier-free guidance) pruning**: not compared. Could be a relevant baseline for VLM hallucination-reduction (Wang et al. 2024 etc.).
- **CoT prompting that explicitly tells the model "ignore the second image"**: Lou and Sun 2024 [arXiv:2412.06593] reports CoT/Reflection/Ignoring-Anchor-Hints are insufficient for *text* anchoring; the paper cites this in §2 as motivation for representation-level intervention. But the paper doesn't run the *prompt-level* baselines on its own VLM stimuli. The motivation rests on prior LLM literature without VLM verification.
- **CAA at K=1 with the same calibration set as E6**: deferred (MAJ-8).
- **ITI with multi-head locus K-sweep on residual stream or attention-head**: deferred.
- **Random-K=8 SVD on non-anchor calibration data**: deferred (MAJ-9).
- **Rejection sampling at inference** (filter answers that match anchor): not considered.

### H. Theoretical depth

- **§6.4 Insight 2 "K=8 sweet spot"**: empirical only on K ∈ {2, 4, 8}. Round 1 MAJOR-7 raised; deferred to spectral study (§8.4 item 1).
- **§5.2 multi-layer redundancy**: asserted; the formal claim is "single-layer ablation produces null Δdf with bootstrap CI overlap zero on 5/5 mech panel models + 5/5 OneVision datasets." This is the right empirical claim. The *interpretation* (signal is redundantly distributed across layers) is one of two: (i) signal is duplicated across layers (true redundancy), (ii) signal at each layer is small and adding many small effects produces the upper-half significant effect (additive accumulation). The paper does not distinguish.
- **Subspace linear assumption**: §6.2.1 SVD on `D[:, L, :]` assumes linear separability of the anchor representation. If the digit-pixel signature is *non-linear* in the residual space (e.g., manifold-like), linear projection misses it. No ablation: kernel SVD, autoencoder bottleneck, etc.
- **§5.4 framework is qualitative narrative**: no inequality, no closed form, no scalar prediction. Round 3 CRIT-N3 forced "통합 설명 framework" relabel.

### I. Framing

- **"Strict free-lunch"**: Round 2 converged to "4-clause free-lunch". The criterion is well-defined and Chand-et-al-positioned. **But**: 4-clause free-lunch's Δdf clause is 1/5 CI-clean (FATAL-B); the headline is on the third clause not the first. The "passes 4-clause" reading slides this over.
- **"Three-gate signature" / "two gates" / "Confidence × digit-pixel × plausibility"**: §1.2 has three pillars; Abstract has "digit-pixel × uncertainty 두 gate의 conjunction"; §1.3 then has "두 gate의 conjunction + plausibility 조건". These are inconsistent — sometimes three pillars, sometimes two gates with a plausibility window. Plausibility is sometimes a gate, sometimes a condition.
- **"VLM-first reasoning amplification"**: Round 3 reformed to "N=1 architecture × N=1 dataset existence proof". Round 4 picks up: even granting the result, this is one model pair on one dataset — no longer a paper-tier finding, more an existence-proof for follow-up. Auxiliary observation status is correct (Round 3); but the §1.4 sub-section retains 5 lines of body prose for an N=1 × N=1 result.

### J. Venue fit (Main vs Findings) — honest verdict

**The paper's roadmap acknowledged behavioral probing alone landed in Findings in prior work.** Now: the paper has added (i) mechanism panel single-layer null + multi-layer redundancy synthesis, (ii) E6 mitigation with 4-clause free-lunch criterion, (iii) γ-β residual-stream bridge.

Substance vs Findings/Main bar:
- **§4 behavioral**: comprehensive, but is the *Findings-bar* substance.
- **§5 mechanism**: 5/5 single-layer null + OneVision 5/5 confirm + dataset-dependent peak — this is real work but does not reach the depth of Weng et al. EMNLP 2024 Main (causal mediation). The §5.4 framework is qualitative-narrative, not formal causal mediation.
- **§6 mitigation**: E6 is the Main-tier candidate IF the cross-architecture transfer were demonstrated. As N=1 model, it is *Findings-bar* mitigation (proof-of-concept).
- **§7 capability preservation**: 6-bench/8-bench macro at single-arch — supports E6 but does not extend it to multi-arch.
- **§4.6 γ-β residual-stream bridge**: 14/84 Bonferroni cells with K=1 vs K=8 partial-falsification of the paper's deployed K=8 — interesting but not load-bearing for either contribution.

**Honest verdict: Findings-tier work positioning for Main on N=1 scope-overclaim. Recommend Findings, with the conditions in "Recommended action" above.**

The user's own roadmap acknowledged this risk earlier. Three rounds of constructive review cleaned the *form*; they did not move the substance from Findings to Main. The contribution surface is one mitigation on one model with one CI-clean Δdf cell out of five and one multiplicity-robust clause out of four. That is a Findings paper.

---

## What rebuttal would NOT change my mind

1. **"The §3.3 panel-scope hedge is in place; the central contribution sentence does not need to repeat it."** Form-correct, function-incorrect. The §1.5 contribution is the most-cited surface; it must carry the scope. (FATAL-A)

2. **"Δem(b) Bonferroni-clean is the multiplicity-robust headline; reviewers should read this as the paper's claim."** Δem(b) is the *non-anchored arm* clause; the anchoring task clause is Δdf and is 1/5 CI-clean. The paper cannot pivot from Δdf to Δem(b) and call the result "anchoring mitigation passes free-lunch". (FATAL-B)

3. **"§5.4 framework is honestly relabeled to *통합 설명 framework*; CRIT-N3 was addressed."** The relabel is correct. The §4.6 prospective leg still tests at K=1 while the deployed mitigation is at K=8 — the mismatch survives the relabel. (FATAL-C)

4. **"The (a − m) Telea inpaint is OCR-verified; no digit pixels remain."** Pixel-level absence does not address representation-level texture confounds. (FATAL-D)

5. **"Three rounds of review have addressed every issue at the form level."** Yes, and that is the diagnosis: the form is clean, the substance is single-cell single-architecture single-dataset.

## What rebuttal would change my mind

1. **Run CAA-at-K=1 on the same PlotQA + InfoVQA pooled (a − m) calibration as E6 on OneVision at L=26.** Report Δdf and Δem(b) on 5 datasets. If CAA-at-K=1 backfires as predicted, §6.5 has empirical content; if not, the unique-passage claim is empirically falsified. ~4-8 H100-hours. (Closes MAJ-8.)

2. **Run random-K=8 SVD on non-anchor calibration data on OneVision at L=26.** Report Δdf, Δem(a), Δem(b) on 5 datasets. If random-K=8 reproduces the +8.8 pp Δem(b), the headline is general regularization. If not, the (a − m) subspace is anchor-specific. ~1 H100-day. (Closes FATAL-D / MAJ-9.)

3. **Run E6 calibration on a second architecture** (e.g., Qwen2.5-VL-7b at its peak L=22 or Gemma3-4b at L=5). Report Δdf and Δem(b) on the 5 datasets. If the cross-architecture E6 produces sign-clean Δdf, the "mitigation" claim has cross-arch grounding. If it does not, the case-study scope is the honest claim. ~3-5 H100-day. (Closes FATAL-A.)

4. **Pre-register a single (K, layer, statistic) cell on the §4.6 sweep** before running the bootstrap, and report whether that single cell is Bonferroni-clean. Replaces the post-hoc 14/84 cell selection. ~free (just rerun the analysis with pre-registered cells). (Closes MAJ-1.)

5. **Compute paired-bootstrap CI on the ×12.7 ratio.** Use the same `gamma-beta-bridge-evidence.md` data; resample sids, recompute correct-base df ratio per resample, percentile CI. ~2 H100-hour. (Closes MAJ-7.)

Any one of (1) (2) (3) closes a FATAL. The combination of (1) + (2) + (4) + (5) without (3) leaves FATAL-A standing — the paper is still N=1 model. A genuine venue-Main verdict would require (3).

---

## Final

**Decision:** REJECT for Main. Borderline acceptable for Findings only after FATAL-A and FATAL-B are surfaced honestly in §1 / Abstract.
**Confidence:** high. Three rounds of constructive review addressed form; substance issues remain unaddressed.
**One-line:** *Three rounds of cleanup produced a clean draft of a Findings paper that markets itself as a Main paper — the cleaning was successful, but the substance is one CI-clean Δdf cell on one dataset on one model, with the multiplicity-robust headline pivoted to a non-anchoring side-effect clause.*
