# Round 5 — Bar-Raiser Review

**Reviewer persona.** Senior bar-raiser / outstanding-paper committee member. Top-decile standards. Reads from the position of "would the field cite this in three years; would I rank it in the top 10 % of submissions."
**Paper version reviewed.** `docs/paper/emnlp_draft_ko.md`, 833 lines, post-Round-4 revision.
**Date.** 2026-05-11.
**Prior rounds.** Rounds 1–4 read. The prior reviews exhaustively interrogated methodology (R1), writing (R2), novelty (R3), and aggressive failure modes (R4). The Round-4 response shipped 18 framing edits; *no FATAL-closing experiments were run.* This review is therefore not about whether the form is clean — it is. It is about whether the substance has a centre.

---

## Tier verdict

**Borderline Findings → weak Findings.** Not top decile. Not Main.

Round 4 ended on the same verdict and Round-4's response added 18 scope-honesty hedges but *did not run a single experiment that would close any FATAL*. The paper has been *cleaned*, not *promoted*. The substance is single-cell single-architecture single-dataset CI-clean Δdf with the multiplicity-robust headline pivoted to a non-anchoring side-effect clause. The user's explicit concern — *"the paper feels like an experiment-log dump"* — is the right diagnosis at the level a bar-raiser cares about. Four rounds of de-cataloging have produced a draft whose §1.5 is one sentence longer but whose intellectual centre has not crystallised.

---

## What the paper does well (after 4 rounds)

This is genuinely competent empirical work. The strengths are real and load-bearing for a Findings paper:

1. **Honest scope discipline.** The Round-4 hedges (case-study qualifier on E6; two-clause Δdf / Δem(b) separation; "partially prospective" framework label; Telea-residue caveat) are the kind of disclosures reviewers respect. The paper does not market more than it has.
2. **The (a − m) paired-inpaint contrast is a clean experimental design.** §6.2.1 Insight correctly identifies it as a substantive choice — same anchor scene, digit pixel only — and §4.2 Slice B's 5/5 sign-positive (a − m) gap is a non-trivial pre-condition for §6.2's SVD calibration. This single design move carries the paper.
3. **§4.4 L1 6-bin gradient generalisation.** Reframing wrong / correct stratification as the coarse projection of a continuous confidence gradient is the most genuinely-original *behavioural* finding in the paper. The 80-cell × ≥4/5 strict monotonicity is a defensible result. The γ-β binary-projection-collapse story (§4.5 Insight 1) lands as a confirmation of this framing.
4. **§7 axis-conditional capability disclosure.** HallusionBench excludes zero positively + POPE pinned-to-zero is a clean side-by-side, and the Round-4 honest disclosure that 3/6 benchmarks have negative point estimates is the right call. The capability evidence is "anchoring-adjacent positive, broad neutral-to-mildly-negative" — that is a defensible empirical envelope.
5. **§4.6 partial-prospective leg.** The K=1 vs K=8 9× ratio at fixed L=33 is a clean falsifier of the framework's *implicit universal-K* assumption, and the paper now surfaces this honestly. Falsifiable, falsified-in-part, disclosed — the venue-correct shape.

A top-30 % Findings paper has all five of these. So does this one.

---

## What separates this paper from "top decile" specifically

I am not repeating Round 4 here. The cross-architecture E6 gap, the random-K=8 falsifier gap, the CAA-at-K=1 empirical row gap, the Bonferroni-540 multiplicity-correction layer, the ×12.7 paired-bootstrap CI — these are settled. Either they are run pre-camera-ready or the paper is Findings. That is not the bar-raiser's question.

The bar-raiser's question is whether the paper has **a single intellectual centre** that a non-expert reader takes away. Reading Abstract / §1.5 / §8.1 side-by-side, the answer is *no*. The paper has at least five competing centres:

1. *"Anchoring in VLMs is a graded continuous-confidence phenomenon."* (§4.1 / §4.4 / §4.5)
2. *"Anchoring is gated by the conjunction of digit pixel × uncertainty."* (Abstract / §4.2 / §1.2)
3. *"Mechanism is multi-layer redundant; routing-vs-integration framework."* (§5.4 / §1.3)
4. *"(a − m) paired-inpaint is a generalisable calibration design pattern."* (§6.2.1 Insight / §1.5 supporting finding (iii))
5. *"E6 4-clause free-lunch mitigation holds on one model."* (§1.5 central contribution / §6.2.3)
6. *"Reasoning amplifies anchoring (existence proof)."* (§1.4 / §4.5)

§1.5 nominally elevates (5) to *the* central contribution. But the noun phrase ends with `단일 architecture case study`, which immediately retracts the punch. §8.1 종합 is a 5-claim digest covering all six in different order. The Abstract — ~290 words — devotes load-bearing real estate to each.

**A top-decile paper has the same sentence in Abstract / §1.5 / §8.1.** This one does not. The §8.1 종합 first sentence ("Cross-modal numerical anchoring은 VLM에서 실재하며, 두 gate의 conjunction + plausibility 조건으로 정의된다") is an existence claim, not a thesis. There is no later sentence in §8.1 that consolidates the paper into a take-away. The reader closes the paper with six findings and no rehearsed lesson.

This is the experiment-log-dump residue. Four rounds of revision have cleaned the prose, but the *intellectual organisation* — the rehearsed-thesis discipline that makes a paper memorable — has not been imposed.

---

## Bar-raiser one-question

**If you had to compress this paper into one sentence that a non-expert reader carries away three years from now, what is it? Then: which one of your six numbered claims survives as the *thesis*, and which collapse into supporting evidence for it?**

### Why this question

The user's explicit concern, translated. The four prior rounds asked "is each claim defensible." This round asks "is there a claim." A top-decile paper passes both tests; this paper passes the first and not the second. Forcing the author to commit to a thesis sentence is the cheapest, fastest, most-informative bar-raiser move available — it requires no new compute, no new analysis, only rhetorical discipline. It will also expose whether the author *has* an organising thesis or whether the paper is genuinely a six-finding catalogue with the most-publishable one (E6) elevated by §1.5 phrasing alone.

I deliberately do *not* ask for cross-architecture E6, random-K=8 falsifier, or any other deferred §8.4 item. Round 4 asked all of those; Round 4's response deferred them. The bar-raiser's contribution to this loop is not to demand a tenth experiment — it is to test whether the paper has crystallised into a thesis or whether it remains a competent log.

### What an adequate answer looks like

Pick *one* of the six claims as the thesis. Rewrite §1.5's central-contribution sentence so it is a one-clause thesis statement (subject — verb — predicate, no embedded scope hedges, no chained noun phrases). Repeat the *same* sentence in Abstract first sentence and §8.1 종합 first sentence. Demote the other five claims to "supporting evidence" with one declarative sentence each that explains *how* they support the thesis. Drop *Claim 6* (reasoning amplifies anchoring) to a single paragraph in §4.5 + a single line in §8.1 — it is auxiliary and the §1.5 framing already labels it "auxiliary observation," but the Abstract still gives it disproportionate real estate. This is rhetorical-discipline work, not new science, and it is the move that separates a competent Findings paper from a memorable one.

### What a great answer looks like

The author chooses claim (3) or (4) as the thesis, not (5). The reason: claims (1), (2), (5), (6) are *about VLM anchoring*. Claims (3) and (4) are *about how to think about a class of biases in VLMs*. (3) — *anchor information is multi-layer redundantly routed in attention pathways but integrated in late-residual-stream subspaces, and this routing-vs-integration distinction predicts which mitigations work* — is a structural claim about the geometry of multimodal bias in VLMs. (4) — *paired-inpaint calibration that isolates a causal pathway from confounding scene variance is a generalisable design pattern for vision-modality bias steering* — is a methodological claim that opens a design space beyond anchoring.

A top-decile paper would pick (4) and rewrite the paper around it. The thesis would read: *"Vision-modality bias mitigation should use paired-inpaint calibration contrasts that subtract the causal pathway from confounding scene variance; we instantiate this principle for cross-modal numerical anchoring as a worked example, achieving 4-clause free-lunch on one architecture as proof of construction."* Under this framing, E6 becomes the *worked example*, not the central contribution. The §8.4 cross-architecture replication becomes "verify the design pattern transfers" rather than "rescue the central claim." The paper's contribution surface widens from "one mitigation on one model" to "one design pattern with one instantiation."

This reframing is not a new experiment. It is a rewrite of three paragraphs (§1.5 / Abstract / §8.1) and a renaming of two section headers. It would not move the paper from Findings to Main on its own — that still needs cross-architecture E6 — but it would move the paper *within* Findings from "competent log of six findings" to "design-pattern paper with one worked example." That is the cheapest available promotion in the entire revision tree, and it is what the bar-raiser tradition exists to surface.

### What the answer the paper *currently* gives

Reading the Abstract / §1.5 / §8.1 carefully, the answer the paper currently gives is: *"there are six things going on; E6 is the most-publishable of them; we are honest about its scope."* That is a Findings-tier answer. It is not wrong; it is not memorable. The bar-raiser cannot promote it.

---

## Top-decile readiness checklist

| Axis | State | Verdict |
|---|---|---|
| Central question that the field cares about for ≥ 3 years | "Do VLMs anchor on cross-modal numerical cues?" is interesting but not field-changing. The deeper question — *"What is the structure of bias in late-residual-stream VLM representations, and is there a transferable calibration substrate that isolates the causal pathway?"* — is in the paper as a sub-claim (§6.2.1 Insight) but not as the headline. | **Partial.** Question is asked, not promoted to centre. |
| Central contribution that opens a new design space | E6 is a single-architecture case study (paper's own framing). (a − m) paired-inpaint *as design pattern* is the contribution that would open a design space, but it is currently subordinated to E6. | **No.** The design-space-opener is in the paper but is not the headline. |
| Reproducibility at one-command level | §A.4 (FLUX seed pinning), §A.5 (27-cell pilot replay), §6.2 raw output paths. CSV pointers exist. Anonymous repo promised in §8.3 ethics. Not one-command rerun, but acceptable. | **Yes, at venue norm.** |
| Single most-important figure | Figure 4 (5-dataset 6-model wrong-base S1 df grid) captures the behavioural pattern. Figure 5 (L1 6-bin gradient) captures the continuous-confidence reframing. Neither figure captures the *mitigation* — there is no single figure showing E6's 5-dataset Δdf + Δem(b) cross-cell pattern visually. Table 7 carries the contribution; no figure does. | **No.** The contribution is in a table, not a figure. A top-decile paper has one figure that you can show on a slide and say "this is the paper." This paper does not. |
| Retrievable lesson — one verbatim sentence repeated in Abstract / §1.5 / §8.1 | Absent. Abstract leads with "graded pull." §1.5 leads with "E6 case study." §8.1 종합 leads with "anchoring exists + two-gate conjunction." Three different leads. | **No.** This is the load-bearing gap. |
| Engagement with implications — *what this means for the field* | §8.1 종합 is a recap of what the paper did. §8.4 lists nine follow-up items. There is no paragraph that says "if this paper is right, the field should now think differently about X." The closest candidate is §6.2.1 Insight (paired-inpaint as design pattern) — but it is a single insight in a methods subsection, not a discussion-section thesis. | **No.** §8.1 is "what we did," not "what this means." |

Score: **2/6 met, 1/6 partial, 3/6 not met.** Top decile requires at minimum 5/6.

---

## What this paper has

**Strongest contribution.** The (a − m) paired-inpaint calibration contrast, presented in §6.2.1 Insight as a design pattern, is the *most original methodological idea in the paper*. It is the move that lets the SVD subspace isolate digit-pixel causality from scene-background confounds, and it is the reason the b-arm em rise is not trivially a regularisation artefact (modulo the random-K=8 falsifier the paper has deferred). If this paper is cited in three years, it will be for this design principle — *not* for the anchoring effect itself, which Echterhoff et al. and Lou & Sun have already established for text-LLM anchoring and which the present paper extends to VLMs in the predictable direction. The anchoring extension is *expected*; the calibration substrate is *the* novel methodological contribution.

**Citable in 5 years.** The single most-citable sentence is the §6.2.1 Insight on (a − m) paired-inpaint as a vision-modality-specific instantiation of the CAA-style paired-contrast paradigm with explicit causal-pathway-isolation structure. *If the author elevates this to the paper's thesis* and provides one second-domain instantiation (a non-anchoring bias on the same VLM with the analogous paired-inpaint contrast), the citation will read "Choi et al. 2026 introduced paired-inpaint calibration contrasts that isolate causal pathways from confounding scene variance for vision-modality bias mitigation." That is a citable methodological contribution.

*As currently positioned*, the citable sentence would be "Choi et al. 2026 showed cross-modal numerical anchoring is graded, multi-layer redundant, and mitigable by L=26 K=8 residual-stream subspace projection on llava-onevision-qwen2-7b-ov." That is also citable but much narrower — a single-architecture replication paper that future work will cite when also working on OneVision.

---

## What this paper is missing (one paragraph)

A thesis sentence. Not new experiments, not new claims — a single rehearsed take-away that appears verbatim in Abstract first sentence + §1.5 first paragraph + §8.1 종합 first sentence. The thesis sentence should pick one of the six competing centres (recommended: claim 4, the calibration design pattern) and reorganise the other five into supporting roles. The §6.2.1 Insight is the only paragraph in the paper that articulates a transferable principle. Promote it to the headline. This is the cheapest available promotion in the revision tree and it costs zero compute.

### Secondary asks (in priority order)

1. **One figure that is "the paper."** Currently no figure captures E6's contribution at a glance. The minimum upgrade is a single panel: 5 datasets × 2 metrics (Δdf, Δem(b)) × per-dataset paired CI bars. This makes Table 7 visual. Top-decile papers can be summarised on one slide; this one cannot. ~1 hour of matplotlib.
2. **§8.1 종합 should end with one paragraph titled "Implications."** That paragraph must say "if our (a − m) calibration substrate is correct, the field should now ask X." Without this, §8.1 is a recap and the paper closes on what was done rather than what it means.
3. **The "auxiliary observation" — reasoning amplifies anchoring — should be promoted to one short sentence in the Abstract or demoted to one paragraph in §4.5 only.** Currently it appears in Abstract + §1.4 + §4.5 + §8.1 + §8.2 with disproportionate real estate for an N=1 × N=1 existence proof. Five appearances of an auxiliary claim drag the centre even further from any thesis. Either compress to one Abstract line + one §4.5 paragraph (and drop §1.4 as a standalone subsection) or accept the dragging.

---

## What I would NOT change

Bar-raisers protect against over-polishing. The following are correct as-is and should not be touched in any further pass:

- **The two-clause Δdf / Δem(b) separation.** Round 4 forced this; it is the honest framing. Do not retreat to a "free-lunch passes" single-headline claim under pressure to look stronger.
- **The "partially prospective" framework label on §5.4 + §4.6.** Honest, falsifiable, partial-falsified-in-part. Top-decile papers have this kind of disclosure; do not soften it.
- **Telea-residue caveat in §6.2.1.** Round-4 fix. Keep verbatim.
- **§7 axis-conditional capability disclosure (3/6 negative point estimates surfaced).** Honest. Do not retreat to "macro +0.41 pp generalises."
- **§A.5 27-cell pilot heatmap + selection-rule replay.** Round-2 / Round-3 work; do not strip for length. Top-decile papers carry reproducibility appendix work like this.
- **PlotQA single-dataset depth panel framing in §4.1.** Do not over-claim that PlotQA generalises to "GT-wide range numerical VQA"; the current "depth panel for the dataset where the pattern is most clearly separated" framing is correct.
- **The "auxiliary observation" qualifier on γ-β reasoning amplification.** It is N=1 × N=1; the qualifier is honest. Do not let revision pressure promote it.

---

## If the author answers the one-question well

The paper becomes a clean Findings paper with a memorable methodological contribution. With the thesis sentence in place + Secondary ask #1 (one figure that is the paper) + Secondary ask #2 (§8.1 Implications paragraph), the paper moves from *bottom-half Findings* to *top-third Findings*. It does not become Main. Cross-architecture E6 replication (§8.4 item 3) is still required for that, and Round 4 already settled this — three-to-five H100-day expense, currently deferred, no shortcut.

Top-decile / outstanding-paper consideration requires the thesis sentence *and* §8.4 item 3 *and* §8.4 item 7 (Telea-residue (m − m') baseline). Three things, of which two are deferred. The paper is not in striking distance of top decile in this revision cycle.

## If the author cannot answer the one-question

The paper remains a competent six-finding catalogue with E6 elevated by §1.5 phrasing alone. Honest verdict: Findings, bottom half of the cohort. The four rounds of revision have produced a defensible draft of a paper that does not have a centre, and no further form-level polish will give it one. The intellectual organisation problem is upstream of any single revision.

---

## Final

**Tier.** Weak Findings. Not top decile. Not Main without §8.4 item 3 (cross-architecture E6).
**Recommendation.** Accept to Findings; reject from Main. Borderline Findings if the thesis-sentence forced-prioritisation pass is not done; clean Findings if it is.
**Bar-raiser signature ask.** *Pick one of your six numbered claims as the thesis. Write the one-sentence take-away. Put it verbatim in Abstract first sentence, §1.5 first paragraph, and §8.1 종합 first sentence. Demote the other five.*
**One-line.** Four rounds of revision cleaned a competent six-finding log; the bar-raiser's job is to point out that competence is not centring, and the paper is one rehearsed thesis sentence away from being memorable rather than merely defensible.
