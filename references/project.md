# Cross-modal anchoring in VLMs — paper plan + 2026-04-23 feasibility review

This document has two parts:

1. **§0 (top, current as of 2026-04-28)** — operational paper outline.
   `references/roadmap.md` is built against this. Read this first.
2. **2026-04-23 feasibility review (below the divider)** — original candid
   feasibility review. Preserved for prior-art landscape, novelty matrix,
   acceptance-bar reasoning, and decision provenance. When the two appear
   to disagree, §0 wins; the review remains as historical context.

---

## §0. Current paper outline (as of 2026-04-28)

**Target venue.** EMNLP 2026 Main, ARR May 25.

**Compute envelope.** 8 × H200 (one shared with vLLM Qwen2.5-32B server,
~60 GB usable per GPU), one month at writeup.

### §0.1 Headline claim

> **Cross-modal numerical anchoring in VLMs is uncertainty-modulated graded
> pull, not categorical capture, and concentrates on a digit-pixel cue
> inside the anchor image.**

Three motivating features carry the §1 lead:

1. **Setup novelty.** No prior work delivers a stand-alone rendered-number
   image as a cross-modal anchor for open numerical VQA. See "novelty
   verdict" in the 2026-04-23 review below for the full prior-art matrix
   and differentiation against VLMBias / typographic attacks / FigStep /
   Tinted Frames / the LLM-anchoring lineage.
2. **Graded vs. categorical.** Phase A
   (`docs/insights/A1-asymmetric-on-wrong.md`) shows wrong-base
   direction-follow exceeds correct-base by **+6.9 to +19.6 pp** on every
   model in the 7-VLM main panel. Paired adoption (M1) is 2-6 % — the
   model rarely *outputs* the anchor literally. The mass of the effect is
   in graded movement toward the anchor, not literal copying.
3. **Cognitive-science alignment.** Uncertainty-proportional anchor pull
   matches Mussweiler & Strack's selective-accessibility model. §6
   generalises this from the binary correctness split to a continuous
   logit-based confidence proxy.

### §0.2 Paper section structure

| § | Title | Headline finding | Primary evidence | Status |
|---|---|---|---|---|
| 1 | Introduction | the §0.1 headline claim | A1 + E1d + E4 | writing |
| 2 | Related work | LLM anchoring lineage, VLMBias, typographic attacks, cognitive-science prior | (review below) | writing |
| 3 | Problem definition + canonical metrics | 4-condition prompt (`b/a/m/d`); JSON-strict template; M2 canonical `adopt_rate`, `direction_follow_rate`, `exact_match`, `anchor_effect = M(anchor) − M(neutral)` | `docs/insights/M2-metric-definition-evidence.md` | M2 refactor pending |
| 4 | Datasets + anchor inventory | VQAv2 number, TallyQA, ChartQA, MathVista. FLUX-rendered digit anchor inventory with mask + neutral counterparts | `inputs/`, fetcher / generator scripts | covered |
| 5 | Distance, plausibility window, digit-pixel causality | `adopt_rate` decays sharply with `\|anchor − gt\|`; digit pixel is operative cause (anchor > masked); per-dataset distance cutoffs validated; cross-model robustness | E5b + E5c + E5d + E5e | partial — cross-model expansion in flight; γ MathVista pending |
| 6 | Confidence-modulated anchoring (logit-based) | `direction_follow_rate` is monotonic with answer-token logit / probability; the wrong/correct binary in §A1 is a coarse projection of the same effect | per-token logit captured (commit `5f925b2`); analysis pending | analysis pending P0 |
| 7 | Attention mechanism + mitigation | anchor-image attention concentrates at per-encoder-family peak layer (E1b 4-archetype); single-layer ablation null but upper-half multi-layer ablation reduces df 5.5–11.5 pp on 6/6 models; mid-stack-cluster attention re-weighting at chosen `s*` reduces df 5.8–17.7 % rel with `exact_match` rising and `accuracy_vqa` invariant | E1 + E1b + E1d + E4 | done; digit-pixel-patch attention reanalysis pending P0 |
| 8 | Future work | LLM/VLM architectural diff (preferred); image-vs-text anchor; reasoning-mode VLMs at scale | scoped only | future |

### §0.3 Canonical metrics (M2)

Settled in `docs/insights/M2-metric-definition-evidence.md` (analysis on 25
`predictions.jsonl` files, 6+7 distinct rate variants). Choices win on rank
consistency across signals (wrong > correct, S1 > S5, anchor > masked):

```
adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pb-gt)·(pa-gt) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
anchor_effect_M       = M(anchor arm) - M(neutral arm)
```

Notation (canonical): `pred_b / pred_a / pred_m / pred_d / anchor / gt`,
booleans `pb_eq_a`, `pa_eq_a`, `gt_eq_a`, `pa_ne_pb`, `pb_eq_gt`. Same form
applies to `pred_m` (mask arm).

### §0.4 Model panel and dataset matrix

| dataset | conditions | models with full run | status |
|---|---|---|---|
| VQAv2 number | b / a / d (3-cond) | 7 main + 7 strengthen | done — M2 re-aggregation pending |
| VQAv2 number | b / a / m / d (4-cond, S1) | 0 — kept as P1 | deferred (kept) |
| TallyQA | b / a / m / d (4-cond, S1) | 3 (E5e) | done — extending |
| ChartQA | b / a / m / d (4-cond, S1) | 3 (E5e) | done — extending |
| MathVista | b / a / m / d (4-cond, S1) | 0 — γ planned | P0 |
| MathVista | reasoning-mode (β) | 0 — γ planned | P0 |

E5b stratified-distance and E5c digit-mask runs are on
`llava-next-interleaved-7b` only (cross-model expansion in flight).

Mechanistic panel (E1 / E1b / E1d): 6 models (`gemma4-e4b`,
`qwen2.5-vl-7b-instruct`, `llava-1.5-7b`, `internvl3-8b`, `convllava-7b`,
`fastvlm-7b`) on n=200 stratified.

Mitigation panel (E4): 3 mid-stack-cluster models full validation
(`llava-1.5-7b`, `convllava-7b`, `internvl3-8b`).

### §0.5 Scoped-out (paper-tier decisions)

- Cognitive-load / distractor breadth — saturated, not pursued.
- Salience / red-circle marker bias — crowded, not pursued.
- Confirmation bias as a separate category — folded into §6 confidence
  framing as a sub-claim of anchoring, not a standalone bias.
- Tinted Frames-style question-form framing — distinct paper, not pursued.
- Closed-model subset (GPT-4o / Gemini 2.5) — not pursued unless a clean
  500-sample reviewer-defuse run becomes cheap; otherwise §"limitations".
- Human baseline (Prolific 50 subjects) — not pursued; cost / time vs.
  yield does not justify on the current ARR clock.
- Dual Process Theory as organising frame — empirically contested
  (VLMBias / LRM-judging shows reasoning can amplify); folded into §8
  future work, not a §1 organising claim.

### §0.6 Reading order for new contributors / coding agents

1. This §0.
2. `references/roadmap.md` — operational status, what runs, what blocks.
3. `docs/insights/M2-metric-definition-evidence.md` — what metric
   definitions the headline numbers use.
4. `docs/insights/00-summary.md` — Phase A umbrella.
5. `docs/insights/E1d-causal-evidence.md` and
   `docs/insights/E4-mitigation-evidence.md` — mechanism + mitigation
   umbrellas.
6. The 2026-04-23 feasibility review below — prior-art context,
   scope-decision reasoning.

---

## Background: 2026-04-23 feasibility review

**Bottom line up front.** The core empirical claim—injecting a rendered *number image* as an anchor alongside a target VQA image, and finding that the effect is **asymmetrically stronger on items the model originally got wrong**—is genuinely novel and defensible against existing literature. However, the paper as currently scoped (7 models × ~5 cognitive biases × VQAv2, no mechanistic analysis, no mitigation) matches the empirical profile of recent cognitive-bias-in-LLM papers that landed in **Findings, not Main**, at EMNLP 2024–2025. To move from Findings-tier to Main-tier, the paper needs to (1) cut breadth and add *mechanistic depth*, (2) add at least a minimal mitigation guided by the mechanism, and (3) reframe around a single sharp scientific claim rather than a cognitive-bias grab-bag. With 8×H200 and one month, this is achievable—but only if the expansion plan is *narrowed*, not widened. The sections below lay out the prior-art landscape, a novelty verdict for each proposed extension, feasibility math, and concrete recommendations.

## The novelty verdict: strong core, weak extensions

The user's central setup has no direct precedent. No published work delivers a standalone rendered-number image as a cross-modal anchor in a multi-image VLM prompt and measures a regression-style shift toward the anchor on open numerical VQA. The closest neighbors each differ along an important axis: **Wang, Zhao, Larson (NAACL 2025, "Typographic Attacks in a Multi-Image Setting")** uses class-label text overlaid on target images for classification; **Nguyen et al. (VLMBias, ICML AI4MATH 2025)** injects subject-identifying labels to probe memorization priors on counterfactual counting, not numerical anchoring; **FigStep (Gong et al., AAAI 2025)** renders text-as-image for jailbreaks, not numerical bias; the "Biasing VLM with Visual Stimuli" note pre-marks multiple-choice answers, not numerical values. The LLM-anchoring parent literature (Jones & Steinhardt 2022; Echterhoff et al. EMNLP Findings 2024; Lou & Sun 2024; O'Leary 2025) is entirely text-only.

**The "stronger on originally-wrong cases" asymmetry is also novel.** No prior LLM or VLM anchoring paper partitions its effect by prior correctness. Lou & Sun (2024) found stronger models anchor *more*, but did not condition item-wise; Jones & Steinhardt merged correct/wrong into an aggregate "functional accuracy drop." This asymmetry has clean cognitive-science grounding (Mussweiler & Strack; Jacowitz & Kahneman — anchoring scales with subjective uncertainty) and is the paper's strongest intellectual hook.

**Dual Process Theory framing is underexplored but contested.** Hagendorff et al. (Nature Computational Science 2023) applied System 1 / System 2 to LLMs; Brady et al. (Nature Reviews Psychology 2025) reviewed the framework for LLMs. For VLMs, there is no published application. But evidence is *mixed*: VLMBias found reasoning VLMs (o3, o4-mini) were **more** biased by in-image textual cues than non-reasoning models, and Wang et al. "Judging Bias in Large Reasoning Models" (2025) shows reasoning amplifies some biases. The user's DPT narrative ("System 2 suppresses anchoring") needs to engage these counterexamples carefully, not ignore them.

The table below summarizes novelty for each proposed extension:

| Extension | Closest prior art | Novelty verdict |
|---|---|---|
| **Cross-modal numerical anchoring (core)** | VLMBias (different setup); typographic attacks (different task) | **Genuinely novel** |
| **"Stronger on wrong cases" asymmetry** | None found in LLM or VLM anchoring | **Genuinely novel, strongest hook** |
| **Dataset expansion beyond VQAv2** | N/A — methodological hygiene | Required, not novel |
| **Thinking vs. instruction-tuned VLMs** | Lou & Sun 2024 (LLMs); Wang 2025 LRM judging; VLMBias | Moderately novel in VLMs; must engage mixed evidence |
| **ViT vs. Conv-encoder VLMs** | Typographic-attack mechanistic work (arXiv 2508.20570) | **Novel angle** — ConvLLaVA / DINO / EVE are excellent negative controls |
| **Framing: text vs. image modality for numbers** | Tinted Frames 2026 (question-form framing only); text-dominance literature | **Novel** — Kahneman-Tversky gain/loss framing via image vs. text is unfilled |
| **Salience bias via boxes/markers** | Shtedritski "red circle" 2023, FGVP, STER-VLM, spatial attention bias 2025 | **Crowded** — differentiation hard |
| **Confirmation bias (anchor near base prediction)** | VLMBias owns "memorization overrides vision" framing | **Partial** — user's version is actually a *sub-finding of anchoring*, not a distinct bias |
| **Cognitive Load Theory / distractors** | Idis (2025), I-ScienceQA, MVI-Bench, MM-NIAH, MIHBench — saturated | **Weak novelty** — heavy overlap |
| **Dual Process Theory as organizing frame** | Hagendorff 2023, Brady 2025 (LLMs only); not applied to VLMs | **Novel framing** but empirically contested |

## Prior art the paper cannot ignore

Eight works should be cited, differentiated, and in some cases directly compared against. **Nguyen et al.'s VLMBias (arXiv:2505.23941)** is the most dangerous neighbor and needs a full paragraph of differentiation: their "cues" are memorized-subject labels, they measure counting on counterfactual canonical images (not numerical shift toward an arbitrary anchor), they don't condition on prior correctness, and they use a different theoretical lens. **AIpsych (arXiv:2507.03123)** is the first psychology-grounded VLM cognitive-bias benchmark and must be cited as the closest comprehensive predecessor even though it covers sycophancy/authority/logical consistency rather than anchoring. **Jones & Steinhardt (NeurIPS 2022), Echterhoff et al. (EMNLP Findings 2024), Lou & Sun (2024), and "Understanding the Anchoring Effect of LLM" (arXiv:2505.15392)** are the canonical LLM-anchoring lineage; the last of these introduces A-Index and R-Error metrics the paper should adopt. **Hagendorff, Fabi & Kosinski (Nature Computational Science 2023)** is the canonical DPT-in-LLMs reference. **Goh et al.'s "Multimodal Neurons" (2021)** and **Wang, Zhao, Larson (NAACL 2025)** are the mechanistic ancestors the paper synthesizes. **Tinted Frames (arXiv:2603.19203)** covers question-form framing in VLMs and must be distinguished from the semantic gain/loss framing the user proposes.

A critical red flag: several related works—especially the typographic-attack literature—share the same *causal mechanism* (text rendered as pixels activates concept neurons) but frame it as adversarial attack rather than cognitive bias. **Reviewers will ask: "Why isn't this just typographic attacks, renamed?"** The answer must be crisp: (a) the injected content is a *numerical value without semantic label or class identity*, (b) the target task is *open-ended numerical estimation* with a measured regression coefficient, not classification flip or jailbreak ASR, (c) the framing connects to human cognitive science with testable predictions (asymmetry on uncertainty, DPT effects). This framing must be explicit, not implicit.

## What EMNLP Main actually demands for this paper type

The empirical base rate is unambiguous. Nearly identical papers on cognitive biases in LLMs—including Echterhoff et al. 2024 with 13,465 prompts, 5 biases, 4+ models, *and* a debiasing method—landed in **Findings**, not Main. "How Does Cognitive Bias Affect LLMs?" (2025) also went to Findings. CIVET (systematic position-understanding evaluation of VLMs) went to Findings. Meanwhile, Weng et al.'s "Images Speak Louder than Words" (EMNLP 2024 Main) combined causal mediation analysis with a 22% mitigation. **The pattern is decisive: black-box behavioral probing alone does not clear the Main bar, even when the scope is large and the findings are interesting.**

What does clear the bar is some combination of: **mechanistic or causal analysis** (attention patching, causal mediation, logit lens, SAEs, probing classifiers), a **mitigation method** even if simple, a **novel methodological framework** rather than new measurements of known phenomena, or a **genuinely surprising scientific claim** grounded in rigorous evidence. Reviewers on behavioral papers consistently cite: (1) lack of mechanistic depth, (2) prompt-sensitivity concerns, (3) "so what?" missing actionable takeaway, (4) loose use of psychology concepts, (5) missing human baseline, (6) single-dataset narrowness, (7) missing frontier closed models as reference points. The scope expectations are 5–15 models, ≥2 datasets, paraphrase robustness across 3–5 prompt variants with statistical testing, and a limitations section that genuinely limits.

## Feasibility with 8×H200 and one month: compute isn't the bottleneck

Raw compute budget is **~5,760 GPU-hours**, well above typical EMNLP VLM-evaluation papers (which run on 50–500 GPU-hours total). A 7B VLM on one H200 with vLLM processes roughly 50k–200k short-answer VQA samples per day; a 30B model, 10k–40k per day. Thinking-mode models with long CoT chains (>1k output tokens) run 10–20× slower and are the true bottleneck. The naive expansion plan—7 models × ViT/Conv × thinking/instruction × 4+ biases × 2+ datasets × paraphrases—explodes to roughly 3–10M inference calls and is only feasible if each bias uses a curated ~2k–5k subset rather than full datasets.

**Compute is abundant; scope discipline and mechanistic depth are the binding constraints.** Spending the last month running a wider grid instead of adding a mechanistic layer would be a strategic error.

## Recommendations ranked by impact on acceptance probability

**Highest impact: add a mechanistic component.** Even a moderate one dramatically changes the paper's profile. Concrete options in order of effort-to-payoff:
- Attention-mass analysis comparing attention to image-tokens vs. anchor-image-tokens vs. text-tokens under biased vs. neutral conditions — runs in hours with HuggingFace `output_attentions=True`, directly addresses the "why does this happen" reviewer complaint.
- Layer-wise logit lens to localize *when* the anchor's influence enters the computation.
- Vision-encoder ablation using **ConvLLaVA (pure ConvNeXt), EVE/Fuyu (encoder-free), and DINO-based VLMs** — directly tests whether the effect is encoder-mediated (expected if it inherits CLIP/SigLIP's typographic weakness documented in arXiv:2508.20570) or LLM-mediated. This is scientifically interesting, novel, and empirically tractable.
- Activation patching between biased and neutral runs to causally localize bias — following the Weng et al. EMNLP 2024 Main template.

**High impact: add a simple mitigation guided by the mechanism.** If attention analysis shows the anchor image draws disproportionate attention, an inference-time re-weighting or contrastive-decoding intervention showing 10–20% effect reduction flips the paper's profile from "observation" to "observation + explanation + fix." This is the single most reliable route to Main.

**High impact: sharpen the narrative to one claim.** Three candidate framings, in decreasing reviewer-excitement order:
1. *"Cross-modal anchoring in VLMs originates in the CLIP-family vision encoder and obeys a cognitive-uncertainty law."* Combines novelty (cross-modal anchoring), mechanism (encoder-level), and psychology (uncertainty-proportional bias).
2. *"Thinking VLMs do not reliably escape System-1 biases: a dual-process audit."* Requires reconciling with VLMBias's contrary evidence; if the user's data supports the claim, this is a highly citable headline.
3. *"Vision-encoder architecture shapes which cognitive biases VLMs inherit."* Leverages the ViT vs. Conv ablation as the organizing axis.

Any of these beats "we tested N VLMs for 5 cognitive biases."

**Medium impact: cut bias breadth, not model breadth.** Drop cognitive load / distractors (saturated) and confirmation bias (mostly redundant with the anchoring core finding). Keep anchoring as the centerpiece. Add **framing with text vs. image modality for numerical quantities** as the cleanest multimodal-native bias—this is genuinely novel and reviewers will see it as a targeted extension rather than a shopping list. If a fourth bias is desired, consider base-rate neglect or availability bias in VLMs, both genuinely unfilled gaps, rather than cognitive load or salience, both crowded.

**Medium impact: scope discipline.** Target **6–8 models along orthogonal axes** (size × encoder family × thinking vs. instruction) rather than 7 models sampled by convenience. Include **at least one frontier closed model** (GPT-4o or Gemini 2.5) on a subset—this defuses the "only open models" complaint reliably. Use **≥2 datasets**: TallyQA and CountBench for clean counting, ChartQA or DocVQA for in-image-number conflict (where the anchor competes with a legible number in the target image—an especially compelling condition), and MathVista for reasoning. Run 3–5 paraphrases with bootstrap confidence intervals and multiple-comparison correction.

**Low-cost but high-credibility: add a minimal human baseline.** A ~50-participant Prolific study replicating one or two of the bias conditions grounds the cognitive-science claims, answers a reviewer question before it's asked, and is doable in a week for <$500. This is disproportionately effective for psychology-framed papers.

## Red flags in the current plan

Five items in the current plan will attract reviewer fire. First, **confirmation bias as proposed ("if base prediction is 3 and anchor is 4, it's more easily pulled")** is not confirmation bias in the classical Wason/hypothesis-testing sense—it's a continuity-of-anchoring finding that the paper should present as a *sub-analysis of anchoring*, not a separate bias. Mislabeling it invites reviewers with psychology backgrounds to flag "loose use of cognitive-science terminology." Second, **cognitive load theory with distractors** overlaps heavily with Idis (arXiv:2511.21397), I-ScienceQA, MVI-Bench, and MM-NIAH, all recent and specifically about visual distractors; the differentiation cost likely exceeds the novelty payoff. Third, **salience bias via markers** is crowded ground (Shtedritski red-circle work, FGVP, STER-VLM, spatial attention bias) and will require substantial differentiation that competes with more novel axes. Fourth, **the Dual Process Theory framing is intellectually appealing but empirically contested**—VLMBias and LRM-judging work show reasoning models can be *more* biased, and the paper must engage this rather than presenting DPT as a clean predictor. Fifth, **VQAv2 alone will not survive review**; multi-dataset corroboration is non-negotiable for behavioral claims at EMNLP.

One additional risk worth flagging: the user should verify that every 2026 arXiv ID cited actually resolves to a paper. Several IDs surfaced in aggregator searches (e.g., 2603.xxxxx, 2604.xxxxx) may be genuine early-2026 preprints or may be search-aggregator artifacts—manual verification before citation is essential.

## A realistic one-month plan

Week 1: lock scope to anchoring (core) + framing (image vs. text modality) + one additional bias, 6–8 models along orthogonal axes, 3 datasets (TallyQA, CountBench, ChartQA/DocVQA), 3 paraphrases. Build the automated pipeline. Run attention-mass extraction alongside generation. Week 2: finish main-grid runs; begin vision-encoder ablation (ConvLLaVA, EVE, one DINO-based VLM) on a smaller prompt set. Launch 50-participant human baseline on Prolific. Week 3: mechanistic analyses (logit lens, attention patterns, encoder ablation) and prototype mitigation (contrastive decoding or attention re-weighting on anchor image tokens). Run closed-model subset (GPT-4o/Gemini 2.5) on a ~500-example stratified sample. Week 4: writing, statistical tests with multiple-comparison correction, limitations section, reviewer-question preemption. Submit May 25 ARR deadline.

## Conclusion: accept probability and what moves it

Honest assessment: the *current* plan—analysis-only, single-dataset, five biases loosely organized under DPT, no mechanism, no mitigation—has a realistic **Findings-not-Main** outcome based on near-identical recent precedents. The core phenomenon is novel enough and interesting enough that Findings acceptance is plausible; Main acceptance as currently scoped is not. The levers that meaningfully move Main-acceptance probability are, in order: (1) adding mechanistic analysis, especially the ViT-vs-Conv-vs-encoder-free ablation which is scientifically interesting and tractable on 8×H200; (2) adding a simple mitigation guided by that analysis; (3) narrowing to three well-chosen biases rather than five loosely chosen ones; (4) sharpening the narrative to one architectural or mechanistic claim; (5) multi-dataset corroboration with paraphrase robustness. Compute is not the constraint—design discipline is. The strongest single intellectual asset in the current preliminary results is the **asymmetric anchoring-on-wrong-cases finding**, which is genuinely novel, has clean cognitive-science grounding, and has no exact precedent in either the LLM or VLM literature. Building the paper around that finding rather than a bias catalogue, and investing the remaining compute in *mechanistic depth on anchoring* rather than *breadth across biases*, is the highest-expected-value path to EMNLP 2026 Main.
