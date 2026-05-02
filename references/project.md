# Cross-modal anchoring in VLMs — paper plan + 2026-04-23 feasibility review

This document has two parts:

1. **§0 (top, current as of 2026-04-28)** — operational paper outline.
   `references/roadmap.md` is built against this. Read this first.
2. **2026-04-23 feasibility review (below the divider)** — original candid
   feasibility review. Preserved for prior-art landscape, novelty matrix,
   acceptance-bar reasoning, and decision provenance. When the two appear
   to disagree, §0 wins; the review remains as historical context.

---

## §0. Current paper outline (as of 2026-05-02 — paper architecture restructure)

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
| 1 | Introduction | the §0.1 headline claim | A1 + E1-patch + E6 | writing |
| 2 | Related work | LLM anchoring lineage, VLMBias, typographic attacks, cognitive-science prior | (review below) | writing |
| 3 | Problem definition + canonical metrics | 4-condition prompt (`b/a/m/d`); JSON-strict template; M2 canonical `adopt_rate`, `direction_follow_rate`, `exact_match`, `anchor_effect = M(anchor) − M(neutral)` | `docs/insights/M2-metric-definition-evidence.md` | done — 3-model × 5-dataset matrix in flight |
| 4 | Datasets + anchor inventory | **5-dataset main matrix**: TallyQA, ChartQA, MathVista, **PlotQA, InfographicVQA**. FLUX-rendered digit anchor inventory with mask + neutral counterparts. VQAv2 dropped (multiple-GT, legacy metric, dataset size impractical at full eval). | `inputs/`, fetcher / generator scripts | 5/5 snapshots ready |
| 5 | Distance, plausibility window, digit-pixel causality | `adopt_rate` decays sharply with `\|anchor − gt\|`; digit pixel is operative cause (anchor > masked); per-dataset distance cutoffs validated; cross-model robustness on 3-model panel × 5 datasets | E5b + E5c + E5d + E5e | Phase 1 in flight — extends 4 → 5 datasets (+ PlotQA + InfoVQA, − VQAv2) |
| 6 | Confidence-modulated anchoring (logit-based) | `direction_follow_rate` is monotonic with answer-token logit / probability; the wrong/correct binary in §A1 is a coarse projection of the same effect | per-token logit captured (commit `5f925b2`); analysis pending | reaggregation pending Phase 1 outputs |
| 7.1–7.3 | Attention mechanism (analysis) | **Anchor pull mechanism is digit-pixel-patch attention concentration**, not full anchor-image attention. E1-patch on **5-model perfect-square panel** (gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b, **llava-interleave-7b**) shows digit/anchor attention ratio +24–40 pp above fair share at peak layer; patch-level causal ablation reproduces df reduction. | E1-patch (peak layer + bbox-restricted ablation) | Phase 3 (llava-interleave-7b add + patch-level ablation refactor) |
| 7.4 | E4 attention re-weighting mitigation | mid-stack-cluster attention re-weighting at chosen `s*` reduces df 5.8–17.7 % rel with `exact_match` rising — **panel extended to add llava-interleave-7b**, archetype-conditional behaviour reported | E4 Phase 1 + Phase 2 + Main extension | Phase 3 (add Main; archetype assignment from E1-patch) |
| 7.4.5 | E6 deployable mitigation | Subspace L31_K04_α=1.0 calibrated on **PlotQA + InfographicVQA pooled** generalises across 5 datasets at full gt range, no per-dataset tuning, no anchor labels at inference | E6 (Main model, 5-dataset full-range eval) | Phase 1 — recalibration + 5-dataset sweep |
| 8 | Future work | LLM/VLM architectural diff (preferred); image-vs-text anchor; reasoning-mode VLMs at scale | scoped only | future |

### §0.3 Canonical metrics (M2)

Settled in `docs/insights/M2-metric-definition-evidence.md` (analysis on 25
`predictions.jsonl` files, 6+7 distinct rate variants). Choices win on rank
consistency across signals (wrong > correct, S1 > S5, anchor > masked):

```
adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pa-pb)·(anchor-pb) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
anchor_effect_M       = M(anchor arm) - M(neutral arm)
```

Notation (canonical): `pred_b / pred_a / pred_m / pred_d / anchor / gt`,
booleans `pb_eq_a`, `pa_eq_a`, `gt_eq_a`, `pa_ne_pb`, `pb_eq_gt`. Same form
applies to `pred_m` (mask arm).

### §0.4 Model panel and dataset matrix

#### §0.4.1 Model tiering (paper-wide consistency, **revised 2026-05-02 v3**)

**Revision rationale (2026-05-02 v3)**: After v2 (Main=Gemma3-27B), reviewer-recognition / citation-friendliness analysis showed Gemma3 is Tier 3 in VLM analysis paper lineage (LLaVA / Qwen-VL are Tier 1 de facto standards). Switched Main to **LLaVA-OneVision-7B-OV** which is the LLaVA flagship 2025 (multi-image native by design + AnyRes high-res via 1-7 crops × 384×384 perfect-square per crop), matching reviewer-expected baseline. The AnyRes multi-crop layout breaks the §7.1-7.3 single-grid-bbox-routing assumption, so the mechanism panel runs as a 4-archetype perfect-square panel (Main not in §7.1-7.3 mech but is in §7.4 + §7.4.5). LLM-layer mechanism methods (E4 attention re-weighting, E6 subspace projection) include Main directly.

| Tier | Models | Used in | Role |
|---|---|---|---|
| **🟢 Main** | `llava-onevision-qwen2-7b-ov` (llava-hf/llava-onevision-qwen2-7b-ov-hf) | §3, §5, §6, §7.4, §7.4.5 — **all headline numbers** | Primary model. SigLIP-So400M-384 + Qwen2-7B LLM. Multi-image + video + single-image unified architecture (OV variant). AnyRes per-image dynamic crops up to 7 × 384×384 (within-crop 27×27 perfect-square; multi-crop layout overall). Tier 1 LLaVA family — direct successor to LLaVA-Interleave with chart-grade resolution restored. |
| **🟡 Sub-A (cross-family)** | `qwen2.5-vl-7b-instruct` | §3, §5, §6 | Tier 1 cross-family. Different ViT (dynamic-resolution Qwen-ViT non-square). Chart-domain pretrained. Strong base accuracy on chart datasets (~78% PlotQA/InfoVQA). Demonstrates anchoring as universal (non-zero on strongest baseline) but graded. |
| **🟡 Sub-B (cross-family + scale-up)** | `gemma3-27b-it` | §3, §5, §6 | Tier 3 family but useful as cross-family large-scale check (27B vs 7B Main). SigLIP-896 fixed perfect-square encoder (different resolution strategy from Main). Mid-tier accuracy (~51-71%). |
| **🔵 Mechanism panel (perfect-square, 4-archetype)** | gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b | §7.1–7.3 (E1-patch digit-bbox attention) | 4-archetype encoder panel: SigLIP-Gemma early / CLIP-336 mid-stack / ConvNeXt / FastViT. All perfect-square single-grid for clean bbox→token routing. **Main (OneVision) is NOT in §7.1-7.3** due to AnyRes multi-crop incompatibility; we instead include Main in §7.4 (E4 attention re-weighting) and §7.4.5 (E6 Subspace projection) which operate at LLM layer and are encoder-agnostic. Phase 2 may add OneVision-specific multi-crop bbox routing to extend §7.1-7.3 to Main. |
| **🟣 Reasoning ablation** | `qwen3-vl-8b` (instruct vs thinking) | §5 γ-β | Reasoning-mode amplification contrast. Single-purpose, disjoint from Sub-A. |
| **🟤 Legacy / breadth (appendix)** | `llava-interleave-7b`, gemma4-31b-it, gemma3-12b/4b-it, qwen3-vl-8b/30b, the 7-model VQAv2 panel | appendix only | LLaVA-Interleave preserved as direct predecessor reference. Gemma3 4B/12B held in reserve for potential within-family scale ablation in Phase 2. Historical 7-model VQAv2 panel preserved for behavioural breadth. |

#### §0.4.2 Dataset finalisation (5-dataset main matrix, finalised 2026-05-02)

| Dataset | Role | Notes |
|---|---|---|
| **PlotQA** (test V1) | **Main dataset** | Real-world chart values. 96K numeric Q-A post-filter (template ∈ {data_retrieval, min_max, arithmetic}, positive int, gt ≤ 10000). Wide gt distribution. **n=5000 stratified by gt-bin (5 × 1000)**. |
| **InfographicVQA** (val) | **Main dataset** | Different visual modality (infographic, not chart or natural image). **1,147 numeric (full)**. Heavy mid-range (gt-bin (20,100] = 479 samples — percent-heavy); cap of 5000 not binding since dataset is data-bound. |
| **TallyQA** (test) | Sub-dataset | Natural image counting (gt 0–8 after `answer_range=8` filter). **n=5000 stratified by gt-value (≤700 per gt; small-gt 0-4 fully filled, gt 5-8 capped by availability)**. Anchors §5/§7.4.5 small-gt regime. |
| **ChartQA** (test) | Sub-dataset | Chart values, mid-to-wide gt range. ~705 numeric (full); cap not binding. |
| **MathVista** (testmini) | Sub-dataset | Mixed math/science VQA. ~385 numeric integer (full testmini after filter); cap not binding. + reasoning-mode γ-β subset. |
| ~~VQAv2 number~~ | **DROPPED** | Multiple-GT (10-annotator vote), open-vocabulary text answers, full-eval impractical, legacy benchmark. Modern numeric VQA papers use TallyQA/Chart/Doc/Math/Plot/Info — VQAv2 not in this lineage. Existing 7-model panel data preserved in appendix for behavioural breadth supplementary. |

All 5 datasets evaluated under the same canonical setup: temperature=0, top_p=1.0, max_new_tokens=16, JSON-strict prompt, 4-condition (b / a-S1 / m-S1 / d). **Per-dataset n target = 5000 stratified by gt-bin** (PlotQA → 5000 fetch-time stratified across 5 gt bins, TallyQA → 5000 runtime stratified via `samples_per_answer=700` × `max_samples=5000`); InfoVQA / ChartQA / MathVista take their **full numeric subset** which falls below the 5000 cap (1147 / ~705 / ~385 respectively). The §7.4.5 sweep cap was raised from 500 → 5000 wrong-base sids per the 2026-05-02 statistical-power revision (evidence: wrong-base direction-follow rate is 3-5× stronger than correct-base across datasets, so a larger wrong-base sweep cell gives tighter mitigation effect estimates).

#### §0.4.3 Section-wise cell summary (Phase 1 target)

| Section | Models | Datasets | Conditions | n per cell | Cells |
|---|---|---|---|---|---|
| §3 main panel (3-model × 5-dataset) | Main(llava-onevision-7b-ov) + Sub-A(qwen2.5-vl-7b) + Sub-B(gemma3-27b) (3) | TallyQA, ChartQA, MathVista, **PlotQA**, **InfoVQA** (5) | b / a-S1 / m-S1 / d (4) | 1k–full | 15 |
| §5 distance + digit-mask | same 3 | same 5 | b + 5×a-strata + 5×m-strata + d (12, E5b/c) | 500–2500 | 15 |
| §6 confidence-modulated | same 3 | same 5 | reuses §3+§5 outputs | (reaggregation) | 0 new |
| §7.1–7.3 mechanism (E1-patch) | 4-archetype perfect-square panel (Main NOT included; Phase 2 adds OneVision multi-crop routing if pursued) | TallyQA + 1 chart + 1 info (3) | b + a + m (3) | 200 stratified | 12 |
| §7.4 E4 attention re-weighting | 4-model (mid-stack cluster: llava-1.5, convllava, internvl3) + Main (llava-onevision-7b-ov) | 1 dataset (TallyQA or PlotQA) | E4 Phase 1 sweep + Phase 2 full | 200 / full | 4 |
| §7.4.5 E6 Subspace mitigation | Main only | all 5 | calibration + sweep cells | up to 5000 wrong-base (capped by per-dataset eligible-4cond wrong-base count) | 5 |
| §5 γ-β reasoning | qwen3-vl-8b instruct vs thinking | MathVista | b/a/m/d (4) | full testmini subset | 2 |
| Appendix (legacy 7-model) | 7 models | VQAv2 only | b/a/d (3) | full | 7 |

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
- **VQAv2 number subset** (2026-05-02 decision) — dropped from the main
  matrix. Multiple-GT (10-annotator vote, "VQA accuracy" averaging) and
  open-vocabulary text answers conflict with our single-integer-GT
  pipeline; full-set evaluation is impractical at panel scale; modern
  2024–2026 numeric-VQA papers (Molmo, LLaVA-NeXT, InternVL2.5,
  Qwen2.5-VL, Cambrian) treat VQAv2 as legacy-breadth supplementary, not
  as a numeric reasoning testbed. The "natural image counting" coverage
  VQAv2 contributed is preserved by TallyQA (counting-specific, single
  integer GT). Existing 7-model VQAv2 panel preserved in appendix for
  breadth supplementary only.
- **gt ∈ [0,8] restriction on §7.4.5 mitigation headline** (2026-05-02
  decision) — removed. The Tally-only N=5000 calibration's gt-bin
  restriction made the E6 result look like a partial solution.
  Recalibrating on PlotQA + InfographicVQA (Main datasets) pooled at
  full gt range targets a stronger headline: "Subspace L31_K04_α=1.0
  generalises across 5 datasets at full gt range, no per-dataset
  tuning". If the recalibrated subspace fails on small-gt regimes
  (TallyQA), plan B = pooled multi-source (chart + info + count)
  calibration or method addendum.
- **InternVL3-8b + Qwen2.5-VL-7b in §7 mechanism panel** (2026-05-02
  decision) — moved to appendix only. Both have non-perfect-square
  visual-token grids (InternVL3 multi-tile, Qwen2.5-VL 17×23) requiring
  per-encoder bbox-to-token routing in `_compute_anchor_bbox_mass`;
  implementation correctness is hard to guarantee uniformly across
  encoders. §7.1–7.3 main panel restricted to **5-model perfect-square**
  (gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b, **llava-interleave-7b**).

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
