# §1. Introduction

## Abstract (draft)

We show that vision-language models (VLMs) exhibit cross-modal numerical
anchoring: presenting an irrelevant image containing a single rendered
digit alongside the target image biases the model's numeric answer
toward that digit. The bias is **uncertainty-modulated graded pull**,
not categorical capture: across seven open-weight VLMs on VQAv2, only
2-7 % of predictions match the anchor digit literally, but the model's
prediction shifts *toward* the anchor side of its own anchor-free
baseline at a rate that monotonically tracks base-prediction
uncertainty (Q4 minus Q1 entropy quartile gap = +15.2 pp on
direction-follow). The effect concentrates on a digit-pixel cue:
inpainting the digit out of the anchor image reduces the effect to
the level of a generic 2-image distractor, while the same scene with
the digit visible drives a ~6 pp wrong-base-S1 adoption gap on VQAv2
and up to +17.9 pp on MathVista. Mechanistically, anchor attention
concentrates at four encoder-family-specific peak layers, but
single-layer ablation is causally null on 6/6 models — the anchor
signal is multi-layer-redundant. Upper-half attention re-weighting on
the mid-stack-cluster (LLaVA-1.5 / ConvLLaVA / InternVL3) reduces
direction-follow by 5.8-17.7 % relative while exact-match *rises* by
0.49-1.30 pp ("free-lunch" mitigation). A sharper finding: a
reasoning-mode VLM (Qwen3-VL-8B-Thinking) **amplifies** rather than
suppresses anchor pull on MathVista (adopt ×1.6, direction-follow
×2.9 vs. its non-reasoning Instruct counterpart), aligning with
recent text-only LRM-judging results.

## 1.1 Motivation

Multi-image prompts are increasingly common: users attach screenshots,
albums, or document scans to a single VQA query. When one of those
images is irrelevant to the question — pulled in by accident, or by
an attacker — does it affect the answer? The cognitive-science
literature has a strong prior: humans anchor on irrelevant numerical
cues even when explicitly told they are irrelevant
[Tversky-Kahneman 1974, Mussweiler-Strack 1997, Jacowitz-Kahneman 1995].
The LLM literature has confirmed this for *text* anchors
[Jones-Steinhardt 2022, Echterhoff et al. 2024, Lou-Sun 2024,
Huang et al. 2025]. We ask: does the same anchoring effect transfer to
the *image* modality, when the cue is a stand-alone rendered-number
image carrying no semantic label?

## 1.2 Setup novelty

No prior work has delivered a stand-alone rendered-number image as a
cross-modal anchor for open-numerical VQA. Closest neighbors differ
on a load-bearing axis: VLMBias [Nguyen et al. 2025] uses
memorized-subject identifiers as cues to probe counterfactual
counting; multi-image typographic attacks [Wang-Zhao-Larson 2025]
overlay class-label text on the target image to flip classification;
FigStep [Gong et al. 2025] renders prompt text as an image to
jailbreak; Tinted Frames [2026] varies question-form framing in VLMs.
None measure regression-style numeric shift toward an arbitrary anchor.
The full prior-art differentiation matrix is in §2.

## 1.3 Headline claim

Cross-modal numerical anchoring in VLMs is **uncertainty-modulated
graded pull, not categorical capture**, and concentrates on a
**digit-pixel cue** inside the anchor image. Three pillars carry
this claim through the paper:

1. **Graded vs. categorical.** Phase-A analysis on VQAv2 shows
   wrong-base direction-follow exceeds correct-base by +6.9 to
   +19.6 pp on every one of seven models. Paired adoption is 2-7 % —
   the model rarely *outputs* the anchor literally; the mass of the
   effect is in graded movement *toward* the anchor (§3-§5, §A1).
2. **Digit-pixel causality.** Replacing the digit pixel region with a
   content-preserving inpaint (Telea, OpenCV) reduces the anchor
   effect to a generic 2-image distractor's level on the masked arm,
   while the digit-visible arm drives substantial paired adoption
   (§5, E5c).
3. **Confidence-modulated.** Direction-follow is monotonic with
   answer-token entropy across 23 of 35 (model × dataset × stratum)
   cells; the binary wrong/correct split is a coarse projection of
   this continuous structure (§6, L1).

## 1.4 Mechanism + mitigation

Anchor attention concentrates at one peak layer per encoder family
(SigLIP-Gemma L5/42, mid-stack cluster L14-16, Qwen-ViT L22/28,
FastVLM L22). But single-layer attention-mask ablation is causally
null on 6/6 models: the anchor signal is multi-layer-redundant.
Upper-half attention re-weighting on the mid-stack cluster reduces
direction-follow by 5.8-17.7 % relative with exact-match *rising*
0.49-1.30 pp and target-only accuracy invariant — a "free-lunch"
mitigation that generalises across CLIP-ViT / ConvNeXt / InternViT
encoders at the same depth (§7).

## 1.5 Reasoning amplifies anchoring

A pair-comparison on MathVista — Qwen3-VL-8B-Instruct vs. the
separately-trained Qwen3-VL-8B-Thinking checkpoint — shows the
Thinking variant *amplifies* anchor pull on every metric (adopt
×1.6, direction-follow ×2.9), confirming on a VLM what VLMBias
[Nguyen et al. 2025] and Wang et al. ["Judging Bias in Large
Reasoning Models", 2025] established for text-only LRMs. Reasoning
trace does not buy accuracy on this panel; it simply lowers anchor
robustness (§7-§8).

## 1.6 Contributions

1. The **first cross-modal numeric anchoring evaluation** for VLMs:
   a 4-condition setup (target / anchor / mask / neutral) with a
   FLUX-rendered digit anchor inventory and OpenCV-inpainted
   digit-mask counterparts.
2. **Canonical M2 metrics** with the C-form direction-follow
   numerator `(pa − pb) · (anchor − pb) > 0 AND pa ≠ pb` — a
   gt-free, baseline-relative measure of anchor pull that is robust
   to per-question stimulus draw.
3. **Cross-dataset evidence** on VQAv2 (number subset, 17,730
   samples), TallyQA (test number-type), ChartQA (integer subset),
   and MathVista (testmini integer subset), spanning seven VLMs on
   the main panel + three on cross-dataset E5e + two on γ-β
   reasoning-mode.
4. **Mechanistic + mitigation pass**: per-encoder-family attention
   localisation (E1/E1b, n=200), causal ablation panel (E1d, 6
   models × 6 ablation modes), and upper-half attention re-weighting
   mitigation validated at full scale on 3/3 mid-stack-cluster
   models (E4 Phase 2, 88,650 records each).
5. **A reasoning-amplifies-anchoring result on a VLM** (γ-β,
   MathVista), aligning the VLM picture with the recent
   reasoning-bias-amplification literature on text-only LRMs.

## 1.7 Reproducibility

All code, configs, anchor inventories (FLUX seeds + OpenCV inpaint
parameters), per-sample predictions, and aggregate CSVs are released
under a permissive license. Models are open weights from
Hugging Face. The C-form metric refactor is documented in
`docs/insights/M2-metric-definition-evidence.md` and
`docs/insights/C-form-migration-report.md`.
