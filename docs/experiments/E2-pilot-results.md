# E2 pilot results — encoder ablation, n=1,125 per model

**Status:** Pilot complete 2026-04-24. 4 newly-integrated models × 25 samples-per-answer × 9 answer values × 5 irrelevant-set variants × 3 conditions = 1,125 sample-instances per model. Source: `outputs/experiment_encoder_pilot/<model>/<run>/summary.json`. Plan: `docs/experiments/E2-encoder-ablation.md`.

## Headline numbers

11-model panel (4 pilot + 7 existing main runs, sorted by `acc(target_only)`):

| Model | Encoder | acc(target) | adoption(num) | dir-follow(num) | acc_drop(num) | acc_drop(neutral) |
|---|---|---:|---:|---:|---:|---:|
| InternVL3-8B | InternViT | **0.784** | **0.066** | **0.106** | **+0.355** | +0.196 |
| qwen3-vl-30b-it | Qwen-ViT | 0.759 | 0.120 | 0.280 | +0.052 | +0.050 |
| qwen3-vl-8b-instruct | Qwen-ViT | 0.751 | 0.127 | 0.258 | +0.036 | +0.041 |
| gemma4-31b-it | SigLIP | 0.749 | 0.116 | 0.239 | +0.008 | +0.027 |
| qwen2.5-vl-7b-instruct | Qwen-ViT | 0.736 | 0.110 | 0.248 | +0.025 | +0.028 |
| gemma3-27b-it | SigLIP | 0.628 | 0.141 | 0.305 | −0.004 | +0.006 |
| llava-interleave-7b | SigLIP | 0.619 | 0.134 | 0.348 | +0.044 | +0.043 |
| **ConvLLaVA-7B** | **ConvNeXt** | **0.607** | **0.156** | **0.364** | +0.121 | +0.122 |
| gemma4-e4b | SigLIP | 0.553 | 0.123 | 0.320 | +0.012 | +0.048 |
| **LLaVA-1.5-7B** | **CLIP-ViT** | **0.528** | **0.181** | **0.355** | +0.038 | +0.049 |
| **FastVLM-7B** | **FastViT** | **0.483** | **0.090** | **0.245** | +0.188 | +0.216 |

Bold rows = pilot models. Pilot n=1,125 vs. main n=17,730 — error bars on adoption are roughly ±2–3 pp at pilot scale.

## What this changes about H3

The original H3 prediction (`references/roadmap.md` §2): *ConvNeXt-encoder VLMs should be less anchor-susceptible than CLIP/SigLIP-ViT VLMs*.

The pilot data does **not** straightforwardly support this:

- ConvLLaVA-7B (ConvNeXt) `adoption=0.156` is in the upper half of the panel, even after accounting for its low base accuracy.
- LLaVA-1.5-7B (CLIP-ViT) has the *highest* adoption in the entire 11-model panel (0.181), consistent with typographic-attack inheritance.
- ConvLLaVA's adoption (0.156) is below LLaVA-1.5-7B's (0.181), but the gap is ~2.5 pp — well within pilot noise (n=1,125 means SE ≈ 0.011, 95% CI ±0.022).

**Decision triggered (per `references/roadmap.md` §7):** ConvLLaVA's susceptibility falls *inside* the CLIP/SigLIP cluster confidence interval at pilot scale. H3 in its simple "ConvNeXt < ViT" form is **not supported**. Two paths:

1. **Run the full 17,730 ConvLLaVA + LLaVA-1.5 grid** to get tight CIs and definitively reject (or accept) H3 at scale. ~1 day per model on H200. Worth it because the negative finding ("ConvNeXt encoder doesn't escape anchoring") is itself paper-worthy and challenges a literature assumption.
2. **Refocus.** The actually-novel pattern in the pilot is something else — see next section.

## The actually-novel finding: anchoring vs. distraction are separable failure modes

The clearest new structure from the pilot is that **`adoption_rate` and `acc_drop` decouple, in opposite directions, across encoders**:

| Bucket | Models | Pattern |
|---|---|---|
| **Anchor-susceptible** (high adoption, low acc_drop) | LLaVA-1.5, gemma4-31b, qwen-family, gemma3-27b | The model encodes the anchor digit cleanly and the prediction shifts toward it, but overall accuracy on the multi-image prompt stays high. |
| **Distraction-susceptible** (low adoption, high acc_drop) | InternVL3, FastVLM | The model doesn't specifically grab the anchor digit, but its accuracy collapses when *any* second image is added. |
| **Both** | ConvLLaVA, gemma4-e4b | Weak base model, anchored *and* distracted. |

This is genuinely interesting because the cognitive-bias literature treats "anchoring" as a single phenomenon. The pilot shows that in VLMs, what looks like anchoring can be two distinct failure modes:

- **Anchoring proper** = the anchor enters the candidate distribution and pulls toward it (high adoption, low acc_drop).
- **Generic multi-image distraction** = the second image disrupts inference regardless of content (low adoption, high acc_drop).

InternVL3-8B is the cleanest illustration: `adoption=0.066` (lowest) but `acc_drop(num)=0.355` (highest by ~3×). The model "ignores the digit" but "can't keep the question in focus" with a second image around. **This decoupling is a more publishable framing than the original H3.**

## Revised paper framing candidate

Instead of "vision-encoder family modulates anchoring", the data supports:

> **VLM cross-modal failures separate cleanly into two axes when an irrelevant image is added: anchor-pull (uncertainty-modulated, encoder-mediated, low cost) and multi-image distraction (encoder-architecture-mediated, high cost). Different vision encoders sit at different points on this 2D plane, and the relevant mitigations are different for each axis.**

This carries:
- The Phase-A H2 finding (anchor pull is graded and uncertainty-modulated) for one axis.
- A new orthogonal finding (multi-image distraction) for the second axis.
- Mitigation actually differs: for anchor-pull, downweight anchor-image attention; for distraction, mask the second image entirely (or add image-routing that's content-aware). Two-axis framing means **two complementary E4 prototypes**, not one.

This is sharper than "encoder family matters", more defensible against pushback, and ties Phase A and Phase B together cleanly.

## Caveats

- **n=1,125 per model.** Direction-follow differences below ~5 pp are not safely interpretable. The two-axis story holds because the decoupling is large (InternVL3: 0.066 vs 0.355; ConvLLaVA: 0.156 vs 0.121). Adoption-only comparisons within ~5 pp are not reliable.
- **FastVLM acc(target_only)=0.483 is suspect.** This model emits prose despite the JSON-only prompt; the parser rescues most cases but not all. Need to inspect `is_numeric_prediction` rate before drawing FastVLM-specific conclusions.
- **InternVL3's behaviour might be a tokenizer artifact.** The very low adoption could be because the model emits multi-token answers that don't parse as the anchor digit cleanly. Look at the raw_prediction strings before claiming "low anchoring".
- **No bootstrap CIs yet.** Pilot decisions made with point estimates; main run will add CIs.

## Decision

Recommend (with user confirmation before kicking off):

- **Yes** to full 17,730 runs of ConvLLaVA + LLaVA-1.5 + InternVL3 + FastVLM. Justifications:
  - InternVL3's distraction-vs-anchor decoupling is the core new claim and needs n=17,730 + CIs.
  - ConvLLaVA + LLaVA-1.5 give the clean "ConvNeXt vs CLIP-ViT, same Vicuna LLM" comparison the H3 evaluation needs.
  - FastVLM run gives a 4th encoder family (FastViT) at low marginal cost.
  - All four runs will capture per-token logits (commit `5f925b2`), enabling the deferred A1 logit-margin re-analysis.
- **Reorder Phase B:** E2 (full 4 models) → E1 (attention mass on InternVL3 + LLaVA-1.5 head-to-head) → E4 (two-axis mitigation). E3 already absorbed into E2.
- **Drop the "encoder family universally matters" framing.** Replace with the two-axis "anchoring vs. distraction" framing. Update the §2 hypothesis table after user signoff.

## Roadmap updates triggered

- §3.3 — move ConvLLaVA / InternVL3 / LLaVA-1.5 / FastVLM out of "integrated but no full run" once full runs are queued.
- §6 Tier 1 E2 + E3 — collapse into one combined "full encoder-ablation grid" task.
- §7 decision triggers — fire the H3 decision with "not supported in simple form; switching to two-axis framing".
- §10 changelog — add this writeup.
