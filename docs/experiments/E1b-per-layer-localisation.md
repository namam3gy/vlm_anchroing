# E1b — per-layer localisation of the anchor-attention gap

**Status:** Follow-up to `docs/experiments/E1-preliminary-results.md`. Now covers the full 6-model panel (4 HFAttention + 2 inputs_embeds-path extensions) at n=200 each. Analysis script: `scripts/analyze_attention_per_layer.py`. Raw per-layer table: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. Figures: `outputs/attention_analysis/_per_layer/fig_delta_by_layer_{answer,step0}.png`.

## TL;DR — three per-layer findings, now on 6 models

1. **The anchor-attention gap is layer-localised, and the peak layer differs by encoder family.** Gemma-SigLIP peaks at **layer 5 / 42 (~12 % depth)** with delta **+0.050** (answer step); the mid-stack cluster — CLIP-ViT (LLaVA-1.5), InternViT (InternVL3), ConvNeXt (ConvLLaVA) — all peak at layers 14–16 (~52 % depth) with δ ≈ +0.019–0.022; Qwen-ViT peaks at layer 22 / 28 (~82 %) with +0.015; FastViT (FastVLM) peaks at layer 22 / 28 with **+0.047** — the second-largest magnitude in the panel, at Qwen's depth. The E1 layer-averaged number (≈ 0.005) was hiding up to a ~10× single-layer concentration.
2. **Gemma-SigLIP's early-layer peak is flanked by layers with *negative* delta, but those negatives are mostly anchor/target trade-off, not anti-anchor.** Layers 0–4, 6–10, 12, 17 all have delta < 0 at the answer step with CI excluding zero. The per-layer budget decomposition (see below) shows most of these are anchor↔target mass swaps between the two images (text untouched), not active anti-anchor layers. Only layer 5 genuinely pulls mass from text into the anchor image (δ_text = −0.038); only layer 17 genuinely reallocates from anchor back to text. The TL;DR "knife-edge" imagery holds for the *spike* at layer 5 but not for the flanking layers.
3. **The A7 susceptibility gap is concentrated at the peak layer, and the magnitudes are strongly hierarchical.** At each model's answer-step peak, top-decile-susceptible vs bottom-decile-resistant delta is: **FastVLM +0.086** (huge, n=28 top-decile so wide CI) → Qwen +0.025 → ConvLLaVA +0.013 → LLaVA-1.5 +0.010 → InternVL3 +0.006 → Gemma +0.001 (near-zero). FastVLM is the single cleanest "susceptibility gates the mechanism" case in the panel. Gemma remains the lone inversion, consistent with its unconditional typographic-attack profile.

## Setup

Same extraction pipeline as E1, plus two extensions:

- **HFAttentionRunner path** (gemma4-e4b, qwen2.5-vl-7b-instruct, llava-1.5-7b, internvl3-8b): image-token id scanned in `input_ids`.
- **ConvLLaVA inputs_embeds path**: inputs built by concatenating text-chunk embeddings with ConvNeXt vision features; image spans tracked at splice time in `scripts/extract_attention_mass.py`.
- **FastVLM -200-expand path**: `-200` markers inline in `input_ids` are replaced by 256 FastViT patch embeddings at model-forward time; spans back-computed from step-0 attention k_len.

For each triplet (sample_instance × {base, number, neutral}) and each layer `l`, compute:

```
delta_l = image_anchor_mass[l](number) − image_anchor_mass[l](neutral)
```

evaluated at two generation-step labels: **answer** (first step whose decoded token contains a digit, per `per_step_tokens`) and **step 0** (first generated token — usually the opening brace of the JSON response). Bootstrap 2,000 iter, 95 % CI.

Answer-step `n` drops below 200 for two models: InternVL3 drops to n=135 and FastVLM drops to n=75 because some generations don't emit a digit token. FastVLM used `max_new_tokens=24` (the other five used 8) because under the JSON prompt it typically emits prose for ~10–12 tokens before (if at all) producing the answer digit — this is the roadmap §9 caveat.

## Per-model peak summary (answer step, overall stratum)

| Model | peak layer | depth % | peak delta | 95 % CI | # sig-positive layers | n |
|---|---:|---:|---:|---|---:|---:|
| gemma4-e4b | **5 / 42** | 12 % | **+0.05007** | [+0.04772, +0.05229] | 19 / 42 | 200 |
| internvl3-8b | 14 / 28 | 52 % | +0.01931 | [+0.01348, +0.02535] | 24 / 28 | 135 |
| llava-1.5-7b | 16 / 32 | 52 % | +0.01883 | [+0.01441, +0.02310] | 26 / 32 | 200 |
| **convllava-7b** | **16 / 32** | **52 %** | **+0.02238** | [+0.01713, +0.02812] | — | 200 |
| qwen2.5-vl-7b | 22 / 28 | 82 % | +0.01527 | [+0.00842, +0.02307] | 22 / 28 | 200 |
| **fastvlm-7b** | **22 / 28** | **82 %** | **+0.04665** | [+0.02528, +0.07157] | — | 75 |

Four tiers now visible (not three): SigLIP-Gemma very early + large; the mid-stack cluster (3 encoders, tight peak agreement at L14–16) + moderate; Qwen-ViT late + moderate; FastVLM late + large (approaching Gemma's magnitude).

The ConvLLaVA row is the key piece of new evidence for the original H3 hypothesis: ConvNeXt encoder, yet peak layer exactly matches CLIP-ViT-based LLaVA-1.5, magnitude within 20 %, mechanism identical. **H3 in "ConvNeXt less susceptible than ViT" form is falsified at the per-layer level, not just the adoption level.**

## Per-model peak summary (step 0, overall stratum)

| Model | peak layer | depth % | peak delta | 95 % CI |
|---|---:|---:|---:|---|
| gemma4-e4b | 6 / 42 | 15 % | +0.04505 | [+0.03817, +0.05177] |
| internvl3-8b | 3 / 28 | 11 % | +0.00760 | [+0.00665, +0.00858] |
| qwen2.5-vl-7b | 19 / 28 | 70 % | +0.00975 | [+0.00726, +0.01225] |
| llava-1.5-7b | 26 / 32 | 84 % | +0.01333 | [+0.01137, +0.01551] |
| **fastvlm-7b** | **27 / 28** | **96 %** | +0.01966 | [+0.01678, +0.02251] |
| **convllava-7b** | **31 / 32** | **100 %** | +0.02299 | [+0.01788, +0.02816] |

New pattern in the two inputs_embeds-path models: their step-0 peaks sit at the very last layer (ConvLLaVA L31/32 = 100 %, FastVLM L27/28 = 96 %) — different from their own answer-step peaks (L16 and L22 respectively) and different from every other model in the panel. Interpretation is speculative without a causal test; one possibility is that inputs_embeds-path models process vision features externally and the LLM's final-layer integration step plays a different role than in models where the vision token is in the raw `input_ids` sequence.

Gemma's step-0 peak is basically co-located with its answer-step peak (layers 5 vs. 6), and the signal is broadcast across 39/42 layers (virtually all significant) — consistent with the E1 claim that Gemma does its anchor-encoding at prompt integration and then *re-uses* the same encoding at answer time.

InternVL3 switches peak layer between the two steps (step-0 peak at layer 3, answer peak at layer 14) — its prompt-read happens early but the answer-decision layer is mid-stack. This pattern makes InternVL3 look like a two-stage reader: early registration + mid-stack decision.

## Four encoder-family archetypes (6-model panel)

Bringing together E1 Test 1 (overall delta), E1 Test 3 (A7 stratum), and this layer localisation:

| Family | Peak layer depth | Peak delta | A7 gap at peak | Budget source at peak | Step-0 vs answer | Interpretation |
|---|---|---|---|---|---|---|
| SigLIP (Gemma) | **very early** (12 %) | large (+0.050) | ~0 (+0.001) | **text** (−0.038) | nearly co-located | Early unconditional registration of in-image text. Anchor pulls from prompt text, consistent with SigLIP's typographic-attack inheritance. Susceptibility is not a conditioner. |
| CLIP-ViT (LLaVA-1.5) | mid (52 %) | moderate (+0.019) | moderate (+0.010) | text (−0.029), target also *gains* (+0.007) | answer peak mid / step-0 peak late | Both images pool attention from text. Anchor doesn't compete with target; they co-aggregate as "visual evidence". |
| InternViT (InternVL3) | mid (52 %) | moderate (+0.019) | small (+0.006) | text (−0.014) with small target (−0.004) | step-0 peak early (L3) / answer peak mid | Text-dominant like LLaVA-1.5 but with slight target displacement. Two-stage: early registration, mid-stack decision. |
| **ConvNeXt (ConvLLaVA)** | **mid (52 %)** | **moderate (+0.022)** | **moderate (+0.013)** | **text (−0.019)** with small target (−0.003) | step-0 peak very late (L31/32 = 100 %) / answer peak mid | Almost indistinguishable from LLaVA-1.5 at the answer step. Step-0 peak uniquely sits at the final LLM layer — interpretation open. |
| Qwen-ViT (Qwen2.5-VL) | **late** (82 %) | moderate (+0.015) | **large** (+0.025, CI-separated) | **target** (−0.010) | co-located late | Late-stack conditional anchor-vs-target competition: anchor displaces the target image specifically at the answer-decision layer. |
| **FastViT (FastVLM)** | **late (82 %)** | **large (+0.047)** | **huge (+0.086)** | **text (−0.034)** with target (−0.014) | step-0 peak very late (L27/28 = 96 %) / answer peak late | Gemma-magnitude late-stack text-stealing with the panel's strongest susceptibility gating. Subject to n=75 caveat. |

Three reads of the 6-model panel:

1. **"ViT vs. ConvNeXt" (original H3 axis) is definitively less informative than "where in the LLM stack the anchor gets read in".** Three encoders (CLIP-ViT, InternViT, ConvNeXt) converge at layers 14–16 with the same mechanism and similar magnitudes. Within this cluster, architecture does not predict attention.
2. **SigLIP and FastViT are architecturally different outliers with overlapping behavioural fingerprints.** Both show large magnitudes (+0.050 and +0.047 respectively) and text-stealing budget sources; they differ in *depth* (Gemma early, FastVLM late) and in *susceptibility gating* (Gemma has none; FastVLM has the most in the panel). The two published VLM failure modes that the literature flags — typographic attack (in-image text read-off) and "which image answers the question" budget confusion — may be co-firing in FastVLM in a way that isolates cleanly in neither Gemma nor Qwen.
3. **The mid-stack cluster is now the cleanest E4 intervention target.** Three different encoders produce the same profile; an architecture-blind intervention at mid-stack layers 14–16 could plausibly work across all three. For Gemma, FastVLM, Qwen, each needs its own design.

## Peak-layer budget decomposition: where does the anchor steal attention from?

Attention mass at each layer sums to 1, so a positive `delta_anchor` at a given layer must be offset by negative deltas somewhere else in the same layer. That offset source is the mechanistic signal — does the anchor compete with the *target image*, with the *prompt text*, or with already-generated tokens?

At each model's answer-step peak layer (number − neutral, answer step, averaged across triplets, raw `sum_check ≈ 0` confirms normalisation):

| Model | peak layer | δ anchor | δ target | δ text | δ generated |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | **+0.05007** | −0.00961 | **−0.03804** | −0.00242 |
| qwen2.5-vl-7b | 22 | +0.01527 | **−0.00972** | −0.00509 | −0.00046 |
| llava-1.5-7b | 16 | +0.01883 | +0.00702 | **−0.02940** | +0.00355 |
| internvl3-8b | 14 | +0.01931 | −0.00384 | −0.01427 | −0.00120 |
| convllava-7b | 16 | +0.02238 | −0.00254 | **−0.01920** | −0.00063 |
| fastvlm-7b | 22 | +0.04665 | −0.01393 | **−0.03427** | +0.00156 |

Two distinct mechanisms still fall out, and now five of six models are in the text-stealing bucket:

- **Text-stealing (Gemma, LLaVA-1.5, InternVL3, ConvLLaVA, FastVLM).** The anchor image gains mass primarily at the expense of the prompt text. In LLaVA-1.5 the target image *also* gains (+0.007) — both images pool attention and together pull from text. ConvLLaVA and InternVL3 show mild target-loss (≈ −0.003); FastVLM pulls moderately from both text (−0.034) and target (−0.014), with text dominating.
- **Target-stealing (Qwen-only).** At Qwen's layer 22, the anchor takes mass mainly from the target image (−0.010) rather than from text (−0.005). By this late in the stack the answer-generation circuit has a fixed budget for "look at the relevant image", and the anchor image displaces the target image within that budget.

This refines the four archetype table above by confirming the budget column:

- **Gemma (SigLIP):** early + text-stealing. Anchor digit is read off the prompt during early prompt-integration layers; the pull comes from text mass, not from target-image mass.
- **Mid-stack cluster (LLaVA-1.5, InternVL3, ConvLLaVA):** mid + text-stealing (mostly). Two of the three also show either small target gain or small target loss — the two images behave cooperatively, not competitively. This explains why these three are the "middle" A7 magnitudes and the most continuous profiles across layers.
- **Qwen (Qwen-ViT):** late + target-stealing. Anchor displaces the target image specifically at the answer-decision layer.
- **FastVLM (FastViT):** late + text-stealing + large. Unique in the panel. Combines Gemma's magnitude and text-source with Qwen's depth.

## Gemma layer 5 revisited — not an anti-anchor circuit, but a localised text → anchor pull

The flanking negative-delta layers (0–4, 6–10, 12, 17) are **not** an "active anti-anchor circuit". A per-layer budget check explains them:

- Layers 0–10 (except 5): δ_anchor and δ_target are near-equal and opposite (e.g. layer 2: δ_target=+0.0136, δ_anchor=−0.0135). These are anchor/target trade-off layers — mechanical, not directional. They swap mass between the two images; text is unaffected (|δ_text| < 0.004).
- Layer 5: δ_anchor=+0.050, δ_text=−0.038, δ_target=−0.010. Text loses the most. This is the one layer where the anchor genuinely pulls from the text stream.
- Layer 17: δ_anchor=−0.020, δ_text=+0.016 (CI excludes 0), δ_target ≈ 0. Anchor loses, but to *text*, not to target. This is closest to an anti-anchor layer, but it's a text-rebalance, not a target-rebalance.

So the "knife-edge spike flanked by negative layers" language in the TL;DR needs one qualification: the spike at layer 5 is real and knife-edge, but most of the flanking negative-delta layers are mechanical anchor/target swaps, not active suppression. Only layer 17 meaningfully redirects mass away from the anchor, and it redirects to text, not to target.

Alternative reading we considered: pure *budget redistribution* (no directional pull, just normalisation artefact of the spike). The δ_text=−0.038 at layer 5 rules this out — the anchor specifically pulls from prompt text, not from a randomly-chosen other region. This is consistent with SigLIP's typographic-attack inheritance: the digit is read off the image as if it were text, and text-mass is the natural source.

The E1 writeup proposed an input-side intervention for Gemma (zero the anchor's visual-projection features, or patch the KV cache at the input layer). The per-layer + budget data now give a concrete target: **a pre-layer-5 projection/KV patch** that denies the text-stealing pull at layer 5. An attention re-weighting at the answer step would not work because the damage happens during prompt integration and is cached.

## A7 susceptibility: per-layer gap

Top-decile-susceptible minus bottom-decile-resistant delta at each model's answer-step peak layer (same layer across both strata):

| Model | peak layer | top delta [CI] | bottom delta [CI] | gap |
|---|---:|---|---|---:|
| **fastvlm-7b** | 22 | **+0.10054 [+0.04728, +0.16007]** | +0.01454 [+0.00650, +0.02237] | **+0.08600** |
| qwen2.5-vl-7b | 22 | +0.02794 [+0.01532, +0.04271] | +0.00259 [−0.00204, +0.00602] | +0.02535 |
| convllava-7b | 16 | +0.02868 [+0.02043, +0.03836] | +0.01607 [+0.01053, +0.02148] | +0.01262 |
| llava-1.5-7b | 16 | +0.02369 [+0.01600, +0.03177] | +0.01398 [+0.00999, +0.01782] | +0.00970 |
| internvl3-8b | 14 | +0.02302 [+0.01273, +0.03766] | +0.01726 [+0.01064, +0.02362] | +0.00576 |
| gemma4-e4b | 5 | +0.05078 [+0.04758, +0.05417] | +0.04936 [+0.04628, +0.05260] | +0.00142 |

Two "cleanest susceptibility gates" in the panel:

- **FastVLM:** top-decile +0.10, bottom-decile +0.015 — a 6.5× ratio with only n=28 top-decile and n=47 bottom-decile (after the 62 % digit-coverage filter). The gap (+0.086) is 3× bigger than any other model's; the top-decile CI is wide but doesn't touch the bottom-decile CI. Caveat: with n=28, the +0.10 point estimate may be optimistic — a larger run would tighten this. Candidate for the highest-leverage E4 intervention site.
- **Qwen-ViT:** the bottom-decile CI *includes zero* at the model's own peak layer — for resistant items, Qwen doesn't even develop a late-stack attention gap. For susceptible items the same layer fires at +0.028. A single layer carries almost all of the A7 signal, and only for susceptible items. The tightest "susceptibility-gated answer-step circuit" in the panel (outside FastVLM's wider CIs).

The ConvLLaVA A7 gap (+0.013) lands between LLaVA-1.5 (+0.010) and Qwen (+0.025), consistent with its mid-stack-cluster membership and confirming that ConvNeXt-based models do develop some susceptibility-gated structure — just less than ViT-based models.

## Implications for E4 (mitigation) — candidate sites, to be tested

These are hypotheses generated from the observational per-layer + budget data; nothing has been ablated or re-weighted yet. The E1 writeup's conclusion stands ("a single universal mitigation will not serve"), and the 6-model per-layer + budget data narrows the *candidate* intervention sites for E4 across four archetypes:

- **SigLIP-Gemma:** input-side candidate. Target layer 5 or earlier, with the specific aim of denying the text→anchor pull that characterises that layer. Attention re-weighting at generation time is unlikely to work (peak is in the KV cache already). Options to try: project-feature masking on the second image, or KV-cache patching at the encoder / early LLM layer boundary.
- **Mid-stack cluster (LLaVA-1.5, InternVL3, ConvLLaVA):** mid-stack attention re-weighting candidate at layer ~14–16 on the answer-step. Because the budget source is mostly text, a down-weight on the anchor will return mass to text, not to target. **Highest leverage in the panel** because three encoders share the profile — a single intervention tuned once could plausibly generalise across CLIP-ViT, InternViT, and ConvNeXt models.
- **Qwen-ViT:** late-stack attention re-weighting candidate at layer 22 ± 2 on the answer-step only, gated by a per-item susceptibility signal. Because at this layer the budget is anchor-vs-target (not anchor-vs-text), a down-weight on the anchor should push mass back to the target image rather than to text — the desired outcome.
- **FastVLM (FastViT):** late-stack re-weighting candidate at layer 22 ± 2, gated by susceptibility. The A7 gap is so large here (top-decile δ ≈ +0.10 vs bottom-decile ≈ +0.015) that an A7-gated rescale could buy an outsized direction-follow reduction — but with only n=28 top-decile triplets surviving the digit filter, we should verify the gap on a larger run (or with `max_new_tokens ≥ 32`) before committing design effort here.

The panel is now consistent with a paper-level hypothesis: **the location, magnitude, and competing region of cross-modal bias in the LLM stack are encoder-family-specific, and an architecture-aware mitigation design is required.** Whether this hypothesis survives the causal E4 intervention is TBD — this is observational mechanistic evidence, not a tested fix.

## Caveats

- `n=200` per model per step 0. At the answer step InternVL3 drops to n=135 and FastVLM drops to n=75 (both due to generations without a digit token; FastVLM hit the roadmap §9 "prose before digit" issue). FastVLM's wide answer-step peak CI ([+0.025, +0.072]) is the main consequence — a larger run or `max_new_tokens ≥ 32` would tighten it.
- "Peak layer" is picked as `argmax(delta_mean)`. With tight CIs at the peak this is stable, but the precise choice of 5 vs 6 for Gemma or 16 vs 17 for LLaVA-1.5 is within CI overlap.
- FastVLM used `max_new_tokens=24`, the other five used 8. This affects which step indices are considered "answer step" but not the delta at whichever step a digit happens to land on (the comparison is within-triplet).
- ConvLLaVA and FastVLM results are from the `inputs_embeds`-path extension added in this round. The extraction-script extension was smoke-tested, span computation was cross-checked against expected expanded seq_len (`sum_check ≈ 0` on the normalised masses and the FastVLM span-reconstruction assertion both pass on every record). Still: this is a first integration — treat quantitative numbers as preliminary until an independent cross-check.
- Layer-wise mass is averaged over heads. Head-level sparsity (a single head at layer 5 in Gemma carrying the signal) would be invisible here and would strengthen the circuit claim. Out of scope for E1b.

## Roadmap update

- §3 status: E1b layer analysis done across 6 models (this file). Cross-family pattern now 4-way: SigLIP early + large / mid-stack cluster of 3 encoders / Qwen-ViT late + target-stealing / FastViT late + large + strongest A7 gating.
- §6 Tier 1 E1: **two open questions closed** — "per-layer localisation" and "ConvNeXt/FastViT extension". Remaining: head-level sparsity, causal test (ablate anchor tokens at each family's peak layer and measure `direction_follow`).
- §10 changelog entries dated 2026-04-24.
