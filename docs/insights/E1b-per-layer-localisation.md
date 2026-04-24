# E1b — per-layer localisation of the anchor-attention gap

**Status:** Follow-up to `docs/experiments/E1-preliminary-results.md`. Re-analyses the same 4 models × n=200 attention runs but disaggregates the layer-averaged delta into per-layer traces. Analysis script: `scripts/analyze_attention_per_layer.py`. Raw per-layer table: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. Figures: `outputs/attention_analysis/_per_layer/fig_delta_by_layer_{answer,step0}.png`.

## TL;DR — three per-layer findings

1. **The anchor-attention gap is layer-localised, and the peak layer differs by encoder family.** Gemma-SigLIP peaks at **layer 5 / 42 (~12 % depth)** with delta **+0.050** (answer step); Qwen-ViT peaks at **layer 22 / 28 (~82 % depth)** with +0.015; CLIP-ViT and InternViT both peak **mid-stack (~52 % depth)** with ~+0.019. The E1 layer-averaged number was hiding the fact that a single layer is doing ~3× more work than a uniform-distribution reading implied.
2. **Gemma-SigLIP's early-layer peak is flanked by layers with *negative* delta, but those negatives are mostly anchor/target trade-off, not anti-anchor.** Layers 0–4, 6–10, 12, 17 all have delta < 0 at the answer step with CI excluding zero. The per-layer budget decomposition (see below) shows most of these are anchor↔target mass swaps between the two images (text untouched), not active anti-anchor layers. Only layer 5 genuinely pulls mass from text into the anchor image (δ_text = −0.038); only layer 17 genuinely reallocates from anchor back to text. The TL;DR "knife-edge" imagery holds for the *spike* at layer 5 but not for the flanking layers.
3. **The A7 susceptibility gap is concentrated at the peak layer, and its magnitude differs by family.** At each model's answer-step peak, top-decile-susceptible vs bottom-decile-resistant delta is: Qwen +0.02535 (CI separation clean) → LLaVA-1.5 +0.00970 → InternVL3 +0.00576 → Gemma +0.00142 (near-zero). The E1 Test-3 A7 pattern (3/4 holds, Gemma inverts) sharpens into a per-family *magnitude* hierarchy once we localise by layer.

## Setup

Same `per_step_attention.jsonl` files as E1 — 4 models × 600 attention records each (3 conditions × 200 susceptibility-stratified questions). For each triplet (sample_instance × {base, number, neutral}) and each layer `l`, compute:

```
delta_l = image_anchor_mass[l](number) − image_anchor_mass[l](neutral)
```

evaluated at two generation-step labels: **answer** (first step whose decoded token contains a digit, per `per_step_tokens`) and **step 0** (opening brace / first generated token). Bootstrap 2,000 iter, 95 % CI, across the ~200 valid triplets per model.

InternVL3 at the answer step drops to n=135 because ~65 triplets had no digit token in either the number or neutral generated sequence (the model occasionally degenerates into longer prose under the JSON prompt).

## Per-model peak summary (answer step, overall stratum)

| Model | peak layer | depth % | peak delta | 95 % CI | # sig-positive layers |
|---|---:|---:|---:|---|---:|
| gemma4-e4b | **5 / 42** | 12 % | **+0.05007** | [+0.04772, +0.05229] | 19 / 42 |
| llava-1.5-7b | 16 / 32 | 52 % | +0.01883 | [+0.01441, +0.02310] | 26 / 32 |
| internvl3-8b | 14 / 28 | 52 % | +0.01931 | [+0.01348, +0.02535] | 24 / 28 |
| qwen2.5-vl-7b | 22 / 28 | 82 % | +0.01527 | [+0.00842, +0.02307] | 22 / 28 |

Gemma-SigLIP's peak delta is ~2.5–3× larger in magnitude and sits at an order-of-magnitude shallower layer fraction than the three ViT-family models. The other three cluster at mid-to-late stack.

## Per-model peak summary (step 0, overall stratum)

| Model | peak layer | depth % | peak delta | 95 % CI | # sig-positive layers |
|---|---:|---:|---:|---|---:|
| gemma4-e4b | 6 / 42 | 15 % | +0.04505 | [+0.03817, +0.05177] | 39 / 42 |
| qwen2.5-vl-7b | 19 / 28 | 70 % | +0.00975 | [+0.00726, +0.01225] | 22 / 28 |
| llava-1.5-7b | 26 / 32 | 84 % | +0.01333 | [+0.01137, +0.01551] | 26 / 32 |
| internvl3-8b | 3 / 28 | 11 % | +0.00760 | [+0.00665, +0.00858] | 26 / 28 |

Gemma's step-0 peak is basically co-located with its answer-step peak (layers 5 vs. 6), and the signal is broadcast across 39/42 layers (virtually all significant) — consistent with the E1 claim that Gemma does its anchor-encoding at prompt integration and then *re-uses* the same encoding at answer time.

InternVL3 switches peak layer between the two steps (step-0 peak at layer 3, answer peak at layer 14) — its prompt-read happens early but the answer-decision layer is mid-stack. This pattern makes InternVL3 look like a two-stage reader: early registration + mid-stack decision.

## The three encoder-family archetypes

Bringing together E1 Test 1 (overall delta), E1 Test 3 (A7 stratum), and this layer localisation:

| Family | Peak layer depth | Peak delta | A7 gap at peak | Budget source at peak | Step-0 vs answer | Interpretation |
|---|---|---|---|---|---|---|
| SigLIP (Gemma) | **very early** (12 %) | large (+0.050) | ~0 (+0.001) | **text** (−0.038) | nearly co-located | Early unconditional registration of in-image text. Anchor pulls from prompt text, consistent with SigLIP's typographic-attack inheritance. Susceptibility is not a conditioner. |
| CLIP-ViT (LLaVA-1.5) | mid (52 %) | moderate (+0.019) | moderate (+0.010) | text (−0.029), target also *gains* (+0.007) | answer peak mid / step-0 peak late | Both images pool attention from text. Anchor doesn't compete with target; they co-aggregate as "visual evidence". |
| InternViT (InternVL3) | mid (52 %) | moderate (+0.019) | small-moderate (+0.006) | text (−0.014) with small target (−0.004) | step-0 peak early / answer peak mid | Text-dominant like LLaVA but with slight target displacement. Two-stage: early registration, mid-stack decision. |
| Qwen-ViT (Qwen2.5-VL) | **late** (82 %) | moderate (+0.015) | **large** (+0.025, CI-separated) | **target** (−0.010) | co-located late | Late-stack conditional anchor-vs-target competition: anchor displaces the target image specifically at the answer-decision layer. |

Key takeaway: "ViT vs. ConvNeXt" (the original H3 axis) is less informative than "where in the LLM stack the anchor gets read in". SigLIP is the outlier both on depth (shallow) and on susceptibility-gating (none). Qwen is the other extreme (deep, heavily susceptibility-gated). The CLIP-ViT and InternViT models sit in the middle on both axes.

## Peak-layer budget decomposition: where does the anchor steal attention from?

Attention mass at each layer sums to 1, so a positive `delta_anchor` at a given layer must be offset by negative deltas somewhere else in the same layer. That offset source is the mechanistic signal — does the anchor compete with the *target image*, with the *prompt text*, or with already-generated tokens?

At each model's answer-step peak layer (number − neutral, answer step, averaged across triplets, raw `sum_check ≈ 0` confirms normalisation):

| Model | peak layer | δ anchor | δ target | δ text | δ generated |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | **+0.05007** | −0.00961 | **−0.03804** | −0.00242 |
| qwen2.5-vl-7b | 22 | +0.01527 | **−0.00972** | −0.00509 | −0.00046 |
| llava-1.5-7b | 16 | +0.01883 | +0.00702 | **−0.02940** | +0.00355 |
| internvl3-8b | 14 | +0.01931 | −0.00384 | −0.01427 | −0.00120 |

Two distinct mechanisms fall out:

- **Text-stealing** (Gemma, LLaVA-1.5, InternVL3). The anchor image gains mass primarily at the expense of the prompt text. In LLaVA-1.5 the target image *also* gains (+0.007) — both images together pull from text. This is consistent with the anchor registering as "another piece of visual evidence" rather than competing with the target for a shared image slot.
- **Target-stealing** (Qwen-only). At Qwen's layer 22, the anchor takes mass mainly from the target image (−0.010), not from text (−0.005). By this late in the stack the answer-generation circuit has a fixed budget for "look at the relevant image", and the anchor image displaces the target image within that budget.

This splits the three-family archetype further:

- **Gemma (SigLIP):** early + text-stealing. Anchor digit is read off the prompt during early prompt-integration layers; the pull comes from text mass, not from target-image mass.
- **Qwen (Qwen-ViT):** late + target-stealing. Anchor displaces the target image specifically at the answer-decision layer.
- **LLaVA-1.5 / InternVL3:** mid + text-stealing (mostly). Both images pool attention from text. This explains why these two are the "middle" A7 magnitudes and the most continuous profiles across layers.

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
| qwen2.5-vl-7b | 22 | +0.02794 [+0.01532, +0.04271] | +0.00259 [−0.00204, +0.00602] | **+0.02535** |
| llava-1.5-7b | 16 | +0.02369 [+0.01600, +0.03177] | +0.01398 [+0.00999, +0.01782] | +0.00970 |
| internvl3-8b | 14 | +0.02302 [+0.01273, +0.03766] | +0.01726 [+0.01064, +0.02362] | +0.00576 |
| gemma4-e4b | 5 | +0.05078 [+0.04758, +0.05417] | +0.04936 [+0.04628, +0.05260] | +0.00142 |

The Qwen bottom-decile row is particularly striking: at the model's own peak layer, the bottom-decile triplets show a delta whose CI *includes zero*. For resistant items, the model doesn't even develop a late-stack attention gap. For susceptible items the same layer fires at +0.028. A single layer in Qwen is carrying almost all of the A7 signal, and the signal exists only for susceptible items. That's the cleanest "susceptibility-gated answer-step circuit" in the four-model panel and a natural E4 intervention site for this family.

## Implications for E4 (mitigation) — candidate sites, to be tested

These are hypotheses generated from the observational per-layer + budget data; nothing has been ablated or re-weighted yet. The E1 writeup's conclusion stands ("a single universal mitigation will not serve"), and the per-layer data narrows the *candidate* intervention sites for E4:

- **SigLIP-Gemma:** input-side candidate. Target layer 5 or earlier, with the specific aim of denying the text→anchor pull that characterises that layer. Attention re-weighting at generation time is unlikely to work (peak is in the KV cache already). Options to try: project-feature masking on the second image, or KV-cache patching at the encoder / early LLM layer boundary.
- **Qwen-ViT:** late-stack attention re-weighting candidate at layer 22 ± 2 on the answer-step only, gated by a per-item susceptibility signal. Because at this layer the budget is anchor-vs-target (not anchor-vs-text), a down-weight on the anchor should push mass back to the target image rather than to text — the desired outcome.
- **CLIP-ViT (LLaVA-1.5) and InternViT (InternVL3):** mid-stack attention re-weighting candidate at layer ~14–16 on the answer-step. Because their budget source is mostly text, a down-weight on the anchor will return mass to text, not to target — less ideal than in Qwen, but still worth testing.

The panel is now consistent with a paper-level hypothesis: **the location *and the competing region* of cross-modal bias in the LLM stack are encoder-family-specific, and an architecture-aware mitigation design is required.** Whether this hypothesis survives the causal E4 intervention is TBD — this is observational mechanistic evidence, not a tested fix.

## Caveats

- `n=200` per model per step. The per-layer CI is tight at the peak but noisier at the tails. Bigger n would tighten the layer-choice confidence (e.g. is Qwen's peak really 22 or could it be 20-23?).
- "Peak layer" is picked as `argmax(delta_mean)`. With tight CIs at the peak this is stable, but the precise choice of 5 vs 6 for Gemma or 16 vs 17 for LLaVA-1.5 is within CI overlap.
- Still no ConvNeXt / FastViT data — ConvLLaVA and FastVLM use the `inputs_embeds` path and need the extraction script extended. Open question 1 from E1-preliminary-results.md is still open; that's the next E1 step.
- Layer-wise mass is averaged over heads. Head-level sparsity (a single head at layer 5 in Gemma carrying the signal) would be invisible here and would strengthen the circuit claim. Out of scope for E1b.

## Roadmap update

- §3 status: E1b layer analysis done (this file). Cross-family pattern now 3-way: SigLIP early / CLIP-Intern mid / Qwen late.
- §6 Tier 1 E1: "per-layer localisation" bullet from open-questions in E1-preliminary-results.md checked off. Still open: ConvNeXt/FastViT extension (next), causal test.
- §10 changelog entry dated 2026-04-24.
