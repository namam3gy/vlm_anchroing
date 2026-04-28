# §8 — Future work, F1: LLM vs. VLM architectural diff

**Status:** First-draft paper prose, 2026-04-29. Anchored to
`references/roadmap.md §6.6 row F1` and `references/project.md §0.5`.
This is the first paragraph of §8 (future work); F2 (image-vs-text anchor
on the same VLM) and F3 (reasoning-mode VLMs at scale) follow as
separate paragraphs. F1 is the preferred direction because it is the
one §7 hands an instrument to and the one whose result would
directly upgrade the headline claim from a VLM-only finding to a
cross-architecture statement about *where in the stack* an anchor
gets integrated as a function of its delivery modality.

---

## §8.1 — F1: same anchor, two modalities, layer-wise comparison

The §7 mechanistic story reads cleanly inside the VLM. The anchor
enters as image tokens, concentrates onto a digit-pixel patch (E1-patch
POC, 2026-04-29), is integrated into the residual stream at one of four
encoder-family-specific peak layers (E1b: SigLIP-Gemma early at
L5/42, the mid-stack-cluster CLIP-ViT/InternViT/ConvNeXt cluster at
L14–16, Qwen-ViT late at L22/28, FastVLM late at L22), and is
suppressible by an encoder-blind upper-half re-weighting at deployable
strength (E4 §7.4). What the paper does *not* test is whether the
*same* numerical anchor — delivered as plain text to a text-only LLM
backbone of the same family, e.g. presenting the integer in the prompt
rather than rendering it into a second image — produces an analogous
layer-wise integration profile. The two extremes bracket a clean
mechanistic question:

> **F1.** Is cross-modal numerical anchoring in VLMs the *image-modality
> instantiation* of a stack-level integration mechanism that exists
> identically in LLMs under text-modality delivery, or is the integration
> profile itself modality-specific?

The minimal experiment isolates modality-of-delivery while controlling
for everything else. (i) Pair each VLM in our panel with its underlying
LLM (the Vicuna behind LLaVA-1.5, the Qwen2.5 behind Qwen2.5-VL, the
InternLM behind InternVL3, the Gemma-3 behind Gemma-4, etc.; this
pairing is exact for a subset of our panel and only approximate for
the rest, which becomes a scope decision). (ii) Build a text-anchored
counterpart of the four-condition prompt: instead of a second image
containing the digit, append a JSON-like fragment in the user message
(e.g. *"For reference, an unrelated number is 7."*) so that the anchor
is identifiable in the prompt but task-irrelevant by construction —
the LLM analogue of the two-image VLM prompt. (iii) Run the §3
canonical metrics (`adopt_rate`, `direction_follow_rate`,
`exact_match`) on the LLM with the text anchor, and the same metrics
on the VLM with the image anchor, on a question subset where the
target image is description-replaceable into text without information
loss (numerical VQA on simple counting questions is the cleanest
bridge; ChartQA and MathVista are not). (iv) Re-run the §7 mechanism
extraction — attention-mass concentration, per-layer logit-lens trace
of the answer-token competition between `pred_b`-favoured and
`anchor`-favoured tokens — on the LLM stack with the text anchor, and
compare layer-by-layer to the corresponding VLM run.

Three empirical patterns would each shift the headline claim differently.
**(A) Same peak depth, same suppression locus.** If text-delivered
anchors integrate at the LLM analogue of the same upper-half locus that
§7.4's re-weighting targets in the VLM, then anchoring is one mechanism
under two delivery channels and the §7.4 mitigation is the LLM-side
intervention restated. **(B) Different peak depth, same effect.** If
text-delivered anchors integrate earlier (e.g. at first-layer KV cache
formation, where text tokens are already in the residual stream before
the upper half) but produce the same direction-follow magnitude, then
the integration locus is *modality-conditional* and §7.4 is genuinely
specific to the cross-modal case. **(C) Asymmetric magnitude.** If the
LLM with a text anchor shows substantially weaker (or stronger)
`direction_follow_rate` per matched uncertainty quartile (§6) than the
VLM with the image anchor, then the cross-modal channel carries an
information advantage (or disadvantage) for the anchor that is not
reducible to delivery modality at the integration site, and the
"digit-pixel cue" claim of §0.1 acquires a stronger justification —
the digit-pixel-as-pixels carries something a digit-as-token does not.

The instrument set we already own — eager-attention extraction across
six encoder families (E1), per-layer localisation (E1b),
upper-half causal ablation (E1d), digit-pixel-patch attention
(E1-patch POC) — transfers to LLM stacks one-for-one (drop the
image-token-span gating, keep the layer-block gating); we estimate one
to two weeks of compute for a five-pair LLM/VLM panel on the same n =
200 stratified subset that anchored §7. F1 is therefore the future
direction with the highest depth-per-additional-compute payoff: the
core question is mechanism-level, the experimental design has no
ambiguous controls, and the result is one figure (paired layer-wise
trace) plus one table (matched-confidence-quartile metrics) regardless
of which of (A)/(B)/(C) holds. F2 (image-vs-text anchor on the *same*
VLM) and F3 (reasoning-mode VLM at scale) are deferred to follow-up
work; F1 is the entry point.
