# §7. Mechanism + mitigation

## 7.1 Anchor attention mass — the layer-averaged signature (E1)

For each of six panel models (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b,
internvl3-8b, convllava-7b, fastvlm-7b), n=200 stratified, we capture
per-layer attention from each generation step to each image-token span
(target image, anchor image, text). The aggregate "anchor attention
mass" — fraction of attention weight directed at the anchor image
tokens — is positive and significant on 4/4 base-experiment models at
the answer step (mean +0.004 to +0.007, CI excludes 0). Anchor
attention is *not* a baseline-level artifact.

Two sub-findings:

- **Wrong>correct attention asymmetry is falsified at 4/4 (and
  later 6/6) models.** Anchor attention mass does not shift up on
  uncertain items. The uncertainty modulation seen behaviourally in
  §5-§6 does not show up in mean attention.
- **Susceptibility (A7 cross-model agreement) tracks attention on
  3/4 models** at the answer step, *inverts* on Gemma-SigLIP — which
  also concentrates anchor attention at step 0 rather than the answer
  step. Consistent with SigLIP's known typographic-attack
  inheritance.

## 7.2 Per-layer localisation — four encoder-family archetypes (E1b)

Layer-resolved E1 attention reveals that anchor mass concentrates at
*one peak layer per encoder family*, with up to 3× the layer-averaged
mass:

| Archetype | Reference model | Peak layer | δ at peak | Budget source |
|---|---|---:|---:|---|
| SigLIP-Gemma early | gemma4-e4b | L5 / 42 (12 % depth) | +0.050 | text-stealing |
| Mid-stack cluster | llava-1.5-7b (CLIP-ViT) | L16 / 32 | +0.019 | text-stealing |
| Mid-stack cluster | convllava-7b (ConvNeXt) | L16 / 32 | +0.022 | text-stealing |
| Mid-stack cluster | internvl3-8b (InternViT) | L14 / 28 | +0.019 | text-stealing |
| Qwen-ViT late | qwen2.5-vl-7b | L22 / 28 (82 %) | +0.015 | target-stealing |
| FastVLM late text-stealing | fastvlm-7b | L22 | +0.047 | text-stealing |

The "mid-stack cluster" — three encoders (CLIP-ViT, ConvNeXt,
InternViT) of architecturally distinct kinds — converges on the same
layer-depth signature. This is the highest-leverage mitigation target.

(H3 retired: encoder-architecture per-se does not predict anchoring
susceptibility at either the behavioural or per-layer level. Three
architecturally different encoders converge on the same mid-stack
signature; the depth-axis framing replaces the architecture-axis
framing for §7.)

## 7.3 Causal ablation — single-layer null, upper-half mitigation works (E1d)

We ablate the anchor span at six different layer sets (single peak,
peak ± 2, lower half, upper half, all layers, layer 0 as control)
for each of the 6 models, n=200 stratified, three conditions.

| Mode | Result |
|---|---|
| `ablate_peak` (E1b's headline) | **Null on 6/6 models** (\|Δ df\| ≤ 2.0 pp; all CIs overlap baseline) |
| `ablate_layer0` (non-peak control) | Null on 6/6 models (Δ df ∈ [−2.7, +0.5] pp) |
| `ablate_all` | **−9.6 to −24.5 pp** on direction-follow, but **fluency degrades on 3/6 models** (mean-distance balloons 4-6× or 1000×) |
| `ablate_upper_half` (mitigation candidate) | **−4.0 to −10.5 pp** on **6/6 models**, fluency-clean on 4/6 (mid-stack cluster + Qwen) |
| `ablate_lower_half` (diagnostic) | **Heterogeneous: 3/6 BACKFIRE, 1/6 reduce, 2/6 flat** |

The single-layer ablation null at the E1b peak *and* at layer 0
(including Gemma's L0-4 anchor↔target swaps) is the surprising
result. The anchor's effect on the answer is *encoded redundantly*
across the LLM stack: removing one layer's view of the anchor at any
single site leaves the rest of the stack to reconstruct it. The peak
layer is where the signal is most visible, not where it is uniquely
produced.

**Upper-half attention re-weighting is the single architecture-blind
mitigation locus that works on the entire 6-model panel** without
exploding fluency on the mid-stack cluster.

## 7.4 Mitigation — upper-half soft re-weighting (E4)

We replace hard masking with a soft strength axis (`exp(strength)`
multiplier on anchor attention), sweep it on n=200 stratified samples
to pick a per-model `s*`, and validate at full scale on the
mid-stack-cluster (LLaVA-1.5, ConvLLaVA, InternVL3), 17,730 base
questions × 3 conditions × 2 modes per model.

**Phase 2 headline (n=88,650 records per model):**

| Model | s* | df baseline | df treated | df Δ pp | df rel | em baseline | em treated | em Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7b | −3.0 | 0.288 | 0.246 | **−4.19** | **−14.6 %** | 0.334 | 0.342 | +0.77 |
| ConvLLaVA-7b | −2.0 | 0.258 | 0.233 | **−2.49** | **−9.6 %** | 0.352 | 0.365 | +1.30 |
| InternVL3-8b | −0.5 | 0.126 | 0.119 | −0.74 | −5.8 % | 0.590 | 0.595 | +0.49 |

**Three patterns hold across the panel:**

- df decreases on all three; em rises on all three (+0.49 to +1.30 pp).
- Per-model `s*` is required (range −0.5 to −3.0, an order of
  magnitude). The mitigation generalises *as a locus + selection
  rule*, not as a single strength constant. (The `s*` values were
  selected by the original Phase 1 sweep under the Phase A pull-form
  direction-follow rate; under the canonical C-form sweep
  re-aggregation a slightly stronger `s*` is preferred for
  LLaVA-1.5/ConvLLaVA, but Phase 2 full-scale validation only exists
  at the historical pull-form `s*` and is reported here under C-form.)
- `accuracy_vqa(b)` (target-only baseline) is invariant on every
  strength on every model. The hook does not leak into single-image
  inference; it acts only on the anchor pathway.

**Paired anchor-damage** on the full sets (n_paired ~17,700 per
model except InternVL3 11,848 — parse-loss caveat):

| Model | em(target_only) | em(num@0) | em(num@s*) | damage | recovery | % recovered |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7b | 0.370 | 0.334 | 0.342 | −3.55 pp | +0.77 pp | 21.7 % |
| ConvLLaVA-7b | 0.445 | 0.352 | 0.365 | −9.34 pp | +1.31 pp | 14.0 % |
| InternVL3-8b | 0.633 | 0.594 | 0.598 | −3.87 pp | +0.40 pp | 10.2 % |

The paired damage / recovery picture is coherent across the cluster:
each model loses 4-9 pp of accuracy when the anchor is shown, and
the upper-half re-weighting recovers 10-22 % of that damage with no
target-only side-effect.

The relative df-reduction is *anti-correlated* with the model's
baseline anchor-pull — InternVL3 (lowest df₀ = 0.126) shows the
smallest relative reduction (−5.8 %), LLaVA-1.5 (highest df₀ =
0.288) shows the largest (−14.6 %). Conjecture (testable in a
follow-up): the upper-half attention pathway carries a *larger
fraction* of the anchor signal in models that use it less; LLaVA
and ConvLLaVA's anchor signals are broadly distributed and the
upper-half re-weighting hits a representative slice; InternVL3's
anchor signal is narrowly concentrated and the re-weighting
displaces less.

## 7.5 Why the mitigation is "free-lunch"

E4 reduces direction-follow without hurting accuracy. Specifically:

- `accuracy_vqa(b)` (target-only) invariant on every model on every
  strength → hook does not leak into single-image inference.
- `accuracy_vqa(d)` (neutral arm) within ±0.5 pp of baseline →
  hook fires but second image carries no legible digit, so no signal
  to remove.
- `exact_match(a)` rises +0.49 to +1.30 pp on all three models →
  the *only* condition where predictions move under the hook is the
  anchor condition, and they move toward correct answers.

The paired-anchor-damage analysis above strengthens this: the
mitigation recovers a meaningful fraction of the anchor's accuracy
hit. Reading: the upper-half pathway carries part of the
anchor-pull signal *that is not load-bearing for the model's own
answer-formation pipeline*. Damping it removes anchor influence
without disrupting normal inference.

## 7.6 §7 summary

The mechanism panel converges on a multi-layer-redundant anchor
pathway with one observable peak per encoder family. Single-layer
attention-mask ablation is causally null on 6/6 models, ruling out
the "peak = causal site" reading. Upper-half attention re-weighting
on the mid-stack-cluster is a fluency-clean mitigation that reduces
direction-follow by 5.8-14.6 % relative while exact-match rises
0.49-1.30 pp and target-only accuracy is invariant — the
"free-lunch" claim. The mitigation is per-model-`s*`-tuned at the
locus + rule level, not at a single strength constant.
