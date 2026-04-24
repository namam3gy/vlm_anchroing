# E1b — The anchor attacks the LLM stack at a different layer, with a different mechanism, per encoder family

**Status:** Re-analysis of the E1 attention data (no new compute). Full writeup: `docs/experiments/E1b-per-layer-localisation.md`. Source data: `outputs/attention_analysis/{model}/{run}/per_step_attention.jsonl`. Aggregate tables: `outputs/attention_analysis/_per_layer/{per_layer_deltas,peak_layer_summary,peak_budget_decomposition}.csv`. Script: `scripts/analyze_attention_per_layer.py`. Reproducer: `notebooks/E1b_per_layer_localisation.ipynb`.

## The question

E1 reported `attention_anchor(number) − attention_anchor(neutral)` **averaged across layers** and found a clean +0.004 to +0.007 signal in all 4 encoder families at the answer-digit step. The averaging obscured two things:

1. **Where in the LLM stack does the anchor actually get read in?** A flat 0.005 across 28 layers and a 0.050 spike at one layer are indistinguishable in the layer-average but imply very different E4 intervention sites.
2. **What does the anchor's attention mass displace?** If the anchor gains at layer `l`, something at that layer must lose. Text? The target image? That distinction is the mechanism.

This insight answers both from the same E1 jsonl files.

## Method

For each triplet `(sample_instance × {base, number, neutral})` and each layer `l`, compute `delta_l = anchor_mass(number, l) − anchor_mass(neutral, l)` at the answer-digit step. Bootstrap 2,000 iter, 95 % CI, across the ~200 valid triplets per model. Then at each model's `argmax_l delta_l` layer, report the full budget: `δ(anchor)`, `δ(target image)`, `δ(text)`, `δ(generated)`. By construction these four sum to zero (attention is normalised); the signs tell us which region the anchor is competing against.

## Result 1: peak layer differs ~6× across encoder families

Answer-step peak layer and its delta, overall stratum:

| Model | peak layer | depth | peak δ | 95 % CI |
|---|---:|---:|---:|---|
| gemma4-e4b (SigLIP) | **5 / 42** | **12 %** | **+0.0501** | [+0.0477, +0.0523] |
| llava-1.5-7b (CLIP-ViT) | 16 / 32 | 52 % | +0.0188 | [+0.0144, +0.0231] |
| internvl3-8b (InternViT) | 14 / 28 | 52 % | +0.0193 | [+0.0135, +0.0254] |
| qwen2.5-vl-7b (Qwen-ViT) | **22 / 28** | **82 %** | +0.0153 | [+0.0084, +0.0231] |

Three tiers: SigLIP-Gemma very early and ~3× the magnitude of the other three; CLIP-ViT and InternViT mid-stack; Qwen-ViT late. The layer-averaged E1 number (≈ 0.005) was hiding a single-layer concentration worth 10× its value.

## Result 2: the budget source splits the panel into two mechanisms

At each model's peak layer, what does the anchor take mass from?

| Model | peak | δ anchor | δ target | δ text | δ gen |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | +0.0501 | −0.0096 | **−0.0380** | −0.0024 |
| qwen2.5-vl-7b | 22 | +0.0153 | **−0.0097** | −0.0051 | −0.0005 |
| llava-1.5-7b | 16 | +0.0188 | +0.0070 | **−0.0294** | +0.0036 |
| internvl3-8b | 14 | +0.0193 | −0.0038 | **−0.0143** | −0.0012 |

Two distinct mechanisms:

- **Text-stealing (Gemma, LLaVA-1.5, InternVL3).** The anchor pulls mass primarily from the prompt text. In LLaVA-1.5 the target image *also* gains (+0.007) — the two images pool attention and together pull from text. Consistent with "the anchor registers as another piece of visual evidence".
- **Target-stealing (Qwen-only).** The anchor takes mass mainly from the target image, not from text. At layer 22 the answer-generation circuit has a fixed budget for "look at the relevant image", and the anchor displaces the target within that budget.

## Result 3: Gemma's "negative-delta" layers around layer 5 are mostly mechanical, not anti-anchor

Layers 0–4, 6–10, 12, 17 all have `delta < 0` with CI excluding zero — superficially "anchor-aversive". Per-layer budget check:

- Layers 0–10 (except 5): `δ_anchor` and `δ_target` are equal and opposite (e.g. layer 2: δ_target = +0.0136, δ_anchor = −0.0135). **Anchor/target mass swaps — text is untouched.** Not active suppression; the two images are exchanging attention between themselves.
- Layer 5: `δ_anchor = +0.050, δ_text = −0.038, δ_target = −0.010`. The **only** layer where the anchor genuinely pulls from the text stream.
- Layer 17: `δ_anchor = −0.020, δ_text = +0.016`. Real reallocation away from the anchor — but to text, not to target.

The "knife-edge spike" imagery holds for layer 5; it does *not* apply to its flanking layers. The alternative-reading (pure budget redistribution artefact) is ruled out by the specific text-direction of the layer-5 pull.

## What this says for mechanism

- **Where the bias lives in the stack is encoder-family-specific, not universal.** A paper claim of the form "VLMs attend to the anchor image" is true on average but misleading in design — different encoders do it at different depths, with different competing regions.
- **SigLIP-Gemma's pattern fits a typographic-attack read-off.** Early layer + text-stealing + no susceptibility-gating (E1 Test 3) together look like "the digit is read off the image as text during prompt integration, regardless of downstream utility", which is the published failure mode of SigLIP-family encoders.
- **Qwen-Vit's pattern is anchor-vs-target competition at the answer layer.** Late layer + target-stealing + susceptibility-gated (bottom-decile CI includes zero at the same peak layer) is consistent with an answer-generation circuit with a fixed "which image to look at" budget that susceptible items expose.
- **CLIP-ViT / InternViT sit in between.** Mid-stack + text-stealing + moderate susceptibility gating. Not a crisp mechanism yet — the per-layer profiles are more continuous than either extreme.

## Implications for the experiment plan (`references/roadmap.md` §6)

- **E4 (mitigation) can no longer design one intervention.** Candidate sites differ by family (observational only, to be tested in E4):
  - SigLIP-Gemma: **input-side pre-layer-5** projection/KV patch. Answer-step re-weighting won't work — damage is in the cache.
  - Qwen-ViT: **late-stack layer 22 ± 2** attention re-weighting, gated by susceptibility. Down-weighting the anchor should return mass to the target image (the preferred outcome) because the budget is anchor-vs-target.
  - CLIP-ViT / InternViT: **mid-stack ~14–16** attention re-weighting. Returns mass to text, not target — less ideal, still worth testing.
- **ConvLLaVA + FastVLM (inputs_embeds path) are the missing rows.** Fills the 6-model panel the paper needs before we can claim "location varies by encoder family". Extending the extraction script is the next E1 step, per `references/roadmap.md` §6.
- **Causal test of the layer-5 pull (open question 4 in the E1 preliminary results).** Ablate anchor-image tokens at Gemma's layer 5 or earlier and measure the `direction_follow` delta. This is the cheapest mechanism-level claim we can make and a direct input to E4.

## Caveats

- `n = 200` per model per step; InternVL3 drops to `n = 135` at the answer step because some generations don't contain a digit token. Peak-layer CIs are tight but the exact choice (e.g. 5 vs. 6 in Gemma, 16 vs. 17 in LLaVA-1.5) is within CI overlap.
- `argmax delta` is stable but ignores multi-peak distributions. The full per-layer trace figures (`fig_delta_by_layer_answer.png`) show Gemma is single-spike while the three ViT-family models are more continuous.
- Layer-wise mass is averaged over heads. A head-level sparsity claim (e.g. "one head in Gemma layer 5 carries the signal") would strengthen the circuit story but is out of scope here.
- Attention correlates, not causes. E4 is where the causal claim gets tested.

## Roadmap entry

§6 Tier 1 E1: "per-layer localisation" open question checked off. Remaining: ConvNeXt/FastViT extension, causal test.
