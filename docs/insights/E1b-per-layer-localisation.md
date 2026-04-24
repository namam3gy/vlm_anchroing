# E1b — The anchor attacks the LLM stack at a different layer, with a different mechanism, per encoder family

**Status:** Re-analysis of the E1 attention data (no new compute). Full writeup: `docs/experiments/E1b-per-layer-localisation.md`. Source data: `outputs/attention_analysis/{model}/{run}/per_step_attention.jsonl`. Aggregate tables: `outputs/attention_analysis/_per_layer/{per_layer_deltas,peak_layer_summary,peak_budget_decomposition}.csv`. Script: `scripts/analyze_attention_per_layer.py`. Reproducer: `notebooks/E1b_per_layer_localisation.ipynb`.

## The question

E1 reported `attention_anchor(number) − attention_anchor(neutral)` **averaged across layers** and found a clean +0.004 to +0.007 signal at the answer-digit step (initially on 4 models, now extended to 6 with the inputs_embeds-path ConvLLaVA and FastVLM). The averaging obscured two things:

1. **Where in the LLM stack does the anchor actually get read in?** A flat 0.005 across 28 layers and a 0.050 spike at one layer are indistinguishable in the layer-average but imply very different E4 intervention sites.
2. **What does the anchor's attention mass displace?** If the anchor gains at layer `l`, something at that layer must lose. Text? The target image? That distinction is the mechanism.

This insight answers both from the same E1 jsonl files.

## Method

For each triplet `(sample_instance × {base, number, neutral})` and each layer `l`, compute `delta_l = anchor_mass(number, l) − anchor_mass(neutral, l)` at the answer-digit step. Bootstrap 2,000 iter, 95 % CI, across the ~200 valid triplets per model. Then at each model's `argmax_l delta_l` layer, report the full budget: `δ(anchor)`, `δ(target image)`, `δ(text)`, `δ(generated)`. By construction these four sum to zero (attention is normalised); the signs tell us which region the anchor is competing against.

## Result 1: peak layer differs ~6× across encoder families

Answer-step peak layer and its delta, overall stratum (6-model panel):

| Model | peak layer | depth | peak δ | 95 % CI | n |
|---|---:|---:|---:|---|---:|
| gemma4-e4b (SigLIP) | **5 / 42** | **12 %** | **+0.0501** | [+0.0477, +0.0523] | 200 |
| internvl3-8b (InternViT) | 14 / 28 | 52 % | +0.0193 | [+0.0135, +0.0254] | 135 |
| llava-1.5-7b (CLIP-ViT) | 16 / 32 | 52 % | +0.0188 | [+0.0144, +0.0231] | 200 |
| convllava-7b (ConvNeXt) | **16 / 32** | **52 %** | +0.0224 | [+0.0171, +0.0281] | 200 |
| qwen2.5-vl-7b (Qwen-ViT) | 22 / 28 | 82 % | +0.0153 | [+0.0084, +0.0231] | 200 |
| fastvlm-7b (FastViT) | **22 / 28** | **82 %** | **+0.0467** | [+0.0253, +0.0716] | 75 |

Four archetypes visible:

- **SigLIP-Gemma:** very early peak (L5, 12 % depth), ~3× magnitude of the three mid-stack models.
- **Mid-stack cluster (CLIP-ViT, InternViT, ConvNeXt):** all peak at layers 14–16 (~52 % depth) with δ ≈ +0.019–0.022. **H3 in "ConvNeXt < ViT" form is definitively falsified** — ConvLLaVA lands at the same peak layer as LLaVA-1.5 with almost identical magnitude.
- **Qwen-ViT:** late peak (L22, 82 %) with moderate δ.
- **FastViT:** late peak (L22, 82 %) but with **Gemma-level magnitude** (+0.047). FastVLM n=75 (max_new_tokens=24 gave 62 % digit coverage under the JSON prompt; roadmap §9 caveat) — CI is wider.

The layer-averaged E1 number (≈ 0.005) was hiding a single-layer concentration worth up to ~10× its value.

## Result 2: the budget source splits the panel into two mechanisms

At each model's answer-step peak layer, what does the anchor take mass from?

| Model | peak | δ anchor | δ target | δ text | δ gen |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | +0.0501 | −0.0096 | **−0.0380** | −0.0024 |
| qwen2.5-vl-7b | 22 | +0.0153 | **−0.0097** | −0.0051 | −0.0005 |
| llava-1.5-7b | 16 | +0.0188 | +0.0070 | **−0.0294** | +0.0036 |
| internvl3-8b | 14 | +0.0193 | −0.0038 | **−0.0143** | −0.0012 |
| convllava-7b | 16 | +0.0224 | −0.0025 | **−0.0192** | −0.0006 |
| fastvlm-7b | 22 | +0.0467 | −0.0139 | **−0.0343** | +0.0016 |

Two mechanisms — but now 5 of 6 models are in the text-stealing bucket, not 3 of 4:

- **Text-stealing (5 of 6: Gemma, LLaVA-1.5, InternVL3, ConvLLaVA, FastVLM).** Anchor pulls mass primarily from prompt text. In LLaVA-1.5 the target image actually *gains* (+0.007) — the two images co-aggregate and together pull from text. ConvLLaVA and InternVL3 show mild target-loss (≈ −0.003); FastVLM pulls moderately from both text (−0.034) and target (−0.014).
- **Target-stealing (Qwen-only).** Qwen is the sole model where the anchor takes mass mainly from the target image (−0.010) rather than text (−0.005). At layer 22 the answer-generation circuit has a fixed budget for "look at the relevant image", and the anchor displaces the target within that budget.

The combined "depth × mechanism" frame now yields four archetypes:

1. **Early + text-stealing + large** — SigLIP (Gemma). Typographic-attack read-off during prompt integration.
2. **Mid + text-stealing + moderate** — CLIP-ViT (LLaVA-1.5), InternViT (InternVL3), ConvNeXt (ConvLLaVA). Three different encoders converge on the same profile.
3. **Late + target-stealing + moderate** — Qwen-ViT (Qwen2.5-VL). The only model where the two images compete for attention at the answer-decision layer.
4. **Late + text-stealing + large** — FastViT (FastVLM). Magnitude rivals Gemma; location matches Qwen; budget source matches Gemma. Unique hybrid in the panel.

## Result 3: Gemma's "negative-delta" layers around layer 5 are mostly mechanical, not anti-anchor

Layers 0–4, 6–10, 12, 17 all have `delta < 0` with CI excluding zero — superficially "anchor-aversive". Per-layer budget check:

- Layers 0–10 (except 5): `δ_anchor` and `δ_target` are equal and opposite (e.g. layer 2: δ_target = +0.0136, δ_anchor = −0.0135). **Anchor/target mass swaps — text is untouched.** Not active suppression; the two images are exchanging attention between themselves.
- Layer 5: `δ_anchor = +0.050, δ_text = −0.038, δ_target = −0.010`. The **only** layer where the anchor genuinely pulls from the text stream.
- Layer 17: `δ_anchor = −0.020, δ_text = +0.016`. Real reallocation away from the anchor — but to text, not to target.

The "knife-edge spike" imagery holds for layer 5; it does *not* apply to its flanking layers. The alternative-reading (pure budget redistribution artefact) is ruled out by the specific text-direction of the layer-5 pull.

## What this says for mechanism

- **Where the bias lives in the stack is encoder-family-specific, not universal.** A paper claim of the form "VLMs attend to the anchor image" is true on average but misleading in design — different encoders do it at different depths, with different competing regions, and at different magnitudes.
- **H3 in "ConvNeXt < ViT" form is dead.** ConvLLaVA's per-layer profile is almost indistinguishable from LLaVA-1.5's (same peak layer, same mechanism, similar magnitude). Pure encoder architecture (Conv vs. ViT) does not predict anchoring. The *post-projection LLM-layer-depth* axis predicts it.
- **SigLIP-Gemma's pattern fits a typographic-attack read-off.** Early layer + text-stealing + no susceptibility-gating (E1 Test 3, A7 gap +0.001) together look like "the digit is read off the image as text during prompt integration, regardless of downstream utility", which is the published failure mode of SigLIP-family encoders.
- **FastViT (FastVLM) adds a new archetype: late + text-stealing + large + strongest A7 gating.** Peak δ +0.047 is 3× the other mid/late models and comparable to Gemma, but the peak sits at Qwen-ViT's depth (L22). A7 gap at peak = +0.086 — 3× Qwen's +0.025 and 8× ConvLLaVA's +0.013. Subject to the n=75 caveat, FastVLM looks like "a late-stack decision layer that, for susceptible items, turns into a typographic-read-off head". The two published failure modes — typographic-attack (read in-image text) and "wrong image for the question" — may be coinciding here.
- **Qwen-ViT's pattern is anchor-vs-target competition at the answer layer.** Late layer + target-stealing + susceptibility-gated (bottom-decile CI includes zero at the same peak layer) is consistent with an answer-generation circuit with a fixed "which image to look at" budget that susceptible items expose.
- **The mid-stack cluster (CLIP-ViT, InternViT, ConvNeXt) share a profile.** Mid-stack + text-stealing + moderate susceptibility gating. The cleanest "default VLM" mechanism in the panel — and the three-encoder replication makes it the most stable archetype.

## Implications for the experiment plan (`references/roadmap.md` §6)

- **E4 (mitigation) can no longer design one intervention.** Candidate sites now differ across *four* archetypes (observational only, to be tested in E4):
  - SigLIP-Gemma: **input-side pre-layer-5** projection/KV patch. Answer-step re-weighting won't work — damage is in the cache.
  - Mid-stack cluster (LLaVA-1.5 / InternVL3 / ConvLLaVA): **mid-stack ~14–16** attention re-weighting at the answer step. Returns mass to text, not target — less ideal, but the *three-encoder replication* makes this the highest-leverage target because one intervention could plausibly generalise to all three.
  - Qwen-ViT: **late-stack layer 22 ± 2** attention re-weighting, gated by susceptibility. Anchor down-weight returns mass to the target image (the preferred outcome) because the budget is anchor-vs-target.
  - FastViT-FastVLM: **late-stack layer 22 ± 2** re-weighting with a susceptibility gate. Because A7 gap is so large here (top-decile δ ≈ +0.10 vs bottom-decile ≈ +0.015), an A7-gated rescale could buy an outsized direction-follow reduction — but n is smallest here, so verify on a larger run before committing.
- **Causal test of the layer-5 pull** (open question 4 in `docs/experiments/E1-preliminary-results.md`). Ablate anchor-image tokens at Gemma's layer 5 or earlier and measure the `direction_follow` delta. This is the cheapest mechanism-level claim we can make and a direct input to E4.
- **inputs_embeds-path extension — DONE.** ConvLLaVA + FastVLM full n=200 runs are now in the panel (open question 1 closed). Remaining E1 open question: head-level sparsity + causal test.

## Caveats

- `n = 200` per model per step at step 0; at the answer step InternVL3 drops to n=135 and FastVLM drops to n=75, both because some generations don't emit a digit token within the generation budget. FastVLM in particular hit the roadmap §9 "prose before the digit" failure — the run used `max_new_tokens=24` (vs 8 for the other five) and still rescued only 62 % of records. FastVLM's wide answer-step CI ([+0.025, +0.072]) reflects this.
- `argmax delta` is stable but ignores multi-peak distributions. The full per-layer trace figures (`fig_delta_by_layer_answer.png`) show Gemma is single-spike while the four mid/late models are more continuous, with FastVLM visually rising monotonically through layers 10–22.
- Layer-wise mass is averaged over heads. A head-level sparsity claim (e.g. "one head in Gemma layer 5 carries the signal") would strengthen the circuit story but is out of scope here.
- FastVLM used `max_new_tokens=24`, the other five used 8. This affects the *distribution* of answer-step indices but not the answer-step mass per se — within each triplet the same step is compared across conditions.
- Attention correlates, not causes. E4 is where the causal claim gets tested.

## Roadmap entry

§6 Tier 1 E1: "per-layer localisation" + "ConvNeXt/FastViT extension" both closed. Remaining: head-level sparsity analysis, causal test (ablate anchor-image tokens at the per-family peak layer, measure `direction_follow`).
