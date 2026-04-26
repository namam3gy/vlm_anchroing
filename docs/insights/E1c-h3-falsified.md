# E1c — H3 is dead: ConvNeXt replicates CLIP-ViT at the per-layer level, not just at adoption level

**Status:** New insight from the 6-model E1b panel (this round). Source data: `outputs/attention_analysis/{convllava-7b,llava-1.5-7b}/<run>/per_step_attention.jsonl`. Aggregate table: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. Full 6-model context: `docs/experiments/E1b-per-layer-localisation.md`.

## The hypothesis and its falsifier

**H3** (from `references/roadmap.md` §2): "Vision-encoder family modulates susceptibility. ConvNeXt/encoder-free should be *less* susceptible than CLIP/SigLIP-ViT (typographic-attack inheritance)."

**Falsifier** (pre-stated on H3): "If ConvLLaVA / EVE / DINO-VLM show statistically equivalent direction-follow gap to CLIP-ViT VLMs, H3 fails."

E2 pilot (1,125-sample) already reported adoption=0.156 for ConvLLaVA vs 0.181 for LLaVA-1.5 — statistically within the CLIP/SigLIP cluster. That was a *behavioural* falsification. H3's partial rescue was the possibility that the mechanism might still differ even when the behaviour doesn't. **The 6-model E1b run ends that rescue.**

## The data

At the answer-step peak layer, number − neutral anchor-mass delta, bootstrap 95 % CI, same n=200 susceptibility-stratified question set:

| Model | encoder | peak layer | depth % | peak δ | budget source at peak | A7 gap |
|---|---|---:|---:|---:|---|---:|
| **llava-1.5-7b** | CLIP-ViT | **16 / 32** | **52 %** | +0.0188 | text (−0.0294) | +0.0097 |
| **convllava-7b** | **ConvNeXt** | **16 / 32** | **52 %** | **+0.0224** | text (−0.0192) | +0.0126 |

- **Same peak layer** (L16 of 32 LLM layers).
- **Same budget source** (both pull anchor mass primarily from prompt text).
- **Same mechanism archetype** (mid-stack text-stealing; neither shows target-stealing).
- **Magnitudes within 19 %** (+0.0224 vs +0.0188).
- **A7 gap magnitudes within 30 %** (+0.013 vs +0.010).

Not "similar up to CI overlap" — their per-layer traces sit on top of each other across the entire LLM stack (see `fig_delta_by_layer_answer.png`). The ConvNeXt encoder, swapped for CLIP-ViT, produces almost the same per-layer attention fingerprint.

InternVL3 (InternViT) at L14 with δ +0.019 and A7 gap +0.006 is also in the same cluster. **Three different encoders — CLIP-ViT, InternViT, ConvNeXt — land on a single "mid-stack text-stealing" archetype.**

## What we expected versus what we got

Pre-registered by H3: ConvNeXt should suppress the anchor effect relative to CLIP-ViT because the published typographic-attack literature cites SigLIP- and CLIP-ViT-family encoders as the mechanism. ConvNeXt's spatial-hierarchy inductive bias was argued to be less susceptible.

What the per-layer data shows: the anchor arrives at exactly the same layer, with the same budget source, at almost the same magnitude. The *encoder architecture* (Conv vs. ViT) does not visibly modulate the per-layer attention signature. What *does* modulate it, per the 6-model panel, is **where in the LLM stack** the anchor gets read in — SigLIP-Gemma does it early, Qwen-ViT does it late, the mid-stack cluster does it around layer 14–16. That axis predicts peak δ, budget source, and A7 gap; encoder architecture does not.

## Why this changes the paper narrative

The roadmap's H3 framing ("ConvNeXt < ViT") was the paper's hook for a mechanistic claim grounded in encoder-architecture differences. With H3 falsified at both adoption and per-layer levels, that hook is gone — and we shouldn't mourn it: it's replaced by something sharper.

The new framing from the 6-model panel (see `docs/experiments/E1b-per-layer-localisation.md`):

- The bias's *location in the LLM stack* is encoder-family-specific, but **not in the way H3 predicted**. It's not Conv vs. ViT — it's SigLIP (early) vs. ViT/Conv cluster (mid) vs. Qwen-ViT (late) vs. FastViT (late + large).
- The *post-projection layer depth* axis predicts peak δ, budget source, and A7 gap; the encoder-architecture (ViT vs. Conv) axis does not.
- A single E4 intervention at mid-stack layers 14–16 is a plausible "encoder-agnostic" mitigation target — the three encoders that share this profile (CLIP-ViT, InternViT, ConvNeXt) are architecturally different, so an intervention that works at L16 is probably keying off LLM-stack properties, not encoder properties.

That last point is the actually-useful paper-level outcome of the falsification: the mid-stack cluster becomes the highest-leverage E4 target precisely *because* three encoders converge on it. A good mitigation there would survive swapping the vision encoder.

## What this doesn't say

- **Encoder architecture never matters.** SigLIP-Gemma really is an outlier, and its typographic-attack inheritance fits the data (early + unconditional + text-stealing + no susceptibility gating). Encoder architecture matters at the *tail* of the distribution (SigLIP outlier, and possibly FastViT's huge A7 gap), not across the bulk.
- **ConvLLaVA is safe to use.** Whatever ConvLLaVA is doing, it's almost exactly what LLaVA-1.5 does on this task. Using ConvLLaVA buys you nothing in terms of anchoring-reduction.
- **ConvNeXt never helps.** We tested ConvLLaVA only. A different Conv+LLM pairing — ConvNeXt with a different backbone, or a different ConvNeXt variant — could show a different profile. The claim is strictly about the architecture pair tested.
- **Behaviour matches mechanism matches attention.** Adoption, direction-follow, and per-layer attention all converge on the "ConvLLaVA ≈ LLaVA-1.5" reading. Other tests (Phase C: paraphrase robustness, closed-model subset) have not yet been done.

## Implications for the experiment plan

- **Replace H3 with the depth-axis framing in the paper.** The mechanistic story becomes "post-projection LLM depth is the axis on which the anchor signature varies; encoder architecture is not." The 6-model panel naturally supports this.
- **E4 mitigation prototype priority — mid-stack cluster first.** Because three encoders share the profile, a single intervention tuned on one (say LLaVA-1.5) can be tested on the other two (ConvLLaVA, InternVL3) as a portability check. If it ports, that's the paper's "architecture-agnostic E4" claim.
- **Drop the planned "encoder-ablation" subsection of E2.** E2 was designed as the H3 test; with H3 dead, E2 as originally framed has nothing left to prove. Re-purpose that compute toward E5 (multi-dataset) or E7 (paraphrase robustness).

## Caveats

- n=200 per model; the ConvLLaVA run and the LLaVA-1.5 run used different random seeds for the susceptibility strata sampling, but drew from the same susceptibility CSV. Peak-layer agreement (16 on both) is within CI overlap and unlikely to be a sampling artefact.
- The comparison is at answer-step peak. The *step-0* peaks do differ (ConvLLaVA L31/32 vs LLaVA-1.5 L26/32), and that difference is flagged in `docs/experiments/E1b-per-layer-localisation.md`. The prompt-integration step has a different architecture-dependence than the answer step; this insight is about the answer step specifically.
- The ConvLLaVA EagerConvLLaVARunner is a first integration (this round). Numbers should be treated as preliminary until an independent cross-check, even though the normalised-mass sanity (sum = 1 per layer per step) passes on every record.
- H3 remains untested on encoder-free VLMs (EVE, DINO-VLM). The headline is that ConvNeXt specifically doesn't help; the broader "encoder architecture matters" claim survives one weaker reading: that only outlier architectures (SigLIP early, possibly FastViT late) produce distinctive profiles.

## Roadmap entry

- §2 H3: status changes from "⚠️ Pilot does not support simple form" to "❌ Falsified at both adoption and per-layer levels. Replaced by the depth-axis framing in E1b." Add cross-link to this insight.
- §6 Tier 2 E2 encoder-ablation: scope reduced — H3 test no longer needed. Compute budget released for E5 or E7.
- §10 changelog entry dated 2026-04-24.
