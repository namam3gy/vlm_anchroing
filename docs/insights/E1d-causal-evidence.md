# E1d — peak-layer attention is correlational; the anchor's causal pathway is multi-layer

**Status:** Causal follow-up to E1b across the 6-model panel. Source data: `outputs/causal_ablation/<model>/<run>/predictions.jsonl`. Aggregate tables: `outputs/causal_ablation/_summary/{per_model_per_mode.csv, by_stratum.csv}`. Full writeup: `docs/experiments/E1d-causal-ablation.md`.

> **2026-04-28 note.** `direction_follow_rate` numbers below predate the
> C-form refactor (commit 6cba878). Pre-refactor results archived at
> `outputs/before_C_form/causal_ablation/`; current `outputs/` has the
> C-form numbers. Qualitative findings (single-layer null, upper-half
> ablation works 6/6) survive the refactor unchanged — verified in
> `docs/insights/C-form-migration-report.md`. Exact pp-shifts (e.g.
> "−5.5 to −11.5 pp") will be re-rendered in a follow-up sweep; until
> then treat absolute numbers as approximate and rank/sign comparisons
> as load-bearing.

## The claim and the test

E1b reported that anchor attention concentrates at a single LLM layer per encoder family — Gemma L5, mid-stack cluster L14–16, Qwen and FastVLM L22. The natural follow-up: if we *remove the model's ability to attend to the anchor at that layer*, does the model's anchor-pull behaviour drop?

We ablate the anchor span at six different layer sets (single peak, peak ± 2, lower half, upper half, all layers, layer 0 as control) for each of the 6 models, n=200 stratified, three conditions, and compare `direction_follow_rate` against baseline. See `scripts/causal_anchor_ablation.py` and `scripts/analyze_causal_ablation.py`.

## What we found

| Mode | What it tests | Result |
|---|---|---|
| `ablate_peak` | E1b's headline | **Null on 6/6 models** (|Δ df| ≤ 3.2 pp, all CIs overlap baseline) |
| `ablate_layer0` | Non-peak control | **Null on 6/6 models** (Δ df ∈ [−0.027, +0.005], all CIs overlap baseline) — including Gemma, whose E1b reported anchor↔target swaps at L0–4 |
| `ablate_all` | Upper bound on causal effect | −10 to −22 pp on direction_follow, but **fluency degrades on 3/6 models** (mean-distance balloons 4–6× or 1000×) |
| `ablate_upper_half` | Mitigation candidate | −5.5 to −11.5 pp on **6/6 models**, fluency clean on 4/6 (mid-stack cluster + Qwen); Gemma marginal; FastVLM broken |
| `ablate_lower_half` | Diagnostic | **Heterogeneous: 3/6 BACKFIRE, 1/6 reduce, 2/6 flat** |

The single-layer ablation null — at the E1b peak *and* at layer 0 — is the most surprising result. E1b's per-layer peak (with up to 5–10× the layer-averaged anchor mass) was the obvious E4 intervention site, and Gemma's L0–4 anchor↔target swaps were the most plausible "early-layer alternative". Neither is load-bearing on its own.

## Reading

Reading **(A) Multi-layer redundancy** is confirmed by the layer-0 control. The anchor's effect on the answer is *encoded redundantly* across the LLM stack — removing one layer's view of the anchor at any single site (peak, layer 0, or anywhere in between, on any of 6 models) leaves the rest of the stack to reconstruct it. The peak layer is where the signal is most visible, not where it is uniquely produced. Reading **(B) "peak is correlational and a different single layer matters"** is unsupported even on Gemma, whose E1b L0–4 anchor↔target swaps made it the most plausible "early-layer alternative" candidate (Gemma layer-0 Δ = +0.005, indistinguishable from its peak Δ = +0.020).

The pragmatic positive result: **`ablate_upper_half` is the single mitigation locus that works on the entire 6-model panel** without exploding fluency on the mid-stack cluster. For LLaVA-1.5, ConvLLaVA, InternVL3, upper-half ablation reduces `direction_follow_rate` by 5.5–9.9 pp while keeping `mean_distance_to_anchor` essentially flat. That is the strongest "architecture-blind" candidate the panel has yielded; the three encoders (CLIP-ViT, InternViT, ConvNeXt) at the same depth respond similarly.

## Why this matters

Two things change for the experiment plan and one stays the same.

**Changes for E4 design.** The per-family-peak attention re-weighting prototype previously planned by E1b is shelved — single-layer null at the peak *and* at layer 0 across all 6 models. Multi-layer intervention is required for any architecture in the panel. The simplest E4 prototype that works is *upper-half anchor attention re-weighting on the mid-stack cluster*.

**Changes for the paper claim.** The "where in the LLM stack the anchor gets read in" framing from E1b/E1c is *correlationally* sharp but does not imply *causal* per-family intervention sites. The paper text should distinguish:

- E1b finding: per-family peak layer is where the anchor signal concentrates in attention space (correlational).
- E1d finding: that single layer is not causally load-bearing on its own (single-layer attention-mask ablation null). The layer-0 control rules out the alternative "the E1b peak is the wrong single layer" — single-layer ablation is insufficient regardless of which layer.
- E1d finding: upper-half is the single architecture-blind mitigation locus that works on the panel — fluency-clean on the mid-stack cluster + Qwen, marginal on Gemma, broken on FastVLM.

**What stays.** The four-archetype encoder-family taxonomy from E1b is still the cleanest description of the *attention* signature. We are not retracting it. We are saying it's correlationally informative, not a recipe for single-layer causal interventions.

## Sub-finding flagged as "do not generalise to single architecture cluster"

ConvLLaVA and LLaVA-1.5 share the E1b answer-step peak (L16), the same text-stealing budget source, and almost-identical magnitudes. **They respond *opposite* to lower-half ablation.** ConvLLaVA `delta_df = −0.120`, LLaVA-1.5 `delta_df = +0.165`. Same-attention-signature does not imply same-causal-structure. We treat this as a caveat against treating "mid-stack cluster" as a *causally* uniform group; it remains a useful *attention-signature* group.

## What this doesn't say

- **The anchor isn't causal.** It is. `ablate_all` reduces `direction_follow` by 11–22 pp universally. The anchor is causally load-bearing in the aggregate; we just can't pin the entire effect on one layer.
- **The peak layer isn't useful.** It is the strongest correlational marker we have for where the anchor signature is in each model. It just isn't a single-layer intervention site.
- **All multi-layer ablations are equivalent.** They are not. `ablate_lower_half` BACKFIRES on 3/6 models (Gemma +27 pp, LLaVA-1.5 +16.5 pp). Multi-layer ablation needs the right *layers*, not just multiple layers.

## What we did NOT test

- Per-head sparsity. A single attention head at the peak layer might still be load-bearing where the layer-aggregate is not. This is a strict generalisation of the present test and is a clean follow-up.
- Vision-token re-projection (zero the anchor's embeddings before they enter the LLM). E1d only manipulates attention masks; the anchor still occupies the residual stream. The "deny the anchor input entirely" intervention class is not tested here.
- Combination ablations (peak + a complementary layer). The single-layer null suggests the right combination might recover the `ablate_all` effect more cleanly than `ablate_all` itself does.

## Implications for the experiment plan

- **E4 intervention class — open question 1 closed.** Single-layer attention re-weighting (at the E1b peak or layer 0) is ruled out as a candidate. Upper-half re-weighting is the prototype.
- **E4 architecture coverage.** The mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) is the single highest-leverage prototype target — three encoders, one shared upper-half-clean response.
- **A new open question: multi-layer combinatorial ablation.** Does the union of (anchor span ablated at peak) + (anchor span ablated at one complementary layer) recover an `ablate_all`-magnitude reduction at lower fluency cost? Cheap to run; informs E4 directly.
- **Roadmap §6 E1 row:** open question "causal test" → **closed (null on single layer; layer-0 control confirms multi-layer redundancy across 6/6 models)**. Open new question: head-level sparsity. Open new question: multi-layer combinatorial ablation.

## Caveats

- Bootstrap CIs are over triplets, not over pairs. Slightly conservative.
- **InternVL3 and FastVLM both lose triplets** under several ablation modes (no-digit emissions). The "BACKFIRE" we report on InternVL3 lower-half is on n=83 vs baseline n=137 — survivor-bias caveat applies. FastVLM under `ablate_all` and `ablate_upper_half` similarly drops triplets. Read those entries as suggestive, not paired.
- The mask is `−1e4` not `−inf` (bf16-safe). Spot-checked on LLaVA-1.5: post-hook anchor-column attention weights are < 1e-4 across all heads at targeted layers. The intervention is empirically a strong suppression, not a guaranteed zero.
- Susceptibility strata loaded from `docs/insights/_data/susceptibility_strata.csv`; same as E1b.
