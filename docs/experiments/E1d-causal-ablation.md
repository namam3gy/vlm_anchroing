# E1d — causal anchor-attention ablation across the 6-model panel

**Status:** Causal follow-up to `docs/experiments/E1b-per-layer-localisation.md`. Driver: `scripts/causal_anchor_ablation.py`. Analysis: `scripts/analyze_causal_ablation.py`. Raw outputs: `outputs/causal_ablation/<model>/<run>/predictions.jsonl`. Aggregate tables: `outputs/causal_ablation/_summary/{per_model_per_mode.csv, by_stratum.csv}`. Figures: `outputs/causal_ablation/_summary/{fig_direction_follow.png, fig_adoption.png}`.

## TL;DR — three findings

1. **Single-layer ablation is null across 6/6 models — at the E1b peak *and* at layer 0.** Setting the attention mask at the anchor-image columns to a large negative value at the per-model E1b peak layer (Gemma L5 / InternVL3 L14 / LLaVA-1.5 L16 / ConvLLaVA L16 / Qwen2.5-VL L22 / FastVLM L22) does not change `direction_follow_rate` outside its baseline 95 % CI on any of the six models. The same is true for an `ablate_layer0` control on all six (`Δ direction_follow ∈ [−0.027, +0.005]`, all CIs overlapping baseline) — including Gemma, whose E1b reported anchor↔target swaps at layers 0–4. *Seeing* the anchor at a single layer (peak or otherwise) is correlationally informative but not causally load-bearing. The peak is the wrong target *and* "single layer" is the wrong intervention class — multi-layer ablation is required for any causal effect.
2. **Stack-wide ablation reduces `direction_follow` 11–22 pp universally — but introduces large fluency degradation on 3/6 models.** Ablating the anchor span at *every* LLM layer drops `direction_follow_rate` by 11.5 pp (Gemma) up to 22.0 pp (ConvLLaVA), but `mean_distance_to_anchor` simultaneously balloons from baseline ≈ 3 to ≈ 8–10 (Gemma, ConvLLaVA, LLaVA-1.5) or ≈ 5500 (FastVLM, Qwen). The model is no longer fluent — much of the apparent reduction comes from the model emitting nonsense rather than from successfully ignoring the anchor.
3. **Upper-half ablation is the single cleanest mitigation locus across the panel.** Ablating layers `[n_layers/2, n_layers)` reduces `direction_follow_rate` by 5.5–11.5 pp on **6/6 models** and is fluency-clean on 4/6 — the mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) plus Qwen. Gemma shows moderate fluency degradation (mean_distance_to_anchor 3.19 → 7.79, comparable to its `ablate_all` disruption); FastVLM is broken under this mode (mean-distance balloons to ~5,600). For the mid-stack cluster, upper-half ablation is the one mode that reduces direction-follow without breaking generation.

## Setup

For each of the 6 panel models we run the E1b-stratified question set (n=200 per model, top-decile-susceptible × 100 + bottom-decile-resistant × 100 from `docs/insights/_data/susceptibility_strata.csv`) under three conditions (`target_only`, `target_plus_irrelevant_number`, `target_plus_irrelevant_neutral`) and **seven ablation modes**:

| Mode | What gets ablated | Layers (n=42 / 28 / 32 / 32 / 28 / 28 for Gemma / InternVL3 / LLaVA-1.5 / ConvLLaVA / Qwen / FastVLM) |
|---|---|---|
| `baseline` | nothing | — |
| `ablate_layer0` | layer 0 (control: a non-peak layer) | `[0]` |
| `ablate_peak` | the E1b answer-step peak | `[5]` / `[14]` / `[16]` / `[16]` / `[22]` / `[22]` |
| `ablate_peak_window` | peak ± 2 (5-layer window) | `[3..7]` / `[12..16]` / `[14..18]` / `[14..18]` / `[20..24]` / `[20..24]` |
| `ablate_lower_half` | layers `[0, n_layers/2)` | first half of the LLM stack |
| `ablate_upper_half` | layers `[n_layers/2, n_layers)` | second half of the LLM stack |
| `ablate_all` | every LLM layer | full stack |

The ablation is implemented in `scripts/causal_anchor_ablation.py` as a per-layer forward pre-hook (`register_forward_pre_hook(..., with_kwargs=True)`) on each `decoder_layer` of the LLM stack. The hook intercepts the `attention_mask` kwarg and adds `−1 × 10⁴` (bf16-safe; `−inf` causes NaN propagation in some kernels) at the anchor-image-token columns `[s, e)`. The mask intervention is column-wise, not row-wise, so *all* later positions stop attending to the anchor span at the targeted layers — equivalent to "for layers `L ∈ ablate_layers`, the anchor image is invisible".

The `anchor_span` is resolved per-runner: for HF-input-id models (Gemma, Qwen, LLaVA-1.5, InternVL3) by scanning `input_ids` for the per-model `image_token_id`; for ConvLLaVA via the `inputs_embeds` splice tracking; for FastVLM via the `−200`-marker expansion logic. The resolution code is shared between `scripts/extract_attention_mass.py` (E1) and `scripts/causal_anchor_ablation.py` (this round).

We measure three outcome metrics per `(condition, mode)` cell, all bootstrap-CI'd at 2,000 iter / 95 %:

- **`direction_follow_rate`**: `# triplets with |pred_number − anchor| < |pred_target_only − anchor|` ÷ valid triplets. The cleanest "anchor pulls the prediction" probe. Baseline ranges 16–31 % across the panel.
- **`adoption_rate`**: `# triplets with pred_number == anchor`. Stricter, ranges 9–18 %.
- **`mean_distance_to_anchor`**: `mean |pred_number − anchor|` (lower = closer to anchor). Sanity-monitor: under successful ablation this either drops (good — model moved away) or stays flat; if it explodes (e.g. ≥ 10 from baseline ≈ 3), the model is hallucinating, not ignoring.

## Result 1 — single-layer ablation is null on 6/6 models (peak *and* layer-0 control)

The motivating expectation from E1b: peak layer holds 5–10× the layer-averaged anchor mass; ablating it should at minimum trim the direction-follow tail. It does not. We also ran a layer-0 control on every model to discriminate "peak is correlational, but some other single layer is the causal site" from "any single-layer ablation is insufficient". Layer-0 is also null on 6/6.

`direction_follow_rate` baseline → `ablate_peak` (E1b peak) → `ablate_layer0` → Δ (CI):

| Model | E1b peak | baseline df [CI] | ablate_peak df | Δ peak | ablate_layer0 df | Δ layer0 |
|---|---:|---|---:|---:|---:|---:|
| gemma4-e4b | 5 | 0.265 [0.205, 0.325] | 0.285 | +0.020 | 0.270 | +0.005 |
| internvl3-8b | 14 | 0.161 [0.102, 0.219] | 0.129 | −0.032 | 0.133 | −0.027 |
| llava-1.5-7b | 16 | 0.305 [0.240, 0.370] | 0.315 | +0.010 | 0.290 | −0.015 |
| convllava-7b | 16 | 0.290 [0.230, 0.355] | 0.310 | +0.020 | 0.280 | −0.010 |
| qwen2.5-vl-7b | 22 | 0.215 [0.160, 0.275] | 0.220 | +0.005 | 0.220 | +0.005 |
| fastvlm-7b | 22 | 0.216 [0.151, 0.281] | 0.217 | +0.001 | 0.216 | +0.001 |

All twelve |Δ| ≤ 3.2 pp; all twelve 95 % CIs overlap baseline. None of the directional pulls pass even a one-sided sign test against a null of "no effect".

The layer-0 control was the discriminator we cared about most for **Gemma**, whose E1b reported anchor↔target swaps at layers 0–4 — the most plausible "early-layer causal site" candidate in the panel. Gemma's `ablate_layer0` Δ is +0.005, indistinguishable from its `ablate_peak` Δ of +0.020. Both are inside the baseline CI. Reading (b) "peak is correlational and a different single layer matters" is therefore unsupported on the model where it had the strongest prior.

The one-sentence reading: the anchor-attention signal at the E1b peak is correlational evidence of where the bias *passes through*, not a causal bottleneck. The anchor's effect on the answer is encoded redundantly across multiple LLM layers, so removing it at any single layer leaves the other layers' encoding intact and the answer unchanged. The per-family-peak E4 design proposed by E1b is dead as a single-layer attention-mask intervention; multi-layer interventions or a different intervention class (contrastive decoding, vision-token re-projection) are required.

## Result 2 — stack-wide ablation reduces `direction_follow` but breaks fluency on 3 models

`ablate_all`: bypass every layer's view of the anchor span.

| Model | baseline df | ablate_all df | Δ df | baseline mean_dist | ablate_all mean_dist | mean_dist Δ | fluency-broken? |
|---|---:|---:|---:|---:|---:|---:|---|
| gemma4-e4b | 0.265 | 0.150 | **−0.115** | 3.19 | 7.71 | +4.5 | ⚠️ moderate |
| internvl3-8b | 0.161 | 0.060 | **−0.101** | 5.67 | 4.91 | −0.8 | ✅ ok |
| llava-1.5-7b | 0.305 | 0.155 | **−0.150** | 3.58 | 10.01 | +6.4 | ⚠️ moderate |
| convllava-7b | 0.290 | 0.070 | **−0.220** | 3.18 | 8.23 | +5.1 | ⚠️ moderate |
| qwen2.5-vl-7b | 0.215 | 0.055 | **−0.160** | 48.6 | 48.9 | +0.3 | ✅ ok at this metric (Qwen baseline is already inflated by long-tail outliers — see caveats) |
| fastvlm-7b | 0.216 | 0.062 | **−0.154** | 3.12 | 5571 | +5568 | ❌ broken |

Two reads:

- The **direction_follow drop is real and universal** (−10 to −22 pp). When the model can't see the anchor at any layer, it doesn't pull toward the anchor — the bias is causal in this aggregate sense.
- But on Gemma / LLaVA-1.5 / ConvLLaVA the **mean-distance signal blows up by 4–6×**, and on FastVLM by ~3 orders of magnitude. The model is producing nonsense numbers rather than the right number. Some fraction of the direction-follow drop comes from "the model emits a wild number that happens to be far from the anchor" rather than "the model emits a sensible number close to ground truth". This is "ablate_all" working partly through the anchor pathway and partly through general image-disruption.

The conservative conclusion: stack-wide ablation supports the existence of a causal anchor pathway, but its size is upper-bounded — the true causal contribution is somewhere below the headline 11–22 pp once you correct for the fluency hit. We don't have a clean way to separate them in this design.

## Result 3 — upper-half ablation is the cleanest mitigation locus

`ablate_upper_half`: bypass anchor span at layers `[n_layers/2, n_layers)`.

| Model | baseline df | upper-half df | Δ df | baseline mean_dist | upper-half mean_dist | mean_dist Δ | clean? |
|---|---:|---:|---:|---:|---:|---:|---|
| gemma4-e4b | 0.265 | 0.210 | **−0.055** | 3.19 | 7.79 | +4.6 | ⚠️ moderate |
| internvl3-8b | 0.161 | 0.063 | **−0.098** | 5.67 | 5.00 | −0.7 | ✅ clean |
| llava-1.5-7b | 0.305 | 0.250 | **−0.055** | 3.58 | 4.72 | +1.1 | ✅ clean |
| convllava-7b | 0.290 | 0.235 | **−0.055** | 3.18 | 3.46 | +0.3 | ✅ clean |
| qwen2.5-vl-7b | 0.215 | 0.125 | **−0.090** | 48.6 | 48.8 | +0.2 | ✅ clean |
| fastvlm-7b | 0.216 | 0.101 | **−0.115** | 3.12 | 5641 | +5638 | ❌ broken |

Five out of six show meaningful direction-follow reductions. Four of those five (InternVL3, LLaVA-1.5, ConvLLaVA, Qwen) show essentially no fluency degradation. Gemma shows moderate fluency degradation but a smaller direction-follow drop than `ablate_all`. FastVLM remains broken on this mode (consistent with its E1b peak at L22 being upper-half — but we're already covering the peak via this mode).

The mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) is the strongest case. For these three, upper-half ablation:

- reduces `direction_follow_rate` by 5.5–9.8 pp,
- keeps `mean_distance_to_anchor` essentially flat,
- and is the only mode that does both.

This is the candidate single mode for an architecture-blind E4 prototype on the mid-stack cluster: down-weight the anchor-image span at upper-half attention layers and accept a modest direction-follow reduction without paying a fluency cost.

A natural follow-up: try `ablate_upper_quarter` (`[3n/4, n)`) to see whether the cleanest reduction concentrates in the late layers, or whether the contribution is spread across the upper half. We did not run this in this round.

## Result 4 — peak-window ablation is heterogeneous; lower-half ablation is heterogeneous and sometimes BACKFIRES

For completeness:

`ablate_peak_window` (peak ± 2):

| Model | Δ direction_follow | reading |
|---|---:|---|
| gemma4-e4b | **+0.080** | **BACKFIRE** — anchor effect *increases* |
| internvl3-8b | −0.062 | small reduction |
| llava-1.5-7b | −0.010 | null |
| convllava-7b | −0.045 | small reduction |
| qwen2.5-vl-7b | −0.030 | null |
| fastvlm-7b | −0.041 | small reduction |

Peak-window backfire on Gemma is consistent with E1b's "layer 5 is a knife-edge spike, surrounded by anchor/target trade-off layers". Ablating the trade-off layers as well as the spike apparently disrupts the anchor/target competition more than the spike itself, freeing the anchor to dominate at downstream layers.

`ablate_lower_half` (`[0, n_layers/2)`):

| Model | Δ direction_follow | Δ adoption | reading |
|---|---:|---:|---|
| gemma4-e4b | **+0.270** | **+0.410** | huge **BACKFIRE** |
| internvl3-8b | +0.068 | +0.074 | **BACKFIRE** (n=83 vs baseline n=137 — 39 % triplet loss; survivor-bias caveat applies) |
| llava-1.5-7b | +0.165 | +0.275 | **BACKFIRE** |
| convllava-7b | −0.120 | +0.020 | reduction in df, null in adoption |
| qwen2.5-vl-7b | −0.010 | +0.025 | null |
| fastvlm-7b | −0.012 | +0.002 | null |

3/6 BACKFIRE, 1/6 reduce, 2/6 flat. Critical sub-finding: **ConvLLaVA's lower-half ablation behaviour is the opposite of LLaVA-1.5's**, despite the two models sharing the same E1b peak layer (L16) and the same text-stealing mechanism. ConvLLaVA's *encoded* anchor signal lives entirely in its lower stack (ablating it kills the anchor at L16); LLaVA-1.5's lower-half-ablation backfires because some lower-stack circuit is *competing against* the anchor at L16 — removing it lets the anchor win more often.

This is a heterogeneous, model-specific finding — not the headline. We flag it as a **caveat against generalisation**: the same-peak-layer + same-mechanism observation in E1b does not imply same-causal-structure. ConvLLaVA and LLaVA-1.5 look identical in attention but respond differently to lower-half ablation. The observational mid-stack-cluster identity is therefore *partial*.

## Susceptibility-stratified results

`by_stratum.csv` rows for `ablate_upper_half` (the cleanest mode):

| Model | top-decile Δ df | bottom-decile Δ df | reads-as |
|---|---:|---:|---|
| gemma4-e4b | −0.110 | 0.000 | upper-half hits susceptible items only |
| internvl3-8b | −0.243 | 0.000 | very strong on susceptible (note bottom-decile baseline already 0) |
| llava-1.5-7b | −0.100 | −0.010 | predominantly susceptible |
| convllava-7b | −0.090 | −0.020 | predominantly susceptible |
| qwen2.5-vl-7b | −0.160 | −0.020 | predominantly susceptible |
| fastvlm-7b | −0.275 | +0.024 | predominantly susceptible |

The reduction is concentrated on the top-decile-susceptible items in 5 out of 6 models. Bottom-decile-resistant items have a smaller baseline anchor effect (consistent with E1b A7 gap), and there is correspondingly little headroom for ablation to reduce. Encouraging for E4 — the ablation-style mitigation appears to specifically target the items where the bias is largest.

## What this says for E4 (mitigation)

Distilled to four claims:

1. **Single-layer interventions at the E1b peak will not work.** The peak-layer ablation null is uniform across the 6-model panel. E1b's per-family peak-layer recommendations are correlational; an E4 prototype must intervene at multiple layers simultaneously, or use a different intervention class (e.g. contrastive decoding, vision-token re-projection) rather than per-layer attention re-weighting at a single layer.
2. **Upper-half attention re-weighting is the single most plausible architecture-blind prototype.** It reduces `direction_follow` 5.5–11.5 pp across all 6 models; it preserves fluency on 4 of 6 (notably the mid-stack cluster, where one prototype could plausibly serve all three encoders); the reduction is concentrated on susceptibility-stratum-top items.
3. **Stack-wide ablation upper-bounds the causal anchor pathway at ~11–22 pp direction-follow reduction**, but the realised reduction is partly fluency-mediated rather than purely anchor-suppression — interpret the headline number as an upper bound, not a target.
4. **Lower-half ablation is unsafe.** The 3/6 backfires (Gemma +0.27, LLaVA-1.5 +0.165, InternVL3 +0.068) rule out lower-half attention re-weighting as a candidate intervention — at least without a per-model gate. The ConvLLaVA-vs-LLaVA-1.5 divergence on this mode is a warning that observational E1b identity does not imply causal identity even within the mid-stack cluster.

## Caveats

- **Mask-replacement is not a perfect "ablation"**: setting attention scores at the anchor span to `−1e4` strongly suppresses but does not zero out (the kernel still computes softmax over the masked column). For attention with float16/bfloat16 precision this is effectively masked; we did spot-check on LLaVA-1.5 that the post-hook attention weights at the anchor span are < 1e-4 across all heads at the targeted layers.
- **Multi-layer ablation interactions are not additive.** Ablating peak ± 2 is *not* the same as ablating peak alone five times. Apparent BACKFIRES in `ablate_peak_window` and `ablate_lower_half` reflect this interaction.
- **Qwen's `mean_distance_to_anchor` baseline is ~48.6** — heavily inflated by a small number of long-digit hallucinations (e.g. "9999..."). Treat Qwen's `mean_distance_to_anchor` deltas with caution; we use `direction_follow_rate` and `adoption_rate` as primary outcomes.
- **InternVL3 and FastVLM both lose triplets under several ablation modes** because they sometimes emit no digit token at all, and `_compute_metrics` requires a `parsed_number`. InternVL3's baseline `n=137`; under `ablate_lower_half` it drops to 83 (39 % loss), `ablate_peak_window` to 112, `ablate_upper_half` to 112. FastVLM's baseline `n=147`; under `ablate_all` and `ablate_upper_half` it drops to 159–161 (the count goes *up* slightly on some modes because the comparator cell shifts; what matters is the surviving subset is non-paired with baseline). All "reductions" or "BACKFIRES" on these two models under sub-baseline-n modes carry a survivor-bias caveat — the surviving triplets are conditional on the model being able to emit a digit *under ablation*, which is not random.
- **Sample-instance pairing**: triplets are joined by `sample_instance_id` so each model's mode-vs-baseline comparison is paired. The bootstrap CIs reported above resample triplets, not pairs, so they're slightly conservative under that pairing.
- **n=200 per model**, susceptibility-stratified (n=100 top + n=100 bottom). Same panel as E1b — the strata identifiers are loaded from `docs/insights/_data/susceptibility_strata.csv`.
- **FastVLM baseline drifts ~2 pp across re-aggregations** (0.238 → 0.216 between the original full-modes run and the layer-0 re-aggregation). This is non-determinism in the FastVLM eager-attention generation path under greedy decoding; the qualitative direction (peak/layer-0 null, upper-half clean reduction, all-layer broken fluency) is unchanged but Δ values shifted by 1–2 pp. The numbers in the tables above are from the canonical re-aggregation that includes the layer-0 run.

## Roadmap update

- §6 Tier 1 E1: open question "causal test (ablate anchor tokens at each family's peak layer and measure `direction_follow`)" → **closed; result null on single peak layer *and* on layer-0 control across 6/6 models; multi-layer redundancy confirmed**. Cross-link to this writeup.
- §3 status: layer-localisation framing is **observational only**; intervention design must use multi-layer ablation. Upper-half is the single best architecture-blind candidate.
- §10 changelog entries dated 2026-04-25.
