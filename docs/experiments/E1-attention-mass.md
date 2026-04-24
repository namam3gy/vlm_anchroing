# E1 — Attention-mass analysis: where does the anchor enter the computation?

**Status:** Plan only. Implementation queued behind E2 pilot. Source: `references/roadmap.md` §6 Tier 1 E1.

## Question

When the model output shifts toward the anchor digit, do the language-model layers actually *attend* to the anchor-image tokens more than to the target-image tokens? Or does the anchor's influence enter via some other route (e.g. a generic "second image" prior in the LLM)?

This is the cheapest mechanistic measurement available and directly addresses the "why does this happen" reviewer complaint flagged in `references/project.md`.

## Method

For a stratified 1k-sample subset (matching across `base_correct` strata so the H2 conditioning works):

1. Re-run inference on each `(model, sample)` pair under both `target_only` and `target_plus_irrelevant_number` conditions, with `output_attentions=True` enabled in `model.generate(...)`.
2. For each attention head, layer, and decoded token, compute attention mass to:
    - target-image tokens (image-1 patch positions)
    - anchor-image tokens (image-2 patch positions when present)
    - prompt text tokens
    - generated text tokens (causal masked → upper-triangular only)
3. Aggregate per layer / per condition / per (model, base_correct) stratum.

### Predictions tied to A1

Phase A established that the bias is *graded pull on uncertain items*. The corresponding attention prediction:

- On items where the model was originally **correct**, anchor-image attention mass should be ≈ neutral-image attention mass (both are noise distractors, both should be largely ignored).
- On items where the model was originally **wrong**, anchor-image attention mass should be **above** neutral-image attention mass — the LLM is recruiting the irrelevant image as evidence under uncertainty.

If this prediction holds, it gives the paper a tight mechanistic story: "uncertainty → broaden visual attention to both images → anchor digits get encoded into the answer-token decision".

### Predictions tied to A7

A7 showed item-level susceptibility correlates moderately across models. Therefore: high-susceptibility items (top-decile cross-model `moved_closer_rate`) should show systematically higher anchor-image attention mass than low-susceptibility items, and this should be visible *before* layer-wise patching — pure observational signal.

## Implementation outline

A new script `scripts/extract_attention_mass.py` that:

1. Loads one model via the existing `build_runner(...)` factory.
2. Adds a thin `_run_with_attention(...)` method that mirrors `generate_number()` but passes `output_attentions=True, return_dict_in_generate=True` and unpacks `out.attentions` → list[layer] of tensors `[batch, heads, q_len, k_len]`.
3. For each step's attention, identifies image-token spans via the processor's `image_token_index` markers (varies per model — `<image>` → repeated visual tokens). Where the runner does manual splicing (FastVLM, ConvLLaVA), reuse the splice points already known to the adapter.
4. Computes per-step mass into 4 buckets and writes a parquet per (model, sample_id, condition).

Memory note: full attention tensors at 1024-token context × 28 layers × 32 heads are ~50 MB per generation. Storing only per-bucket per-layer scalars drops this to ~10 KB per generation — comfortable.

## Compute budget

1k samples × 3 conditions × N models × few seconds per generation. On 4 models × 60 GB / each on the H200, this is **a single afternoon**, not a multi-day commitment. Will be scheduled after E2 full run finishes.

## Deliverable

- `scripts/extract_attention_mass.py`
- `outputs/attention_analysis/<model>/<run>/per_layer_mass.parquet`
- `docs/insights/E1-attention-results.md` with: the per-stratum attention bar plot, the per-layer trace, the cross-susceptibility comparison.

## Linkage to mitigation E4

If E1 shows anchor-image attention is the differentiator on uncertain items, the natural E4 prototype is **conditional attention re-weighting** — at inference time, when the LLM's per-token entropy on the answer position exceeds threshold τ, rescale anchor-image attention by α < 1. Two parameters, both tunable on a held-out split. This is the cheapest E4 implementation and falls out directly from E1 — they are designed to chain.
