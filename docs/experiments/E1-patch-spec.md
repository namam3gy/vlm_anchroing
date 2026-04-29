# E1-patch — Digit-pixel-restricted attention reanalysis (spec)

**Status:** spec written 2026-04-29; **executed 2026-04-29 (POC) +
extended same day to 4-model perfect-square panel**. Results doc:
`docs/insights/E1-patch-evidence.md` (4-model panel — gemma4-e4b /
llava-1.5-7b / convllava-7b / fastvlm-7b; peak `digit/anchor` 0.468–0.631,
+24 to +40 pp above fair share on every panel model). Two non-square
archetypes (internvl3-8b multi-tile, qwen2.5-vl-7b 17×23 grid) and the
masked-arm causal control are deferred to P3 in roadmap §6.5 — see the
"E1-patch non-square archetypes" and "E1-patch masked-arm causal
control" rows there. The body of this spec is preserved as the design
record; "Required work" §1 has landed in `scripts/extract_attention_mass.py`.

## Motivation

E1 / E1b / E1d aggregate attention over the **whole anchor image** span —
all visual patches between `<image>` tokens of the second image. The
question §7 of the paper needs to answer is whether the anchor's
attention concentrates on the **digit pixel patch** specifically, or
whether the model attends to the whole image roughly uniformly. User has
flagged this as: *"Attention Patch에 한정하는게 더 나을 듯"* (roadmap
§206 follow-up (c)).

E5c established that the digit pixel is the operative *causal* driver of
adoption (anchor − masked gap is positive on the wrong-base S1 cell). If
attention also concentrates on the digit-pixel patch above the anchor's
background patches, the mechanistic claim is tight: digit pixels both
*draw* attention and *cause* the behavioural pull.

## Why the existing dump is insufficient

`outputs/attention_analysis/<model>/<run>/per_step_attention.jsonl` records
attention sums only over four region buckets per layer per step:

```python
region_specs = [
    ("image_target",  span_start_t, span_end_t),
    ("image_anchor",  span_start_a, span_end_a),
    ("text",          ...),
    ("generated",     ...),
]
```

The full anchor image span (e.g. `[583, 1159]` on LLaVA-1.5 = 576 patches
at 24×24) is collapsed into a single number per (layer, step). Post-hoc
recovery of digit-vs-background separation is impossible from this dump
— the per-token attention values were never persisted.

## Required work

### 1. Modify `scripts/extract_attention_mass.py` to add digit-bbox region

Add two new regions:
```python
("image_anchor_digit",      span_start_a + bbox_token_offset_lo, span_start_a + bbox_token_offset_hi),
("image_anchor_background", complement of digit_token range within anchor span),
```

Bbox-to-token mapping per model:
- LLaVA-1.5 (CLIP-ViT-L/14, 336×336 → 24×24): bbox in pixels → patch grid
  index → token offset.
- Qwen2.5-VL: native-resolution patches; bbox-to-token mapping needs
  per-image processor query.
- Gemma3 / Gemma4 (SigLIP, 384×384 → variable): same.
- ConvLLaVA (1536×1536 → 144 tokens after pooling): bbox-to-token differs
  from ViT. Skip if mapping is ill-defined.
- InternVL3 (InternViT, 448×448 → 256 tokens + tile expansion):
  bbox-to-token + tile span needs care.
- FastVLM (FastViT, multi-scale + -200 marker expansion):
  bbox-to-token mapping unclear for multi-scale.

### 2. Bbox extraction per anchor image

For each `inputs/irrelevant_number/<value>.png` we have a paired
`inputs/irrelevant_number_masked/<value>.png` with the digit replaced by
content-preserving inpaint. The bbox is recoverable as the bounding box
of nonzero pixel diff between the two:

```python
bbox = bbox_of_diff(anchor_img, masked_img, threshold=8)
```

`generate_anchor_masked_images.py` likely already computes this — check
whether it logs bbox to a sidecar JSON. If not, add that and re-run mask
generation (cheap; CPU-only; it just re-runs OpenCV inpaint).

Cache bbox per anchor value in `inputs/irrelevant_number_bboxes.json`.

### 3. Re-extract attention with new regions

`scripts/extract_attention_mass.py --include-digit-region` walks the same
n=200 stratified subset as the existing E1 / E1b / E1d run, but emits
`image_anchor_digit` and `image_anchor_background` mass alongside the
existing four. Compute cost: same as the original extraction (~few hours
per model on H200 since inference cost is the dominant term).

### 4. Analysis

Re-run E1 / E1b / E1d-style analyses on the new dump. Headlines to test:

1. **Digit-pixel concentration ratio.** `digit / anchor_total` per layer.
   Expected: digit patch holds disproportionate mass relative to its area
   share (~1/24×24 ≈ 0.17 %).
2. **Per-archetype peak comparison.** Does the E1b "mid-stack cluster"
   (CLIP-ViT / InternViT / ConvNeXt) concentrate more sharply on the
   digit patch than Qwen-ViT or SigLIP-Gemma?
3. **Digit / background ratio under the upper-half ablation (E1d).**
   Does the upper-half ablation that reduces df 5.5-11.5 pp also reduce
   the digit-patch concentration ratio? (If yes — the locus story is
   tight: ablate the heads that look at the digit; if no — the
   df-reducing locus is not the digit-attention site, and the §7 claim
   needs to soften.)
4. **Anchor-arm vs. mask-arm digit-patch attention.** On the mask arm
   (`m`), the digit pixel is inpainted out. Compare digit-patch
   attention between `a` and `m` to test whether the "digit attention"
   really tracks the digit, or whether it's a pixel-position effect that
   would fire even on inpainted pixels.

### 5. Writeup

`docs/insights/E1-patch-evidence.md`. Tables per archetype × layer
(digit-fraction-of-anchor, digit-vs-background ratio, peak-layer
digit fraction). Scope-clean: skip models where bbox-to-token mapping
is uncertain (FastVLM, ConvLLaVA can be partial).

## Effort estimate

- Bbox extraction (CPU): 0.5 day.
- `extract_attention_mass.py` update: 0.5 day.
- Re-extract on 6 models × n=200: ~1 day GPU.
- Analysis + writeup: 0.5 day.

**Total: ~2.5 days, all P0-priority compute and engineering.**

## Decision points

- If §3.5 (anchor mask vs. anchor digit attention) shows digit patch
  has *no* concentration above background (rare null), the §7 paper
  claim shifts: attention is not localized on the digit, but the
  behaviour is — implication is the digit acts via earlier-stage
  encoder pathway, not via top-down attention. Updates required to §7
  narrative; would also fold the "image attribution rotation" angle
  (E1b 4-archetype) into §7.4 as an alternative locus story.
- If digit-patch concentration is robust on mid-stack cluster but not
  on Qwen / Gemma / FastVLM — confirms 4-archetype split with new
  dimension: "where the attention looks" maps onto "what kind of bias".
