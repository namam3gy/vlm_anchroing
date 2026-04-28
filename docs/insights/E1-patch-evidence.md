# E1-patch — Digit-pixel attention concentration (POC)

**Status:** Generated 2026-04-29 from
`scripts/analyze_attention_patch.py`. Source: bbox-enabled attention
extraction via `scripts/extract_attention_mass.py --bbox-file
inputs/irrelevant_number_bboxes.json` on 2 representative models
(llava-1.5-7b, gemma4-e4b). Bbox JSON: 128 anchor digits, computed by
diffing anchor PNG vs masked PNG (`scripts/compute_anchor_digit_bboxes.py`).

This is a POC. Both POC models pass the headline test (digit-pixel
attention concentration far above fair share); analysis extends to
the full 6-model E1 panel as a clear next step. The masked-arm causal
control is deferred (current `configs/experiment.yaml` is the legacy
3-condition pipeline without the masked arm — addressed in §6 below).

## TL;DR

> **Both POC models concentrate attention sharply on the digit-pixel patch
> of the anchor image — digit-fraction-of-anchor-mass exceeds the bbox
> area share by +24 to +40 pp at the answer step.** §7 of the paper can
> claim *"anchor attention is anchored on digit pixels, not on the
> anchor image as a whole"* with confidence on the SigLIP-Gemma and
> CLIP-ViT mid-stack-cluster archetypes.

| model | encoder archetype | digit-fraction peak L | digit/anchor at peak | fair-share baseline | concentration above fair |
|---|---|---:|---:|---:|---:|
| **gemma4-e4b** | SigLIP early (E1b L5) | **L9 / 42** | **0.631** | ~0.227 | **+0.404** |
| **llava-1.5-7b** | CLIP-ViT mid-stack (E1b L16) | **L7 / 32** | **0.468** | ~0.227 | **+0.241** |

Two qualitatively different layer profiles (see §3) — Gemma is
*globally* digit-concentrated (every layer 0.5-0.6), while LLaVA-1.5
shows a *peaked* profile (rises to ~0.46 around L7-L10, then falls
back below fair share by L30).

## 1. Setup

### Models picked for POC

| model | encoder | E1b archetype | E1d locus | E4 mitigation effect |
|---|---|---|---|---|
| `llava-1.5-7b` | CLIP-ViT-L/14 (336×336) | mid-stack cluster, peak L16 | upper-half multi-layer | df −17.7 % rel (Phase 2 full) |
| `gemma4-e4b` | SigLIP-So (256-token / 16×16 grid) | SigLIP early peak L5 | text-stealing | n/a (different archetype, no E4) |

Bbox-to-token mapping uses normalized pixel coords mapped to row-major
patch indices, processor-agnostic. Both models produce perfect-square
anchor spans:
- llava-1.5-7b anchor span = 576 tokens = 24 × 24 grid (CLIP 336/14).
- gemma4-e4b anchor span = 256 tokens = 16 × 16 grid (SigLIP-So 224/14).

Multi-tile / multi-scale models (InternVL3, FastVLM) and ConvLLaVA's
ConvNeXt span are skipped pending per-encoder bbox-mapping logic.

### Inputs and wall

- Susceptibility-stratified n=400 samples per model (200 top + 200
  bottom decile, same as existing E1 dump).
- 3 conditions per sample under `configs/experiment.yaml` (legacy
  3-cond pipeline): target_only, anchor (a), neutral (d). The
  masked arm (m) is **not** present in this dump — see §6.
- Wall: llava-1.5-7b ~12 min, gemma4-e4b ~13 min on H200 with
  `output_attentions=True`. Total ~25 min for both.

### Output schema additions

Each `per_step` record on the anchor arm now carries:
- `image_anchor_digit`: list (n_layers) — attention summed over the
  digit-bbox token positions inside the anchor span.
- `image_anchor_background`: list (n_layers) — `image_anchor − image_anchor_digit`.

The target_only arm has no anchor span, so these fields are absent. The
neutral arm has no anchor value, so the bbox lookup returns None and the
fields are absent.

## 2. Concentration above fair share

Digit-fraction-of-anchor-mass at the answer step, paired against
the bbox area share (= `bbox_area / image_area` per anchor value, mean
across the n=400 samples).

| model | layer | digit/anchor (mean, n=400) | fair share | concentration above fair |
|---|---:|---:|---:|---:|
| gemma4-e4b | 0 | 0.606 | ≈ 0.227 | +0.379 |
| gemma4-e4b | 5 (E1b peak) | 0.567 | ≈ 0.227 | +0.340 |
| gemma4-e4b | 9 (this peak) | **0.631** | ≈ 0.227 | **+0.404** |
| gemma4-e4b | 21 | 0.610 | ≈ 0.227 | +0.384 |
| gemma4-e4b | 41 (last) | 0.527 | ≈ 0.227 | +0.301 |
| llava-1.5-7b | 0 | 0.254 | ≈ 0.227 | +0.027 |
| llava-1.5-7b | 7 (this peak) | **0.468** | ≈ 0.227 | **+0.241** |
| llava-1.5-7b | 10 | 0.406 | ≈ 0.227 | +0.179 |
| llava-1.5-7b | 16 (E1b peak) | 0.321 | ≈ 0.227 | +0.094 |
| llava-1.5-7b | 31 (last) | 0.212 | ≈ 0.227 | −0.015 |

Reading: with uniform attention across the anchor image, the digit
patch (which covers ~22.7 % of the image area on average) would receive
~22.7 % of the attention. Both models show *much higher* digit-fraction
than that — the model preferentially looks at the digit.

## 3. Per-layer profile differences

| layer band | gemma4-e4b digit/anchor | llava-1.5-7b digit/anchor |
|---|---:|---:|
| 0-5 (early) | 0.55 - 0.61 | 0.23 - 0.32 |
| 6-15 (mid-early) | 0.56 - 0.61 | 0.40 - 0.47 |
| 16-25 (mid) | 0.55 - 0.61 | 0.24 - 0.32 |
| 26-end (late) | 0.50 - 0.55 | 0.21 - 0.27 |

**gemma4-e4b**: digit concentration is high at every layer — SigLIP-Gemma
"sees" the digit globally. Consistent with E1b's report of early-layer
typographic-like inheritance (L5 peak on total mass) extending more
broadly across depth on the digit-fraction axis.

**llava-1.5-7b**: digit concentration is *peaked* at L6-L11 (0.40-0.47),
falls back near fair-share by L20 and below by L30. The mid-stack
(L8-L11) is where digit attention is concentrated — slightly *earlier*
than the total-anchor-mass peak at L16 from E1b. Plausible reading:
mid-early layers focus on the digit specifically; mid-late layers
spread attention back across the anchor (perhaps integrating the
digit's value into the answer prediction).

The two archetypes therefore contribute to anchoring through *different
attention pathways*: Gemma's global digit-concentration aligns with the
"typographic-attack inheritance" framing (the SigLIP encoder's
typographic feature is preserved through the LLM stack); LLaVA-1.5's
peaked mid-stack profile aligns with the "mid-stack text-stealing"
cluster from E1b (the layers where E4 mitigation operates).

## 4. Implications for §7 of the paper

The §7.1 sub-claim "anchor attention is anchored on digit pixels, not
on the anchor image as a whole" promotes from observation to structural
claim on both archetypes.

**Per-layer profile difference is a §7.2 sub-finding worth keeping**:
Gemma's globally-digit-concentrated SigLIP and LLaVA's peaked-mid-stack
CLIP profile are two distinct mechanisms of digit-driven anchoring.
This refines the E1b 4-archetype split with an additional axis:
"globally vs. locally digit-attention-concentrated".

**Mitigation locus reinterpretation**: E4's upper-half multi-layer
ablation (L16-L31 on llava-1.5-7b) on the mid-stack cluster reduces
df 17.7 %. From E1-patch, the digit-concentration peak on llava-1.5-7b
is at L7-L11 (the *lower half*). So E4's success isn't because it kills
the digit-attention site — it operates on a different, broader pathway
that includes the digit-aware layers (L7-L11) but extends across more
of the stack. Worth a §7.4 nuance: "mitigation locus and digit-attention
concentration locus are correlated but not identical".

## 5. Where E1-patch sits relative to E1 / E1b / E1d

| metric | E1 / E1b | E1-patch |
|---|---|---|
| measure | total `image_anchor` mass | `image_anchor_digit / image_anchor` ratio |
| llava-1.5-7b peak layer | L16 | L7 |
| gemma4-e4b peak layer | L5 | L9 (broad plateau) |
| addresses | "where in the stack does the anchor get attention" | "where in the anchor does the attention go" |

The two metrics are complementary, not redundant. E1-patch confirms
that *within the anchor*, attention concentrates on the digit; E1b
locates *which layers* this concentration peaks. They peak at
different layers — interesting, not contradictory.

## 6. Caveats (POC scope)

- **Masked-arm causal control is not yet computed.** The current
  `configs/experiment.yaml` is the legacy 3-condition pipeline without
  the `masked` extra. The masked control would test "is digit attention
  really tracking the digit, or is it a position-specific feature?" To
  add: extend `configs/experiment.yaml` with `stratified_extras: ["masked"]`
  (or use a new config), re-run extraction. The fair-share baseline test
  in §2 already rules out the trivial "uniform attention coincidentally
  hits the bbox" reading; the masked-arm control is a strictly stronger
  test that the E1-patch expansion to 6 models should include.
- **Only 2 of 4 archetypes covered** (mid-stack cluster + SigLIP-early).
  Qwen-ViT late and FastVLM late text-stealing not tested yet.
- **ConvLLaVA / InternVL3 / FastVLM** all have non-square or non-uniform
  patch grids. Per-encoder bbox mapping needed for full-panel coverage.
- **Bbox extracted from anchor − masked diff.** Robust on simple digits
  but may include some inpaint-artefact pixels at the edges. Threshold = 8
  on per-pixel sum-abs-diff is conservative.
- **Attention here uses the answer step** (= first generated digit token);
  earlier steps (image processing, prompt parsing) have different
  signatures. Step-0 vs answer-step comparison deferred.

## 7. Next steps

| ID | Action | Effort |
|---|---|---|
| **E1-patch-A** | Extend `configs/experiment.yaml` with `masked` extra; re-run extraction on llava-1.5-7b + gemma4-e4b; add `a − m` digit-mass gap table | ~0.5 day (compute) |
| **E1-patch-B** | Extend bbox-to-token mapping to ConvLLaVA / InternVL3 / FastVLM; re-run on those + qwen2.5-vl-7b; complete 6-model panel | ~1 day (engineering + compute) |
| **E1-patch-C** | Step-0 vs answer-step comparison on the bbox dump | analysis only |
| **E1-patch-D** | Re-run E1d (causal ablation) with bbox-enabled extraction; report ablation effect on `image_anchor_digit` specifically vs `image_anchor_background`. Tests whether the upper-half ablation in E1d kills digit attention or background attention. | ~0.5 day (compute + analysis) |

## 8. Headline number to quote in the paper

For §7.1 (mechanism): "On both POC models, the digit pixel patch (~22 %
of the anchor image area) receives more than 46 % of the attention
mass to the anchor at the model's digit-attention peak layer
(llava-1.5-7b L7: 47 %; gemma4-e4b L9: 63 %), exceeding the
fair-share baseline by +24 to +40 pp."
