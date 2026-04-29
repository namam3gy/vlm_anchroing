# E1-patch — Digit-pixel attention concentration (4-model panel)

**Status:** POC (2026-04-29) extended to a **4-model panel** the same
day on the perfect-square subset of the E1 mechanism panel. Source:
bbox-enabled attention extraction via `scripts/extract_attention_mass.py
--bbox-file inputs/irrelevant_number_bboxes.json` on
gemma4-e4b, llava-1.5-7b, **convllava-7b, fastvlm-7b**. Bbox JSON: 128
anchor digits, computed by diffing anchor PNG vs masked PNG
(`scripts/compute_anchor_digit_bboxes.py`). Aggregate via
`scripts/analyze_attention_patch.py --print-summary`. Outputs:
`docs/insights/_data/E1_patch_per_layer.csv`,
`docs/insights/_data/E1_patch_concentration_per_layer.csv`.

The panel covers all four E1b encoder archetypes whose anchor span is
a perfect square:

- **SigLIP-Gemma early** (gemma4-e4b, 16×16 = 256 tokens)
- **CLIP-ViT mid-stack** (llava-1.5-7b, 24×24 = 576 tokens)
- **ConvNeXt mid-stack** (convllava-7b, 24×24 = 576 tokens)
- **FastViT late text-stealing** (fastvlm-7b, 16×16 = 256 tokens)

Two archetypes have non-perfect-square anchor spans and remain deferred
pending per-encoder bbox-to-token mapping logic:

- **InternViT multi-tile** (internvl3-8b, 3328 tokens, multi-tile)
- **Qwen-ViT non-square** (qwen2.5-vl-7b, 391 tokens = 17×23)

Each requires its own routing through the existing
`int(math.isqrt(n)) ** 2 == n` gate in `_compute_anchor_bbox_mass`;
roadmap §6.5 carries the deferred-task entry. The masked-arm causal
control is also deferred (current `configs/experiment.yaml` is the
3-condition pipeline without the masked arm) — see §6 below.

## TL;DR

> **4 of 4 perfect-square panel models concentrate attention sharply
> on the digit-pixel patch of the anchor image — digit-fraction-of-
> anchor-mass exceeds the bbox area share by +24 to +40 pp at each
> model's peak layer.** §7 of the paper can claim *"anchor attention
> is anchored on digit pixels, not on the anchor image as a whole"*
> with confidence on the SigLIP-Gemma, CLIP-ViT mid-stack,
> CLIP-ConvNeXt mid-stack, and FastViT late archetypes.

| model | encoder archetype | digit-fraction peak L | peak depth | digit/anchor at peak | fair share | concentration above fair |
|---|---|---:|---:|---:|---:|---:|
| **gemma4-e4b** | SigLIP early (E1b L5/42) | **L9 / 42** | 21 % | **0.631** | ~0.227 | **+0.404** |
| **convllava-7b** | CLIP-ConvNeXt mid-stack (E1b L16/32) | **L7 / 32** | 22 % | **0.552** | ~0.227 | **+0.325** |
| **fastvlm-7b** | FastViT late text-stealing (E1b L22) | **L4 / 28** | 14 % | **0.531** | ~0.227 | **+0.304** |
| **llava-1.5-7b** | CLIP-ViT mid-stack (E1b L16/32) | **L7 / 32** | 22 % | **0.468** | ~0.227 | **+0.241** |

Three qualitative profile shapes (see §3):

1. **Globally digit-concentrated** (gemma4-e4b) — every layer 0.50-0.63;
   SigLIP "sees" the digit globally across the LLM stack.
2. **Peaked mid-early then decay** (llava-1.5-7b, convllava-7b) —
   sharp rise to 0.47-0.55 around L7, falls back to fair share by
   L15-L17 and sub-fair by L29-L31. Same shape on two architecturally
   distinct mid-stack-cluster encoders.
3. **Sharp early peak with sustained mid-stack plateau** (fastvlm-7b)
   — peaks at L4 (0.53), drops to ~0.32 at L6, and maintains a 0.35-0.45
   plateau through L12; falls to fair share by L18.

## 1. Setup

### Models in the perfect-square panel

| model | encoder | E1b archetype | E1d locus | E4 mitigation effect | anchor span |
|---|---|---|---|---|---|
| `gemma4-e4b` | SigLIP-So-256 (16×16 grid) | SigLIP early peak L5 | text-stealing | n/a (different archetype, no E4) | 256 = 16² |
| `llava-1.5-7b` | CLIP-ViT-L/14 (336×336) | mid-stack cluster, peak L16 | upper-half multi-layer | df −14.6 % rel (Phase 2 full) | 576 = 24² |
| `convllava-7b` | CLIP-ConvNeXt | mid-stack cluster, peak L16 | upper-half multi-layer | df −9.6 % rel (Phase 2 full) | 576 = 24² |
| `fastvlm-7b` | FastViT (16×16 grid) | FastViT late, peak L22 | text-stealing | n/a (different archetype, no E4) | 256 = 16² |

Bbox-to-token mapping uses normalized pixel coords mapped to row-major
patch indices, processor-agnostic. `_compute_anchor_bbox_mass` operates
on perfect-square spans only:

```python
grid = int(math.isqrt(n_tokens))
if grid * grid != n_tokens or grid <= 0:
    return None
```

InternVL3's multi-tile span (3328 tokens) and Qwen2.5-VL's
non-square span (391 = 17×23) hit this gate and produce no
`image_anchor_digit` / `image_anchor_background` fields. They are
listed in roadmap §6.5 as "E1-patch non-square archetypes" P3.

### Inputs and wall

- Susceptibility-stratified n=400 samples per model (200 top + 200
  bottom decile, same as existing E1 dump).
- 3 conditions per sample under `configs/experiment.yaml` (legacy
  3-cond pipeline): target_only, anchor (a), neutral (d). The
  masked arm (m) is **not** present in this dump — see §6.
- Wall: gemma4-e4b ~13 min, llava-1.5-7b ~12 min, **convllava-7b
  ~5 min, fastvlm-7b ~18 min** on H200 with `output_attentions=True`.
  Total panel runtime ~50 min cumulative.

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
| convllava-7b | 0 | 0.274 | ≈ 0.227 | +0.048 |
| convllava-7b | 6 | 0.487 | ≈ 0.227 | +0.260 |
| convllava-7b | 7 (this peak) | **0.552** | ≈ 0.227 | **+0.325** |
| convllava-7b | 9 | 0.466 | ≈ 0.227 | +0.239 |
| convllava-7b | 16 (E1b peak) | 0.256 | ≈ 0.227 | +0.030 |
| convllava-7b | 31 (last) | 0.122 | ≈ 0.227 | −0.104 |
| fastvlm-7b | 0 | 0.284 | ≈ 0.227 | +0.057 |
| fastvlm-7b | 4 (this peak) | **0.531** | ≈ 0.227 | **+0.304** |
| fastvlm-7b | 8 | 0.445 | ≈ 0.227 | +0.219 |
| fastvlm-7b | 12 | 0.359 | ≈ 0.227 | +0.132 |
| fastvlm-7b | 22 (E1b peak) | ≈ 0.30 | ≈ 0.227 | ≈ +0.07 |
| fastvlm-7b | 27 (last) | 0.281 | ≈ 0.227 | +0.054 |

Reading: with uniform attention across the anchor image, the digit
patch (which covers ~22.7 % of the image area on average) would receive
~22.7 % of the attention. **All four models show much higher
digit-fraction than that** — every panel model preferentially looks at
the digit, and the per-layer peak exceeds fair share by +24 to +40 pp
on every model.

## 3. Per-layer profile differences

| layer band (relative depth) | gemma4-e4b digit/anchor | llava-1.5-7b digit/anchor | convllava-7b digit/anchor | fastvlm-7b digit/anchor |
|---|---:|---:|---:|---:|
| 0-15 % (very early) | 0.55 - 0.61 | 0.23 - 0.32 | 0.27 - 0.32 | 0.28 - 0.38 |
| 16-40 % (mid-early) | 0.56 - 0.63 | **0.40 - 0.47** | **0.35 - 0.55** | **0.45 - 0.53** |
| 41-65 % (mid) | 0.55 - 0.61 | 0.24 - 0.32 | 0.25 - 0.35 | 0.30 - 0.45 |
| 66-100 % (late) | 0.50 - 0.55 | 0.21 - 0.27 | 0.12 - 0.29 | 0.27 - 0.34 |

**Three qualitative profile shapes** emerge across the panel:

**(A) Globally digit-concentrated — `gemma4-e4b`.** SigLIP-Gemma
"sees" the digit at every layer (0.50-0.63 across the 42-layer stack).
Consistent with E1b's report of early-layer typographic-like
inheritance (L5 peak on total mass) extending more broadly across
depth on the digit-fraction axis.

**(B) Peaked mid-early then decay — `llava-1.5-7b` and
`convllava-7b`.** Two architecturally distinct mid-stack-cluster
models (CLIP-ViT vs CLIP-ConvNeXt) converge on the same profile: a
sharp digit-concentration peak at L7 (0.47, 0.55) that decays to
fair share by L15-L17 and below by L29-L31. Notable: convllava's
peak (0.552) is *higher* than llava-1.5's (0.468) at the same
layer L7 — an internal architecture comparison favouring ConvNeXt's
spatial encoding for digit attention.

**(C) Sharp early peak + sustained mid-stack plateau — `fastvlm-7b`.**
FastViT shows a sharp peak at L4 (0.531) that drops to ~0.32 at L6,
then maintains a 0.35-0.45 plateau through L12 before falling to fair
share by L18. The sustained mid-stack plateau is qualitatively
different from llava-1.5's clean fall-off pattern.

The three profile shapes contribute to anchoring through *different
attention pathways*: Gemma's global digit-concentration aligns with
the "typographic-attack inheritance" framing (the SigLIP encoder's
typographic feature is preserved through the LLM stack); the
mid-stack-cluster's peaked mid-early profile aligns with the
"text-stealing" cluster from E1b (the layers where E4 mitigation
operates); FastViT's sharp early peak with sustained plateau
suggests a hybrid pattern that doesn't fit either framing cleanly.

## 4. Implications for §7 of the paper

The §7.1 sub-claim "anchor attention is anchored on digit pixels, not
on the anchor image as a whole" promotes from observation to
**panel-wide structural claim**: 4 of 4 perfect-square archetypes
exhibit it.

**Profile-shape difference is a §7.2 sub-finding worth keeping**:
the three shapes (globally digit-concentrated SigLIP / peaked
mid-stack-cluster / FastViT early-and-sustained) refine the E1b
4-archetype split with a "globally vs. locally vs. plateau"
digit-attention axis.

**Mitigation locus reinterpretation (extended)**: E4's upper-half
multi-layer ablation reduces df 9.6-14.6 % on llava-1.5 and
convllava-7b. From E1-patch, both models' digit-concentration peak is
at L7 (the *lower half*). E4's success therefore isn't because it
kills the digit-attention site — it operates on a different, broader
pathway that *includes* the digit-aware lower-half layers but extends
across the upper half. Worth a §7.4 nuance: "mitigation locus and
digit-attention concentration locus are correlated but not identical;
the locus the mitigation acts on is broader than the digit-attention
peak, and includes layers where the digit signal has already dissipated
back to fair share."

## 5. Where E1-patch sits relative to E1 / E1b / E1d

| metric | E1 / E1b | E1-patch |
|---|---|---|
| measure | total `image_anchor` mass | `image_anchor_digit / image_anchor` ratio |
| llava-1.5-7b peak layer | L16 | L7 |
| convllava-7b peak layer | L16 | L7 |
| gemma4-e4b peak layer | L5 | L9 (broad plateau) |
| fastvlm-7b peak layer | L22 | L4 (sharp early peak) |
| addresses | "where in the stack does the anchor get attention" | "where in the anchor does the attention go" |

The two metrics are complementary, not redundant. E1-patch confirms
that *within the anchor*, attention concentrates on the digit; E1b
locates *which layers* this concentration peaks. They peak at
different layers on every panel model — a consistent pattern
where digit-attention is concentrated **earlier in the LLM stack**
than total-anchor mass on 3 of 4 models (gemma being the
globally-concentrated exception).

## 6. Caveats (panel scope)

- **Masked-arm causal control is not yet computed.** The current
  `configs/experiment.yaml` is the legacy 3-condition pipeline without
  the `masked` extra. The masked control would test "is digit attention
  really tracking the digit, or is it a position-specific feature?" To
  add: extend `configs/experiment.yaml` with `stratified_extras: ["masked"]`
  (or use a new config), re-run extraction. The fair-share baseline test
  in §2 already rules out the trivial "uniform attention coincidentally
  hits the bbox" reading; the masked-arm control is a strictly stronger
  test that the future panel extension should include.
- **4 of 6 archetypes covered** (SigLIP-early + 2 mid-stack-cluster +
  FastViT-late). The two non-square archetypes — InternViT multi-tile
  (internvl3-8b) and Qwen-ViT (qwen2.5-vl-7b, 17×23 non-square span) —
  remain deferred pending per-encoder bbox-mapping logic. Roadmap
  §6.5 carries the entry.
- **Bbox extracted from anchor − masked diff.** Robust on simple digits
  but may include some inpaint-artefact pixels at the edges. Threshold = 8
  on per-pixel sum-abs-diff is conservative.
- **Attention here uses the answer step** (= first generated digit token);
  earlier steps (image processing, prompt parsing) have different
  signatures. Step-0 vs answer-step comparison deferred.

## 7. Next steps

| ID | Action | Effort |
|---|---|---|
| **E1-patch-A** | Extend `configs/experiment.yaml` with `masked` extra; re-run extraction on the 4-model panel; add `a − m` digit-mass gap table | ~1h GPU/model + 4-cond config wiring |
| **E1-patch-B** | Per-encoder bbox-to-token mapping for InternVL3-8b (multi-tile) and Qwen2.5-VL-7b (17×23 grid via `grid_thw`); re-run on those two; complete 6-model panel | ~1-2 days/model implementation + ~12 min/model GPU |
| **E1-patch-C** | Step-0 vs answer-step comparison on the 4-model bbox dump | analysis only |
| **E1-patch-D** | Re-run E1d (causal ablation) with bbox-enabled extraction; report ablation effect on `image_anchor_digit` specifically vs `image_anchor_background`. Tests whether the upper-half ablation in E1d kills digit attention or background attention. | ~0.5 day (compute + analysis) |

## 8. Headline number to quote in the paper

For §7.2 (per-layer mechanism, panel-wide claim): "On 4 of 4
perfect-square panel models — spanning SigLIP, CLIP-ViT, CLIP-ConvNeXt,
and FastViT vision encoders — the digit pixel patch (~22.7 % of the
anchor image area) receives 47-63 % of the attention mass to the
anchor at the model's digit-attention peak layer (gemma4-e4b L9: 63 %;
convllava-7b L7: 55 %; fastvlm-7b L4: 53 %; llava-1.5-7b L7: 47 %),
exceeding the fair-share baseline by +24 to +40 pp."
