# E4 — upper-half attention re-weighting reduces anchor pull on the mid-stack cluster

**Status:** Phase 1 sweep complete on all 3 mid-stack-cluster models
(llava-1.5-7b, convllava-7b, internvl3-8b). Phase 2 full-scale validation in flight
(2026-04-25; chained llava → convllava → internvl3). Source data:
`outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl`. Aggregate tables:
`outputs/e4_mitigation/_summary/{sweep_pareto.csv, chosen_strength.json}`. Full writeup:
`docs/experiments/E4-mitigation.md`.

## The claim and the test

E1d showed `ablate_upper_half` is the only multi-layer attention-mask intervention that
reduces `direction_follow_rate` on 6/6 panel models without breaking fluency on the
mid-stack cluster. E4 replaces hard masking with a soft strength axis (`exp(strength)`
multiplier on anchor attention), sweeps it on n=200 stratified samples, and picks the
smallest |strength| that achieves ≥ 10 % relative reduction in `direction_follow_rate`
with ≤ 2 pp drop in standard VQA accuracy.

## What we found (Phase 1, n=200 stratified per model)

| model | baseline df | s* | df at s* | em(num) at s=0 | em(num) at s* | em(target_only) invariant? |
|---|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 0.305 | **−3.0** | 0.265 (−13 %) | 0.365 | 0.370 (+0.5 pp) | ✓ (0.435) |
| convllava-7b | 0.290 | **−2.0** | 0.260 (−10 %) | 0.375 | 0.375 (+0.0 pp) | ✓ (0.500) |
| internvl3-8b | 0.161 | **−0.5** | 0.132 (−17.7 %) | 0.591 | 0.610 (+1.9 pp) | ✓ (0.568) |

**Headline:** the mitigation hits target on every mid-stack-cluster model at n=200. All
three meet ≥ 10 % `direction_follow_rate` reduction with em either flat or rising. Crucially,
`em(target_plus_irrelevant_number)` does *not* drop on any model — at saturation
(`s = −10⁴`) it rises +0.030 to +0.061. The mitigation appears safe for accuracy and mildly
beneficial. Phase 2 at full scale will tell us whether this holds with tight CIs.

**Anti-correlation surprise.** The mitigation effect is *anti-correlated* with baseline
anchor-pull, not proportional. InternVL3 has the lowest baseline df (0.161, the
"distraction-not-anchoring" model from H6) but the largest relative drop (−17.7 % at the
selected s*, −61 % at saturation). LLaVA-1.5 has the highest baseline df (0.305) but the
smallest relative drop (−13 % at s*, −18 % at saturation). The upper-half attention pathway
appears to carry a *larger fraction* of the anchor signal in the model that uses it less —
consistent with the anchor signal being narrowly concentrated in InternVL3 and broadly /
redundantly distributed across LLaVA-cluster layers. To revisit at Phase 2 scale.

## Per-condition sanity (llava-1.5-7b, n=200)

The strength hook attaches to the second-image token span; for `target_only` (no second
image) the hook is a no-op by construction, and for `target_plus_irrelevant_neutral` the
hook fires but the second image carries no legible digit. Phase 1 confirms:

- `em(target_only)`: **0.435 invariant** across all 7 strengths — hook does not leak into
  single-image inference.
- `em(target_plus_irrelevant_neutral)`: 0.355–0.365 (Δ ≤ 0.01) — the hook fires but doesn't
  change predictions when there's no anchor signal to remove.
- `em(target_plus_irrelevant_number)`: 0.365 → 0.395 (+3 pp) — the only condition where
  predictions move under the hook, and they move toward correct answers.

## A stronger framing — anchor damage and partial recovery

The em columns invite a sharper claim than "mitigation is safe for accuracy". Reading
the `em_target_only` column as the un-anchored ceiling:

| model | em(target_only) | em(num) at s=0 | anchor-induced em loss | em(num) at saturation s=−10⁴ | recovery from saturation |
|---|---:|---:|---:|---:|---:|
| llava-1.5-7b | 0.435 | 0.365 | −0.070 (16 % rel) | 0.395 | +0.030 (recovers 43 % of loss) |
| convllava-7b | 0.500 | 0.375 | −0.125 (25 % rel) | 0.405 | +0.030 (recovers 24 % of loss) |
| internvl3-8b | 0.568 | 0.591 | +0.023 (no damage) | 0.652 | +0.061 (anchor-condition em rises further) |

InternVL3 reads differently: em(num) ≥ em(target_only) at s=0 — the anchor doesn't damage
its em on this stratified set. (Caveat: the em columns are computed on different surviving
sample subsets because of InternVL3's parse-leak, n=137 vs n=200; comparing absolute em
levels across conditions is approximate. The strength-axis monotonicity within
target_plus_irrelevant_number is the load-bearing signal.) The anchor-damage / partial-
recovery framing therefore holds cleanly on the LLaVA-cluster (LLaVA-1.5, ConvLLaVA): the
anchor demonstrably damages accuracy, and upper-half attention re-weighting recovers a
non-trivial slice. On InternVL3, the same intervention raises em further regardless — the
upper-half locus is doing useful work even on a model where the anchor damage signal is
lower-bounded by 0.

At the chosen working point `s*`, the recovery is smaller (em delta ≈ 0 pp on LLaVA /
ConvLLaVA, +1.9 pp on InternVL3); the visible recovery is at saturation. Phase 2 will tell
us whether (a) `s*` is undersized and the working point should sit deeper into the
strength axis, or (b) saturation is over-aggressive and the modest df reduction at `s*` is
the right operating point.

## Why this matters

E1d closed one open question (the per-layer attention peak from E1b is correlational; the
causal pathway is multi-layer) and opened the mitigation question. E4 closes that one, on
this model at this scale. Two potential paper-level claims if Phase 2 holds:

1. **A single architecture-blind intervention site (upper-half LLM layers) reduces
   cross-modal anchoring on the mid-stack cluster of VLMs.** Mid-stack cluster (LLaVA-1.5,
   ConvLLaVA, InternVL3) shares the same E1b peak/mechanism *and* a fluency-clean upper-
   half-ablation response (E1d).
2. **The mitigation does not trade off accuracy.** Where E1d's hard mask flagged fluency
   risks at full ablation, E4's soft strength regime sits at a working point where direction-
   follow drops while exact-match holds steady or rises.

## Caveats

- **n=200 CIs are wide.** Bootstrap CIs at sweep scale overlap heavily across strengths.
  The strength-axis monotonicity is informative; the per-strength deltas are not yet
  load-bearing. Phase 2 at n=17,730 carries the headline numbers.
- **InternVL3 prose-leak parse loss.** ~30 % of records drop out of valid-triplet count
  due to InternVL3 emitting prose tokens that the parser mis-classifies as numeric strings.
  Per-strength deltas are still legitimate within-model comparisons (baseline and treated
  share the same noise), but absolute em levels across conditions in the anchor-damage
  table are approximate. Driver fix (regex parse-rescue in
  `scripts/e4_attention_reweighting.py`) is logged for Phase 2.
- **ConvLLaVA causal-structure caveat (E1d).** Same-attention-signature ≠ same-causal-
  structure. ConvLLaVA's lower-half-ablation behaviour is the *opposite* of LLaVA-1.5's.
  If Phase 2 shows ConvLLaVA's E4 response diverges from LLaVA-1.5/InternVL3, demote it
  to a discussion caveat.
- **Selection at n=200 may shift at n=17,730.** Phase 2 may pick a different `s*` per
  model. Phase 2 reports both the Phase 1-chosen `s*` and the full bootstrap distribution.
- **Per-model `s*`, not shared.** `s*` ranges from −0.5 (InternVL3) to −3.0 (LLaVA-1.5);
  a single shared strength would over-mitigate one and under-mitigate the others. The
  mitigation generalises *as a locus + selection rule*, not a single strength constant.

## Open follow-ups (post Phase 2)

- Whether the anti-correlation between baseline anchor-pull and mitigation effect-size
  (InternVL3 = lowest baseline / largest mitigation; LLaVA-1.5 = highest baseline / smallest
  mitigation) holds at full scale. If yes, this is a paper-grade finding about how the
  upper-half locus shares the anchor signal across the mid-stack cluster.
- Whether the accuracy gain at saturation is anchor-specific (rules out generic accuracy
  bump from down-weighting any second-image attention).
- If the mitigation works, whether it generalises beyond VQAv2-number to ChartQA/TallyQA
  (E5).
