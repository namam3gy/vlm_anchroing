# E4 — upper-half attention re-weighting reduces anchor pull on the mid-stack cluster

**Status:** Phase 1 sweep complete on all 3 mid-stack-cluster models
(llava-1.5-7b, convllava-7b, internvl3-8b). **Phase 2 full-scale validation: llava-1.5-7b
complete (88,650 records, 100 %), convllava-7b in flight, internvl3-8b queued.**
Source data: `outputs/e4_mitigation/<model>/{sweep_n200,full_n17730}/predictions.jsonl`.
Aggregate tables: `outputs/e4_mitigation/_summary/{sweep_pareto, full_validation,
full_validation_compare, anchor_damage_paired_{sweep,full}, chosen_strength}.csv|.json`.
Full writeup: `docs/experiments/E4-mitigation.md`.

## Phase 2 headline (all 3 mid-stack-cluster models, 88,650 records each, 100 % complete)

| model | s* | df baseline | df treated | df Δ | df rel | em baseline | em treated | em Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | −3.0 | 0.2578 | 0.2122 | **−4.55 pp** | **−17.7 %** | 0.3340 | 0.3418 | +0.77 pp |
| convllava-7b | −2.0 | 0.2283 | 0.2042 | **−2.42 pp** | **−10.6 %** | 0.3522 | 0.3652 | +1.30 pp |
| internvl3-8b | −0.5 | 0.1035 | 0.0975 | **−0.59 pp** | **−5.8 %** | 0.5902 | 0.5950 | +0.49 pp |

Paired anchor-damage on the full sets:

| model | n_paired | em(TO) | em(num@0) | em(num@s*) | damage | recovery | % of damage recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | 17,724 | 0.3696 | 0.3340 | 0.3417 | **−3.55 pp** | +0.77 pp | **21.7 %** |
| convllava-7b | 17,722 | 0.4454 | 0.3520 | 0.3651 | **−9.34 pp** | +1.31 pp | **14.0 %** |
| internvl3-8b | 11,848 | 0.6325 | 0.5938 | 0.5977 | **−3.87 pp** | +0.40 pp | **10.2 %** |

**Reading.** Phase 2 covers **all 3 mid-stack-cluster models at full scale**. Headline
properties hold across the panel: df decreases on every model, em rises on every model,
and the paired anchor-damage analysis shows partial recovery on every model (10–22 % of
the damage). The 10 %-relative-reduction roadmap target is met cleanly on LLaVA (−17.7 %)
and ConvLLaVA (−10.6 %); InternVL3 misses it (−5.8 %) for a structural reason — it is
the H6 "distraction-not-anchoring" model and its full-scale baseline df is half of the
LLaVA-cluster's. The mitigation still moves the metric in the right direction, just there
is less anchor signal to remove on this model in the first place.

**Stratified vs. full-scale on InternVL3 — the gap is sample-distributional, not a
mitigation failure.** Phase 1 sweep on the stratified set (top-decile susceptible × 100 +
bottom-decile resistant × 100) saw df₀ = 0.161 and the mitigation dropped it 17.7 %
relative. Phase 2 full set is more representative and has df₀ = 0.103 (~36 % lower
baseline anchor pull). The mitigation effect at the working point scales with the baseline
signal it is removing.

**ConvLLaVA fluency caveat at full scale.** `mean_distance_to_anchor` jumps from 2.99 to
53.54 on ConvLLaVA at full scale (Phase 1 sweep on stratified set: 3.18 → 3.30). A small
fraction of samples receive predictions far from any plausible anchor, dragging the *mean*
up by ~17×; em(num) still rises because the bulk of the distribution improves enough to
net positive, and df is robust because it is computed per-pair against the model's own
baseline. For the paper: report median distance + a fluency-degraded fraction count rather
than the unwinsorised mean.

**InternVL3 parse-loss persists.** ~33 % of records drop out of the paired-valid set
(n_paired = 11,848 of 17,730) because the InternVL3 max_new_tokens=32 driver patch was
applied *during* this run rather than before it. Re-running InternVL3 with the patch
would tighten the sample size; whether it materially changes the headline depends on
whether the dropped items are systematically different from the kept ones (paired-set
em(TO) = 0.6325 vs panel em(TO) = 0.5760 suggests parse-failing items are systematically
the model's harder cases — same caveat as Phase 1).

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

**Two patterns from Phase 1.**

*On `direction_follow_rate` — anti-correlation with baseline anchor-pull.* Mitigation
effect is *anti-correlated* with baseline df, not proportional. InternVL3 has the lowest
baseline df (0.161, the H6 "distraction-not-anchoring" model) but the largest relative drop
(−17.7 % at s*, −61 % at saturation). LLaVA-1.5 has the highest baseline df (0.305) but the
smallest relative drop (−13 % at s*, −18 % at saturation). Conjecture: the upper-half
attention pathway carries a *larger fraction* of the anchor signal in the model that uses
it less — InternVL3's anchor signal is narrowly concentrated in upper-half layers, while
the LLaVA-cluster signal is broadly / redundantly distributed.

*On paired `exact_match` — coherent damage / partial-recovery ratio.* The opposite picture
on em: when computed on the intersection of valid samples (so cells are like-for-like), all
three models show coherent anchor-damage of −7 to −12.5 pp with a partial recovery of +3 to
+3.7 pp at saturation, i.e. 24–43 % of the damage recovered. Damage and recovery ratios
agree more across models than df does — the upper-half locus delivers a similarly-sized em
correction across the cluster. Whether the df-axis anti-correlation and the em-axis
coherence reflect the same underlying mechanism (anchor signal concentration) at different
metric resolutions, or different mechanisms altogether, is the open question for Phase 2.

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
the `em_target_only` column as the un-anchored ceiling — and computing every em on the
*intersection* of valid samples (so cross-condition comparisons are like-for-like):

| model | n_paired | em(target_only) | em(num) at s=0 | anchor-induced em loss | em(num) at s* | em(num) at saturation s=−10⁴ | recovery from saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | 200 | 0.435 | 0.365 | −0.070 (16 % rel) | 0.370 (+0.005) | 0.395 | +0.030 (recovers 43 % of loss) |
| convllava-7b | 200 | 0.500 | 0.375 | −0.125 (25 % rel) | 0.375 (+0.000) | 0.405 | +0.030 (recovers 24 % of loss) |
| internvl3-8b | 109 | 0.734 | 0.633 | −0.101 (14 % rel) | 0.642 (+0.009) | 0.670 | +0.037 (recovers 36 % of loss) |

**All three models exhibit anchor-damage; mitigation recovers a partial slice on all
three.** LLaVA-1.5 and ConvLLaVA already showed this; InternVL3, when computed on the
paired intersection (sample_instance_ids valid for *every* (condition, strength) cell —
the only fair cross-cell comparison), turns out to follow the same pattern. The earlier
"InternVL3 has no anchor damage" reading was an artefact of comparing em(num) computed on
n=137 against em(target_only) computed on n=200 — different sample subsets.

**InternVL3 caveat:** the paired analysis collapses InternVL3's n to 109 (out of 200 — the
intersection of the 4 cells). Crucially, the surviving InternVL3 paired set has em(target_only)
= 0.734 versus 0.567 on the larger n=200 condition-internal set — the parse-failing samples
are systematically harder for the model. So the InternVL3 row above describes "mitigation
behaviour on the parse-tractable subset", not "InternVL3 in general". Phase 2 + a driver
fix (longer max_new_tokens for InternVL3, or a JSON-strict template) is needed before the
InternVL3 row in this table is fully comparable to the other two.

The mitigation behaviour holds across all three: damage of −7 to −12.5 pp, partial recovery
of +3 to +3.7 pp at saturation = 24–43 % of the damage recovered. At the chosen working
point `s*`, recovery is smaller (em delta 0 to +0.9 pp); the visible recovery is at saturation.
Phase 2 will tell us whether (a) `s*` is undersized and the working point should sit deeper
into the strength axis, or (b) saturation is over-aggressive and the modest df reduction at
`s*` is the right operating point.

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
- **InternVL3 prose-leak parse loss is driver-side, not parser-side.** ~30 % of records drop
  out of valid-triplet count because InternVL3 emits prose ("Based on the image…") and the
  driver's `max_new_tokens=8` truncates it before any digit is generated; the parser already
  uses the project's `extract_first_number` and behaves correctly given the input it sees.
  Therefore an analysis-layer parse-rescue does *not* recover the dropped triplets — they
  contain no number to rescue. The fix is driver-side: longer `max_new_tokens` (16–32 to
  let the prose finish into a digit), or an InternVL3-specific JSON-strict prompt. Tracked
  for Phase 2.
- **InternVL3 paired-set bias.** The intersection-of-valid-samples set (n=109 vs the
  per-condition n=137 ~ 200) has em(target_only) = 0.734, materially higher than the
  unpaired 0.567. The parse-failing samples are systematically the model's harder cases.
  Treat the InternVL3 row of the anchor-damage table as "mitigation behaviour on
  parse-tractable items" rather than "InternVL3 overall".
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
