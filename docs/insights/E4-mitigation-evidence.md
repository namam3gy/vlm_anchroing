# E4 — upper-half attention re-weighting reduces anchor pull on the mid-stack cluster

**Status:** Phase 1 sweep complete on llava-1.5-7b; convllava-7b and internvl3-8b sweep in
flight. Phase 2 full-scale validation pending. Source data:
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

## What we found (preliminary, n=200)

| model | baseline df | s* | df at s* | em(num) at s=0 | em(num) at s* | em(target_only) invariant? |
|---|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 0.305 | **−3.0** | 0.265 (−13 %) | 0.365 | 0.370 (+0.5 pp) | ✓ (0.435) |
| convllava-7b | 0.290 | **−2.0** | 0.260 (−10 %) | 0.375 | 0.375 (+0.0 pp) | ✓ (0.500) |
| internvl3-8b | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Headline (so far):** the mitigation works on llava-1.5-7b at n=200 and meets the roadmap
target. Crucially, `em(target_plus_irrelevant_number)` does *not* drop — at saturation
(`s = −10⁴`) it climbs from 0.365 to 0.395. The mitigation appears not just safe for
accuracy but mildly beneficial. Phase 2 at full scale will tell us whether this is real.

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
- **ConvLLaVA causal-structure caveat (E1d).** Same-attention-signature ≠ same-causal-
  structure. ConvLLaVA's lower-half-ablation behaviour is the *opposite* of LLaVA-1.5's.
  If Phase 2 shows ConvLLaVA's E4 response diverges from LLaVA-1.5/InternVL3, demote it
  to a discussion caveat.
- **Selection at n=200 may shift at n=17,730.** Phase 2 may pick a different `s*` per
  model. Phase 2 reports both the Phase 1-chosen `s*` and the full bootstrap distribution.

## Open follow-ups (post Phase 2)

- Whether the pattern is universal across the 3 mid-stack models or LLaVA-1.5-only.
- Whether the accuracy gain at saturation is anchor-specific (rules out generic accuracy
  bump from down-weighting any second-image attention).
- If the mitigation works, whether it generalises beyond VQAv2-number to ChartQA/TallyQA
  (E5).
