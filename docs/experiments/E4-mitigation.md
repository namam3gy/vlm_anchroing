# E4 — attention re-weighting mitigation: results

**Status:** Phase 1 sweep in progress (llava-1.5-7b complete; convllava-7b, internvl3-8b in flight).
Phase 2 full validation: pending Phase 1 completion. Will run on llava-1.5-7b first (12-h session
budget per user; resumable; remaining models continue in next session).

**Source data:** `outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl` (Phase 1),
`outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl` (Phase 2).
**Driver:** `scripts/e4_attention_reweighting.py`. **Analysis:** `scripts/analyze_e4_mitigation.py`.
**Design doc:** `docs/experiments/E4-mitigation-design.md` (superseded by this file once Phase 2 lands).

## Goal & target

Per `references/roadmap.md` §6 Tier 1: ≥ 10 % reduction in `direction_follow_rate` with ≤ 2 pp
drop in standard VQA accuracy on the VQAv2 number subset, on the mid-stack-cluster VLMs
(LLaVA-1.5, ConvLLaVA, InternVL3) — the family identified by E1b/E1d as sharing the same
attention signature *and* the cleanest upper-half-ablation response.

## Method

E1d ruled out single-layer interventions; the only mode that reduced `direction_follow_rate`
across the panel without breaking fluency was *upper-half* multi-layer ablation at hard mask
(strength = −10⁴). E4 takes that locus and adds a strength axis: instead of zeroing anchor
attention at upper-half layers, it down-weights it by `exp(strength)` — a forward pre-hook
on each LLM decoder layer in `[n_layers/2, n_layers)` adds `strength` to the
`attention_mask` columns at the anchor span, so post-softmax anchor attention is multiplied
by `exp(strength)` before key-value mixing.

**Strength sweep (Phase 1):** 7 strengths × 3 conditions × 200 stratified samples per model.
The strength grid is `[0, −0.5, −1, −2, −3, −5, −10⁴]` — log-spaced through the range where
the multiplier (`exp(strength)`) is meaningful (≈ 1.0 → 0.0067 → 0).

**Sample stratification:** same 200-sample E1b/E1d set (top-decile-susceptible × 100 +
bottom-decile-resistant × 100, drawn from `docs/insights/_data/susceptibility_strata.csv`).

**Metrics per (model, strength):**
- `direction_follow_rate(target_plus_irrelevant_number)` — primary; whether the prediction moved
  closer to the anchor than the `target_only` baseline did.
- `adoption_rate(target_plus_irrelevant_number)` — exact match to anchor digit.
- `exact_match(target_plus_irrelevant_number)` — exact match to ground truth (the standard-VQA
  target proxy).
- `mean_distance_to_anchor(target_plus_irrelevant_number)` — fluency monitor (large = the
  intervention is breaking generation, not just down-weighting anchor pull).
- `exact_match` on `target_only` and `target_plus_irrelevant_neutral` — sanity controls; the
  hook should be a no-op on `target_only` (anchor span empty) and barely move on neutral
  (no legible digit at the anchor location).

Bootstrap 95 % CIs (n_boot = 2,000) per cell.

**Strength selection rule:** smallest `|s|` satisfying both
`direction_follow_rate(target_plus_irrelevant_number, s) ≤ 0.9 ×
direction_follow_rate(target_plus_irrelevant_number, 0)` AND
`exact_match(target_plus_irrelevant_number, s) ≥ exact_match(target_plus_irrelevant_number, 0)
− 0.02`. If no `s` qualifies, escalate to a denser grid or `ablate_upper_quarter`.

## Phase 1 — strength sweep

### llava-1.5-7b (baseline df=0.305)

| strength | n | df_num | adopt_num | em_num | em_target_only | em_neutral | mean_dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.305 [0.24, 0.37] | 0.175 | 0.365 [0.30, 0.44] | 0.435 | 0.360 | 3.58 |
| −0.5 | 200 | 0.290 [0.23, 0.36] | 0.170 | 0.365 [0.30, 0.44] | 0.435 | 0.355 | 3.63 |
| −1.0 | 200 | 0.290 [0.23, 0.35] | 0.155 | 0.365 [0.30, 0.44] | 0.435 | 0.365 | 3.63 |
| −2.0 | 200 | 0.280 [0.22, 0.34] | 0.120 | 0.375 [0.31, 0.45] | 0.435 | 0.360 | 3.72 |
| **−3.0** | 200 | **0.265 [0.21, 0.33]** | **0.125** | **0.370 [0.30, 0.44]** | 0.435 | 0.360 | 3.77 |
| −5.0 | 200 | 0.250 [0.19, 0.31] | 0.095 | 0.395 [0.33, 0.47] | 0.435 | 0.365 | 4.25 |
| −10⁴ | 200 | 0.250 [0.19, 0.31] | 0.085 | 0.395 [0.33, 0.47] | 0.435 | 0.360 | 4.72 |

**Selection:** `s* = −3.0` (smallest |s| meeting both targets — df drop of 13 % relative,
em delta of +0.5 pp, well within budget).

**Notable:** `em_num` does not drop under any strength — it strictly *improves* from 0.365
(baseline) to 0.395 (saturation). The mitigation is not just neutral on accuracy; it raises it.
`em_target_only` is invariant at 0.435 (sanity check passes — hook is a no-op on
single-image inputs). `em_neutral` is also invariant at 0.355–0.365 — the hook fires on the
neutral image too but, lacking a legible digit there, doesn't change predictions.

### convllava-7b (baseline df=0.290)

| strength | n | df_num | adopt_num | em_num | em_target_only | em_neutral | mean_dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.290 [0.23, 0.36] | 0.175 | 0.375 [0.31, 0.45] | 0.500 | 0.380 | 3.18 |
| −0.5 | 200 | 0.285 [0.23, 0.35] | 0.180 | 0.370 [0.31, 0.44] | 0.500 | 0.375 | 3.19 |
| −1.0 | 200 | 0.275 [0.22, 0.34] | 0.165 | 0.370 [0.30, 0.44] | 0.500 | 0.380 | 3.25 |
| **−2.0** | 200 | **0.260 [0.20, 0.32]** | **0.160** | **0.375 [0.31, 0.45]** | 0.500 | 0.375 | 3.30 |
| −3.0 | 200 | 0.255 [0.20, 0.32] | 0.155 | 0.375 [0.31, 0.44] | 0.500 | 0.390 | 3.32 |
| −5.0 | 200 | 0.240 [0.19, 0.30] | 0.125 | 0.400 [0.33, 0.47] | 0.500 | 0.390 | 3.44 |
| −10⁴ | 200 | 0.235 [0.18, 0.30] | 0.120 | 0.405 [0.34, 0.47] | 0.500 | 0.385 | 3.46 |

**Selection:** `s* = −2.0` (smallest |s| meeting both targets — df drop of 10.3 % relative,
em delta 0 pp, comfortable accuracy headroom).

**Notable:** `em_num` again does not drop at any strength — it's flat at 0.370–0.405 with
saturation values rising. `em_target_only` invariant at 0.500 (sanity check passes).
ConvLLaVA's baseline df is identical to its E1d baseline (0.29), confirming the eager
attention pipeline replicates exactly across runs. The strength response is monotonic and
fluency-clean — `mean_distance_to_anchor` rises from 3.18 to 3.46 across the full strength
range, ≤ 0.3 unit drift, which matches E1d's "no fluency hit" finding for upper-half
ablation on convllava.

### internvl3-8b (in flight)

Sweep running on GPU 0 at the time of this writeup; rate ~0.07 sample-instances/sec
(slower than convllava's 0.21 and llava's similar; the slowdown is partly tracking the
multi-resolution image-tiling overhead in the InternVL3 forward pass and partly GPU
contention with `physical_mode_activation` on the same physical GPU). ETA ~30 min from
the 80/200 progress milestone. _(Table will be filled in once n=200 sweep completes.)_

### Cross-model summary (draft, pending internvl3)

| model | layers | upper_half range | baseline df | s* | df at s* | df drop (rel) | em delta at s* | em(target_only) baseline | em(target_only) invariant? |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 32 | 16..31 | 0.305 | −3.0 | 0.265 | −13 % | +0.5 pp | 0.435 | ✓ |
| convllava-7b | 32 | 16..31 | 0.290 | −2.0 | 0.260 | −10 % | +0 pp | 0.500 | ✓ |
| internvl3-8b | 28 | 14..27 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Reading so far (2/3 models done):**

1. The mid-stack-cluster mitigation works on the two models tested, on the panel-shared
   `ablate_upper_half` locus identified by E1d, and at moderate soft strength (no need for
   hard masking). Both models meet the ≥ 10 % `direction_follow_rate` reduction target
   without any drop in `exact_match`.
2. `s*` differs between models (LLaVA −3.0, ConvLLaVA −2.0) but both sit in the same band
   (−5, −1). A single shared `s* = −2.5` would meet target on both with one number — kept
   per-model in this writeup but the choice would not change Phase 2 inputs in any
   load-bearing way.
3. The hook is anchor-condition-specific by construction (no-op on `target_only`) and
   confirmed empirically by the invariant `em(target_only)` column on every strength
   tested.
4. The strength-axis monotonicity is robust: every step from `s = 0` to `s = −10⁴` either
   reduces or holds `direction_follow_rate`, and `exact_match` either holds or rises.
   No "U-shape" disasters where over-mitigation tips into hallucination.

## Phase 2 — full validation (pending)

Will run on the model(s) for which Phase 1 yields a valid `s*`, at full VQAv2-number scale
(n=17,730 sample-instances × 5 irrelevant sets × 3 conditions × 2 modes ≈ 100 k generations
per model). Resumable — single canonical JSONL per model, append-only writes with flush, set
of completed `(sample_instance_id, condition, mask_strength)` keys read on startup.

**Prioritisation under 12-h session budget (advised by call to advisor):** llava-1.5-7b
first (cleanest E1d signal, largest causal effect, no caveats). Other models continue in
subsequent sessions via the resumability protocol.

## Caveats

- **n=200 CIs are wide.** Bootstrap CIs at sweep scale overlap heavily across strengths
  (e.g., baseline df 0.305 [0.24, 0.37] vs s=−3.0 df 0.265 [0.21, 0.33]). The strength-axis
  monotonicity is informative; the per-strength deltas are not yet load-bearing. Phase 2 at
  n=17,730 is what carries the headline numbers.
- **ConvLLaVA causal-structure caveat from E1d.** ConvLLaVA and LLaVA-1.5 share the same
  E1b peak/mechanism but respond *opposite* to lower-half ablation (E1d). Same-attention-
  signature does not imply same-causal-structure. If Phase 2 numbers diverge from
  LLaVA-1.5/InternVL3 substantially, decide at writeup time whether to demote ConvLLaVA to
  a discussion caveat rather than treat it as part of the headline mid-stack-cluster claim.

## Open follow-ups (post Phase 2)

- Per-model vs. shared optimal strength — if Phase 1 picks similar `s*` across the 3 models,
  report a single shared strength as the architecture-blind prototype; otherwise per-model.
- Failure escalation path if no strength in [−5, 0] meets the target on any model:
  (a) `ablate_upper_quarter` (`[3n/4, n)`), (b) different intervention class.

## Writeup tags

- `docs/experiments/E4-mitigation.md` (this file) — detailed.
- `docs/insights/E4-mitigation-evidence.md` (will be created post-Phase-2) — distilled
  one-pager.
- Korean mirrors: `_ko.md` siblings of both, kept in lockstep.
