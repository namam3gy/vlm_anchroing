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

### convllava-7b (in flight)

Sweep running on GPU 0; rate ≈ 0.22 sample-instances/sec; ETA ~15 min from kick-off.
Initial partial table (n=26, all in `target_only` / no-shift cases) shows the hook is a
no-op on target_only as expected (em_target_only=0.731 across all strengths).

_(Table will be filled in once n=200 sweep completes.)_

### internvl3-8b (queued)

Will kick off as soon as convllava-7b finishes; same Phase 1 sweep.

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
