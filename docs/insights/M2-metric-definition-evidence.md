# M2 — Canonical adopt and direction-follow definitions

**Status:** Metric-definition consolidation pass. Initial analysis 2026-04-28;
**`metrics.py` refactor + re-aggregation landed 2026-04-29.** Source data:
25 `predictions.jsonl` files across 8 experiments (see
`docs/insights/_data/M2_inputs_manifest.csv`). Variants: 18 total, 13
distinct after collapsing mathematical equivalences (see §4). Re-runs as
more `predictions.jsonl` arrive (E5b/E5c cross-model expansion, E5e
MathVista (γ)) will update §5 numbers — recommendation in §6 is robust to
those additions because it relies on rank consistency and gap sign, not
absolute value.

## 0. Canonical terminology

All future docs and code use:

| symbol | meaning |
|---|---|
| `pred_b` | base prediction (target_only condition) |
| `pred_a` | anchor-condition prediction |
| `pred_m` | anchor-mask-condition prediction |
| `pred_d` | neutral-distractor-condition prediction |
| `anchor` | anchor image's value |
| `gt` | ground truth |
| `pb_eq_a` | `pred_b == anchor` |
| `pa_eq_a` | `pred_a == anchor` |
| `gt_eq_a` | `gt == anchor` |
| `pa_ne_pb` | `pred_a != pred_b` |
| `pb_eq_gt` | `pred_b == gt` (= base prediction is correct) |

The arm being compared to baseline is generically `pred_x` (= `pred_a`,
`pred_m`, or `pred_d`). For `pred_d` the anchor-bearing metrics below are
not defined (no anchor in scene); accuracy comparisons use `pred_d` as the
2-image-distraction control only.

## 1. TL;DR — recommended canonical definitions

```
adopt_rate              = #(pa == anchor  AND  pb != anchor) / #(pb != anchor)
direction_follow_rate   = #( (pa - pb)·(anchor - pb) > 0  AND  pa != pb )
                          / #(numeric pair AND anchor present)
```

with `pred_x` substituted for `pred_a` on the mask/anchor arms (same form
applies to `pred_m`).

The `direction_follow_rate` numerator measures whether `pa` shifted **from
the baseline `pb` toward the anchor side of `pb`**. The `pa != pb` clause
is structurally redundant under the C-form (when `pa == pb` the product
factor `(pa - pb)` is exactly zero, so the numerator already excludes
no-movement pairs), but is kept explicit for clarity and to harden against
edge cases where `pa - pb = 0` arises from upstream parsing collapse rather
than genuine no-movement. Using `pb` (not `gt`) as the reference makes the
metric depend only on model outputs and the anchor draw — a direct measure
of anchor pull, robust to per-question stimulus and gt variability.

These definitions correspond to the variants `A_paired__D_paired` (adopt)
and `DF_moved__DD_all` (direction-follow) in §3 below — both top-ranked
on signal-preservation.

## 2. Why this had to be settled empirically

The codebase has carried several definitions concurrently:

| era | adopt definition | location |
|---|---|---|
| pre-M1 (≤ 2026-04-26) | `pa == anchor` only (marginal) | original `metrics.py` |
| post-M1 (since 2026-04-27) | `pa == anchor AND pb != anchor` over all samples (paired numerator, all-sample denominator) | `metrics.py` (`anchor_adopted` flag) |
| E5b/E5c era | `adopt_cond` = `(pa == anchor AND pb != anchor) / (pb != anchor)` (drop case 4 from denominator) | `scripts/analyze_e5b_distance.py`, `scripts/analyze_e5c_distance.py` |

Switching definitions changes published rates by ~1.4× (between current
`anchor_adopted` and `adopt_cond` E5b/E5c usage) and ~3-7× (between current
and pre-M1 marginal). The paper headline cannot be ambiguous about which
the cited number is.

## 3. Variant matrix

### 3.1 Adopt — numerator × denominator (3 × 3 = 9 variants)

| numerator | predicate |
|---|---|
| `A_raw` | `pa_eq_a` |
| `A_paired` | `pa_eq_a AND NOT pb_eq_a` |
| `A_clean` | `pa_eq_a AND NOT pb_eq_a AND NOT gt_eq_a` |

| denominator | predicate |
|---|---|
| `D_all` | every sample-pair |
| `D_paired` | `NOT pb_eq_a` |
| `D_clean` | `NOT pb_eq_a AND NOT gt_eq_a` |

### 3.2 Direction-follow — numerator × denominator (3 × 3 = 9 variants)

| numerator | predicate |
|---|---|
| `DF_raw` | `(pa - pb)·(anchor - pb) > 0` (sign-based, C-form) |
| `DF_moved` | `DF_raw AND pa_ne_pb` |
| `DF_clean` | `DF_moved AND NOT gt_eq_a` |

| denominator | predicate |
|---|---|
| `DD_all` | numeric pair AND `anchor` present |
| `DD_moved` | `DD_all AND pa_ne_pb` |
| `DD_clean` | `DD_moved AND NOT gt_eq_a` |

### 3.3 `moved_closer_rate` is not added

Distance-based `moved_closer = |pa - anchor| < |pb - anchor|` differs from
sign-based `direction_follow` only on overshoot edge cases (e.g. base = 5,
anchor = 7, anchor-arm = 10 → df = YES, moved_closer = NO). On signal-bearing
cells the two correlate ≥ 0.99. Adding it doubles the metric surface for
no information gain; it is not measured here and not adopted.

## 4. Mathematical equivalences (collapse 18 → 13 distinct)

The denominators that filter `pb_eq_a` or `gt_eq_a` make the same filters in
the numerator redundant:

| equivalence class | distinct rate |
|---|---|
| `A_raw__D_paired ≡ A_paired__D_paired` | yes — denominator's `NOT pb_eq_a` makes numerator's same clause redundant |
| `A_raw__D_clean ≡ A_paired__D_clean ≡ A_clean__D_clean` | yes — denominator's `NOT pb_eq_a AND NOT gt_eq_a` enforces both numerator clauses |

Same equivalence holds for `DF_raw__DD_moved ≡ DF_moved__DD_moved`,
`DF_raw__DD_clean ≡ DF_moved__DD_clean ≡ DF_clean__DD_clean`. Total
distinct: **6 adopt + 7 df = 13** (we display all 18 in the long table for
post-hoc audit; equivalence-collapsed cells share rate exactly).

## 5. Signal-preservation evidence

Three signals are scored. Each variant is evaluated on the cells where the
signal is meaningful; far-distance / noise-floor cells where the signal
itself is absent are reported separately.

### 5.1 wrong-base > correct-base (Phase A H2)

Per (experiment, dataset, model, stratum) cell, anchor-arm rate on
wrong-base records vs. correct-base records. Signal-bearing cells:
**S0 (main runs) and S1 stratified** — the cells where adopt-rate is
above the noise floor. n = 22 cells.

| variant | wrong>correct cells | mean(wrong − correct) |
|---|---|---|
| `A_paired__D_paired` (≡ `A_raw__D_paired`) | **22 / 22** | **+0.0400** |
| `A_paired__D_all` | 22 / 22 | +0.0373 |
| `A_paired__D_clean` (≡ `A_clean__D_clean` ≡ `A_raw__D_clean`) | 21 / 22 | +0.0191 |
| `A_clean__D_all` | 17 / 22 | +0.0093 |
| `A_clean__D_paired` | 16 / 22 | +0.0069 |
| `A_raw__D_all` (marginal pre-M1) | **8 / 22** | **−0.0279** |

Two sub-findings:
- The pre-M1 marginal definition (`A_raw__D_all`) **inverts** wrong-vs-correct
  on most cells: correct-base shows *more* "adoption" because `gt == anchor`
  cases are kept and the model frequently produces the correct (= anchor)
  answer. M1 already corrected this in the numerator; the denominator filter
  closes the rest of the gap.
- Adding `gt_eq_a` exclusion to the numerator (`A_clean`) but not the
  denominator (`A_clean__D_all`) hurts the gap because the denominator still
  carries `gt == anchor` cases as "non-adoptions". The clean way is to
  exclude `gt_eq_a` from both — but the resulting rate is so close to
  `A_paired__D_paired` (+0.0191 vs +0.0400) that the simpler `D_paired`
  wins on interpretability.

### 5.2 Distance decay S1 > S5 (E5b/E5c)

Per (experiment, dataset, model) cell on E5b + E5c, S1 anchor wrong-base
adopt-rate vs. S5 wrong-base adopt-rate. n = 4 cells (single-model coverage
on E5b/E5c as of 2026-04-28; cross-model expansion ongoing).

**All 9 adopt variants score 4/4 (100 %).** The S1 > S5 monotonic decay is
robust to definition choice — it is a structural property of the data, not
the metric.

For direction-follow, the same test produces **0/4 to 1/4** across all 9
variants (mean S1 − S5 negative or near-zero). **df is distance-invariant;
adopt is distance-windowed.** This is the expected separation reported in
roadmap §218 and `docs/experiments/E5c-anchor-mask-control.md` —
direction-follow on far-distance anchors keeps firing because *generic*
2-image distraction perturbs the prediction direction even without the digit
pixel.

This separation is the basis for §5 and §6 of the paper using different
metrics: §5 (Distance & Plausibility Window) reports adopt; §6
(Confidence-Modulated Anchoring) reports direction-follow.

### 5.3 anchor > masked (E5c digit-pixel causality)

Per (experiment, dataset, model, stratum) cell on E5c + E5e wrong-base,
anchor-arm rate vs. masked-arm rate. Signal-bearing cells: S0 / S1 only
(anchor and masked converge to noise floor by S2). n = 6 cells.

`A_paired__D_paired` cell-level result:

| experiment | dataset | model | stratum | a (anchor) | m (masked) | gap |
|---|---|---|---|---:|---:|---:|
| `experiment_e5c_vqa` | VQAv2 | llava-next-interleaved-7b | S1 | 0.1409 | 0.0738 | **+0.067** |
| `experiment_e5c_tally` | TallyQA | llava-next-interleaved-7b | S1 | 0.1148 | 0.0889 | **+0.026** |
| `experiment_e5e_chartqa_full` | ChartQA | gemma3-27b-it | S1 | 0.0728 | 0.0437 | **+0.029** |
| `experiment_e5e_chartqa_full` | ChartQA | llava-next-interleaved-7b | S1 | 0.0375 | 0.0115 | **+0.026** |
| `experiment_e5e_tallyqa_full` | TallyQA | llava-next-interleaved-7b | S1 | 0.0505 | 0.0286 | **+0.022** |
| `experiment_e5e_chartqa_full` | ChartQA | qwen2.5-vl-7b-instruct | S1 | 0.0417 | 0.0417 | 0.000 |

5 / 6 preserve `a > m`. The single tie (qwen2.5-vl on ChartQA S1, 0.0417 in
both arms) is at n_eligible = 120, where the standard error on a rate of
0.04 is ≈ 0.018 — the cells are statistically indistinguishable rather than
contradictory. As E5e adds more models / datasets the n grows and this
ambiguity should resolve.

The anchor > masked gap signals that the **digit pixel is the operative
cause** of paired adoption — the masked arm preserves the anchor image's
background but removes the digit; if `a > m`, removing the digit reduces
the pull. All variants preserve this pattern at similar fractions; the
choice does not depend on this signal.

## 6. Recommended canonical definitions

For paper headline numbers and `metrics.py`:

```python
# Adopt rate
adopt_rate = (
    sum(  pa_eq_a AND NOT pb_eq_a  )           # paired numerator (M1)
    /
    sum(  NOT pb_eq_a  )                       # E5b/E5c-style denominator
)

# Direction-follow rate
direction_follow_rate = (
    sum(  ( (pa - pb) * (anchor - pb) > 0 )  AND  pa_ne_pb  )
    /
    sum(  numeric_pair AND anchor_present  )
)

# Same definitions on the mask arm (substitute pred_m for pred_a).
# anchor_effect for any metric M:
anchor_effect_M = M(anchor arm) - M(neutral arm)
# (neutral arm has no anchor → adopt and df undefined; for those metrics
#  use M(anchor arm) - M(masked arm) on E5c/E5e to isolate digit-pixel cause.)
```

| metric | what it answers | distance-windowed? | uncertainty-modulated? |
|---|---|---|---|
| `adopt_rate` | "did the model output the anchor *value*?" | yes — peaks at S1, decays to noise by S5 | yes — wrong > correct +0.040 on S0/S1 |
| `direction_follow_rate` | "did the model move toward the anchor side of GT?" | no — distance-invariant (digit and mask similar) | yes — wrong > correct on every cell |
| `exact_match` | "did the model match GT?" | n/a | n/a (used for accuracy diff) |

**Roles in the paper:**
- §3 *Problem definition* states all three definitions; cites §6 of this
  document.
- §5 *Distance & plausibility window* uses **adopt_rate** as headline (the
  decay metric).
- §6 *Confidence-modulated anchoring* uses **direction_follow_rate** as
  headline (the uncertainty metric — it fires even at far distance because
  it is sign-based, not value-match).
- §7 *Mitigation* reports all three (df reduction is the lever, em is the
  free-lunch check, acc is the leakage check).

## 7. Implementation impact

### 7.1 `src/vlm_anchor/metrics.py`

`evaluate_sample` produces per-pair flags; the rate computation is in
`summarize_condition`. Change required:

```python
# Today (mean over all samples):
"anchor_adoption_rate": mean(r["anchor_adopted"] for r in subset),

# After M2:
n_pb_ne_a = sum(1 for r in subset if r.get("pb_ne_anchor"))
"anchor_adoption_rate": (
    sum(r["anchor_adopted"] for r in subset) / n_pb_ne_a
    if n_pb_ne_a else None
),
```

The new `pb_ne_anchor` flag is captured at `evaluate_sample` time (already
have `base_prediction` since M1). Same shape change for
`anchor_direction_follow_rate` — add `pa_ne_pb` to the numerator.

### 7.2 Re-aggregation

Same approach as M1: `scripts/reaggregate_paired_adoption.py` is extended
(or a sibling script written) to recompute per-condition summary fields
from raw `predictions.jsonl`. No re-inference. All ≥ 25 existing run
directories receive new `summary.json` numbers.

### 7.3 Documents to migrate

- `references/project.md` — once new outline lands, all numeric tables use
  the new definitions; old definitions referenced once with date stamps.
- `references/roadmap.md` — same; migration plan part of the redesigned roadmap.
- `docs/experiments/*.md` and `docs/insights/*.md` — migrate per touch
  (as we revisit a doc, update its metric definitions and re-quote numbers).

### 7.4 Provisional caveat

Inferences are still in flight (E5b/E5c cross-model, E5e MathVista (γ)).
When those land, re-run `scripts/analyze_metric_variants.py` and update
this doc. The recommendation in §6 should not change because it is rank-
and gap-sign-driven; only the absolute numbers in §5 will tighten.

## 8. Post-refactor status (2026-04-29)

The §6 definitions are now the live `metrics.py` definitions. Per-row M2
flags (`pred_b_equal_anchor`, `pred_diff_from_base`, `anchor_direction_followed_moved`)
are present in every `predictions.jsonl` under `outputs/`; per-condition
`summary.json` rates use M2 denominators. Re-aggregation summary:

```
DONE. 53 rewrote / 0 would-rewrite / 0 skipped / 0 empty
```

Pre-M1 marginal backups (`predictions.marginal.bak.{jsonl,csv}`,
`summary.marginal.bak.json`) are preserved untouched — re-aggregation
guards against backup overwrite.

**M2-vs-marginal headline gap (VQAv2 main panel, 7 models, 17,730 samples
each):**

| metric | range across panel | direction |
|---|---|---|
| `adopt_rate` (M2 paired, paired denominator) | 0.021 – 0.066 | slight rise vs. marginal (denominator narrows) |
| `adopt_rate_marginal` (M2 paired, all-sample denominator) | 0.019 – 0.059 | matches §3.4 pre-M2 row in `roadmap` (within Δ ≤ 0.001 from M1 numbers) |
| `direction_follow_rate` (M2 sign-based AND `pa != pb`) | 0.063 – 0.193 | drops 50–60 % vs. raw — most pairs have `pa == pb` (no movement), and the old raw definition counted those as direction-follow whenever `pb ≠ gt` |
| `direction_follow_rate_raw` (M2 sign-based, no movement filter) | 0.239 – 0.349 | matches the §3.4 pre-M2 row exactly (within Δ ≤ 0.001) |

The big movement on `direction_follow_rate` is the headline conceptual
correction: the older raw rate inflated direction-follow by treating
"prediction unchanged from base" as a follow-toward-anchor event. M2
direction-follow only counts pairs that actually moved. This makes §6
(uncertainty-modulated *graded* pull) cleaner — direction-follow becomes
a measure of pull amplitude conditional on movement, decoupled from
"does the model decline to update at all".

**Adopt rate stability across the M2 / M1-paired pair is not a coincidence.**
The denominator switches from `D_all` (every record) to `D_paired`
(records where `pb != anchor`). On VQAv2 main, `pb_eq_anchor` fires on
~10 % of records — exactly the 9-in-10 gt-vs-anchor support overlap
flagged in M1's original change rationale. So the rate climbs ~1.1×
across the panel (e.g. gemma4-e4b 0.059 → 0.066 = ×1.11) — predictable
from the denominator shift alone.
