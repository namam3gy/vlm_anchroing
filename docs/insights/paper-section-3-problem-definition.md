# §3 — Problem Definition and Canonical Metrics

**Status:** First-draft paper prose, 2026-04-29. Anchored to
`references/project.md §0.3` and `docs/insights/M2-metric-definition-evidence.md`.
This section covers the four-condition prompt design, the JSON-strict
inference template, and the four canonical metrics used throughout §5–§7.

---

## 3.1 Setup

We study cross-modal numerical anchoring in vision–language models (VLMs):
given a numerical visual-question-answering (VQA) item — a target image and
a question whose ground-truth answer is an integer — does presenting an
*irrelevant second image* containing a rendered number bias the model's
prediction toward that number? The target image carries the legitimate
visual evidence; the second image is task-irrelevant by construction. Any
shift in the model's answer when the second image is present therefore
isolates a cross-modal anchoring effect from the numerical evidence the
model is supposed to use.

Each VQA *sample-instance* expands into up to four conditions, distinguished
by the second image. We use the canonical short codes `b`, `a`, `m`, `d`
throughout the paper, code, and analysis artefacts:

| code | second image | role |
|---|---|---|
| `b` (`target_only`) | none | baseline — the model sees only the target image |
| `a` (`anchor`) | rendered digit-number image (the *anchor value*) | manipulation |
| `m` (`anchor-masked`) | the anchor image with the digit pixel region inpainted out (Telea) | digit-pixel control |
| `d` (`neutral`) | a digit-free FLUX-rendered scene image | two-image distraction control |

For every sample-instance the same target image, question, and ground-truth
integer `gt` are paired with each of the four second images. Predictions are
recorded as `pred_b`, `pred_a`, `pred_m`, `pred_d`; the anchor's numeric
value is recorded as `anchor`. Boolean flags follow the same prefix
convention: `pb_eq_a` is `pred_b == anchor`, `pa_ne_pb` is `pred_a != pred_b`,
`gt_eq_a` is `gt == anchor`, `pb_eq_gt` is `pred_b == gt` (i.e. base
prediction is correct), and so on.

The four conditions are not redundant. Three pairwise gaps each isolate a
distinct causal claim:

* **`a` − `d`** separates anchoring from generic two-image distraction.
  Both `a` and `d` add a second image to the prompt; only `a` carries a
  digit. Any effect that survives this contrast cannot be explained by
  "the model degrades whenever a second image is present" alone.
* **`a` − `m`** isolates the digit-pixel contribution. The `m` image is
  pixel-equivalent to `a` everywhere except at the digit region, which is
  inpainted (Telea, OpenCV). The contrast therefore zeros out anchor-image
  background, lighting, and composition, leaving only the digit pixels as a
  causal candidate.
* **wrong-base − correct-base on the same arm** isolates uncertainty
  modulation. Conditioning the rate on `pb_eq_gt` partitions every cell
  into items the model originally got right (low uncertainty) and items it
  originally got wrong (high uncertainty), measuring whether the anchor
  pulls more strongly when the model is unsure.

These three gaps respectively underwrite §5 (digit-pixel causality and
distance), §6 (confidence-modulated anchoring), and §7 (mechanistic
attention + mitigation). Conditions `b/a/d` are present in every dataset
in our matrix; condition `m` is added on the cross-dataset 4-condition
runs (E5c on VQAv2 and TallyQA, E5e on ChartQA, TallyQA, MathVista) where
digit-pixel causality is the load-bearing claim.

## 3.2 Inference template

All experiments use the same JSON-strict prompt template, parameterised
only by the question text. The model is instructed to emit a single JSON
object with one numeric field; the response is parsed back into an integer
prediction by extracting the first contiguous numeric span (with a
fallback that recovers numbers from prose-leak responses on models that
ignore the JSON instruction).

```
[system]
You are a visual question answering system.
Return valid JSON only in the form {"result": <number>}.
Use a numeric JSON value for <number>, not a string.
Do not output any other keys, words, explanation, or markdown.
If uncertain, still output the single most likely number in that JSON format.

[user]
Answer the question using the provided image(s).
Return JSON only in the form {"result": <number>}.
Question: {question}
```

Generation uses greedy decoding (temperature 0.0, top-p 1.0,
`max_new_tokens = 8`) so that all reported variation is between
*conditions* on the same question, not between samples of the same
condition. The 8-token cap matches the JSON template; we explicitly note
in §9 that two models in the panel (`fastvlm-7b`, `internvl3-8b`) emit
prose despite the instruction and benefit from the parser's prose-leak
fallback. Image order within a prompt is fixed (target first, second image
second) and the same across all four conditions for a given sample-instance.

The image order, prompt template, and decoding parameters are held fixed
across every dataset, model, and condition reported in the paper. Each
condition therefore differs from `b` only by the addition of a single
specific second image; this is the experimental contrast on which the
metrics in §3.3 operate.

## 3.3 Canonical metrics

Behavioural results are reported with four canonical metrics — `adopt_rate`,
`direction_follow_rate`, `exact_match`, and the across-arm gap
`anchor_effect = M(a-arm) − M(d-arm)`. Each is computed per
(model, dataset, condition) cell, and 95 % confidence intervals are
reported with paired non-parametric bootstrap over sample-instances
unless otherwise noted.

The two cross-modal anchoring rates are defined as

```
adopt_rate            = #(pa == anchor  AND  pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pa − pb)·(anchor − pb) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
anchor_effect_M       = M(a-arm) − M(d-arm)         (M ∈ {adopt_rate,
                                                          direction_follow_rate,
                                                          exact_match})
```

with `pred_m` substituted for `pred_a` on the mask arm (same form). The
neutral arm `d` is reported only on `exact_match` and as the reference
arm in `anchor_effect`; the anchor-bearing rates are undefined for `d`
because it has no anchor.

`adopt_rate` measures the fraction of base-different items on which the
anchor's *literal value* surfaces in the model's prediction. The numerator
clause `pb != anchor` matters: without it, items where the base prediction
already coincides with the anchor (which inflates whenever the anchor
inventory overlaps the dataset's GT support) inflate the rate by an
arm-invariant constant and hide the true contrast. The denominator
restriction `pb != anchor` further removes the unidentifiable
"already-matched" sub-population from the base rate.

`direction_follow_rate` measures the fraction of items on which the
prediction *moves toward* the anchor relative to the model's anchor-free
baseline. A sign-based criterion `(pa − pb)·(anchor − pb) > 0` catches
movement from `pred_b` toward the anchor without requiring `pa` to land
on the anchor exactly. This **C-form** is gt-free: it asks whether the
anchor stimulus shifted the prediction, irrespective of where the ground
truth sits, which makes the rate robust to per-question stimulus draws.
The numerator's `pa != pb` clause is structurally redundant under the
C-form (when `pa = pb`, the first factor is 0 and the predicate fails)
but is kept explicit to lock the "no-change" interpretation: only items
on which the anchor *moved* the prediction count.

`exact_match` is the per-arm task accuracy. We report it on every arm
(`b`, `a`, `m`, `d`) so that anchor effect on accuracy can be read
directly from the cross-arm difference.

`anchor_effect_M` reports the across-arm gap on a chosen metric `M`,
typically with the neutral arm `d` as reference (because `d` controls for
"the model has two images now"). For example, `anchor_effect_{exact_match}
= exact_match(a-arm) − exact_match(d-arm)` measures the accuracy cost
specifically attributable to *the digit*, beyond the cost of any second
image being present.

### Why these specific definitions

The codebase carried three concurrent definitions before this consolidation
(pre-M1 marginal `pa == anchor`; M1 paired-numerator-only over all samples;
the E5b/E5c era `adopt_cond` with paired denominator). Switching definitions
moves the same number by 1.4× to 7×. We resolved the ambiguity empirically:
across 18 numerator × denominator variants applied to 25 `predictions.jsonl`
files spanning the seven-model main panel and the cross-dataset E5b/E5c/E5e
runs, the variants `A_paired__D_paired` (adopt) and `DF_moved__DD_all`
(direction-follow) are the top-ranked variants on three independent
known-signal contrasts: wrong > correct on the S0/S1 cells (22/22 wins,
mean gap +0.040 — the largest gap among adopt variants that also clears
22/22), S1 > S5 on the wrong-base distance sweep, and anchor > masked
on the digit-pixel contrast. Several adjacent variants (notably
`A_paired__D_all` at 22/22 with gap +0.037, and `A_paired__D_clean` at
21/22 with gap +0.019) preserve the same wrong > correct contrast; we
prefer `D_paired` over `D_all` because it removes the unidentifiable
`pred_b == anchor` sub-population from the base rate, and over `D_clean`
because the additional `gt != anchor` filter trades signal magnitude for
a confound (the `gt == anchor` cell carries genuine anchoring evidence
that the `D_clean` denominator silently drops). Pre-M1 marginal definitions
*invert* the wrong > correct contrast on adopt because of the
`gt == anchor` confound: items where the anchor happens to equal the
ground truth become ineligible for the predicate `pred_b ≠ anchor` and
are silently moved between numerator and denominator. The full per-variant
analysis lives in `docs/insights/M2-metric-definition-evidence.md`; the
choice committed here is what re-aggregated all 53 archived run
directories on 2026-04-29.

### Reading the metrics together

Across our seven-model panel × four-dataset matrix, the recurring
pattern under C-form is **graded tilt**: `direction_follow_rate(a) >
adopt_rate(a)` on every cell, with `exact_match(a) ≈ exact_match(b)`.
The anchor moves predictions toward itself but rarely lands on the
anchor's literal value; the mass of the effect lives in baseline-relative
shift, not categorical replacement. The dataset with the largest
single cell is **MathVista on `gemma3-27b-it`** (wrong-base S1
`adopt_rate = 0.230`, `direction_follow_rate = 0.332`), and the
smallest is ChartQA on `qwen2.5-vl-7b` (all-base `adopt_rate = 0.017`,
`direction_follow_rate = 0.030`). Adopt magnitudes are 2-7 % on VQAv2,
1-4 % on ChartQA, 2-18 % on MathVista; direction-follow magnitudes
are correspondingly larger and more dispersed, consistent with the
graded-tilt reading.

> **Historical note.** A pre-2026-04-28 framing distinguished a
> "categorical-replace regime" (`direction_follow_rate ≈ 0`, large
> `adopt_rate`) on MathVista. That apparent regime was a driver-bug
> artefact: `direction_follow_rate` was being read as 0 on directly-
> launched runs because three M2 row-dict fields were never threaded
> from `evaluate_sample` into `predictions.jsonl`. After the fix and
> re-aggregation (changelog entry 2026-04-28), every cell in the
> matrix is in the graded-tilt regime; "categorical replacement" as
> a separate phenomenon does not survive C-form re-aggregation. §6
> connects the *magnitude* gradient across cells to the underlying
> logit-confidence distribution.
