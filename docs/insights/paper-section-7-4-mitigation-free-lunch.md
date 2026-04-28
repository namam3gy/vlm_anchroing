# §7.4 — Mitigation: a "free-lunch" attention re-weighting

**Status:** First-draft paper prose, 2026-04-29. Anchored to
`docs/insights/E4-mitigation-evidence.md` (full Phase 2 numbers),
`docs/experiments/E4-mitigation.md` (design + per-strength sweep),
and `references/roadmap.md §3.3` (panel-level headline numbers).
This subsection sits inside §7 (attention mechanism + mitigation).
§7.1–§7.3 establish the four-archetype attention picture (E1, E1b),
the falsified single-layer ablation, and the upper-half multi-layer
ablation as the only architecture-blind mitigation locus (E1d).
§7.4 carries the locus from a hard mask to a soft, deployable
intervention and tests it at full data scale.

---

## 7.4.1 From ablation to a continuous strength axis

E1d localised mitigation to the **upper half of the LLM stack** as the
only attention-mask intervention that reduced `direction_follow_rate` on
all six panel models without breaking fluency on the mid-stack cluster
(LLaVA-1.5, ConvLLaVA, InternVL3). E1d's hard mask is not deployable —
zeroing entire layer-block attention rows degrades fluency on three of
the six models — but it tells us *where* to intervene. E4 replaces the
hard mask with a continuous strength axis: an `exp(strength)` multiplier
applied only to *anchor-image* attention rows on upper-half layers, with
a per-model `s*` chosen to minimise the smallest `|strength|` that hits
≥ 10 % relative reduction in `direction_follow_rate` while keeping the
single-image `exact_match` drop within 2 pp on the stratified Phase 1
sweep (n = 200, 7 strengths).

The hook attaches to the second-image token span only. On the `b`
condition (no second image) the hook fires on an empty span and is a
no-op by construction. On the `d` (neutral) condition it fires on the
neutral image's tokens, which carry no anchor signal to suppress. On
`a` (anchor) and `m` (anchor-masked) it fires on the digit-bearing or
inpainted span. This per-condition behaviour is what underwrites the
acc(b) invariance claim in §7.4.2: the intervention is
*anchor-condition-specific* by construction, not as an empirical
coincidence.

## 7.4.2 The free-lunch result

Phase 2 validates the Phase 1 working point on the **full** VQAv2
number subset (17,730 sample-instances per model × 5 conditions ≈
88,650 records per model), with bootstrap confidence intervals roughly
ten times tighter than Phase 1's. Across the three mid-stack-cluster
models, three properties hold simultaneously at the Phase-1-chosen `s*`:

> **`direction_follow_rate(a)` decreases**, **`exact_match(a)` rises**,
> and **`exact_match(b)` is invariant.**

| Model | `s*` | df(a) `s=0` → `s*` | rel Δ df(a) | em(a) `s=0` → `s*` | abs Δ em(a) | em(b) `s=0` | em(b) `s*` | em(b) flat? |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `llava-1.5-7b` | −3.0 | 0.2578 → 0.2122 | **−17.7 %** | 0.3340 → 0.3418 | **+0.77 pp** | 0.3696 | 0.3696 | ✓ |
| `convllava-7b` | −2.0 | 0.2283 → 0.2042 | **−10.6 %** | 0.3522 → 0.3652 | **+1.30 pp** | 0.4454 | 0.4454 | ✓ |
| `internvl3-8b` | −0.5 | 0.1035 → 0.0975 | **−5.8 %** | 0.5902 → 0.5950 | **+0.49 pp** | 0.6325 | 0.6325 | ✓ |

`em(b)` is the model's `exact_match` on the `target_only` condition
measured on the same paired sample-instances used for the anchor arm.
The invariance is not the trivial statement that we chose `s*` to keep
it flat: `em(b)` is the same across *all seven* sweep strengths in
Phase 1 (from `s = 0` through saturation `s = −10⁴`) on every model,
because the hook never fires when there is no second image. The flat
column is a structural property of where the strength multiplier
attaches, and Phase 2 reproduces it at n = 17,730.

The three properties are mutually consistent under a single
interpretation. The intervention removes a portion of the anchor-image
contribution to the upper-stack residual stream; when there is anchor
signal to remove (the `a` arm), `direction_follow_rate` decreases and
some answers that were anchor-pulled from correct toward incorrect are
recovered, lifting `exact_match(a)`; when there is no anchor signal
(the `b` arm), the hook fires on an empty span and the model's normal
behaviour is preserved.

## 7.4.3 Why this is "free lunch", and what the recovery ratio limits

We use the term *free lunch* to mean three things together. (i) **No
trade-off on the metric we are intervening on.** Most attention-side
debiasing interventions in the LLM literature improve a target metric
at measurable accuracy cost; here `exact_match(a)` rises by +0.49 to
+1.30 pp on the very arm where `direction_follow_rate` falls.
(ii) **No collateral damage on the unintervened arm.** `exact_match(b)`
is invariant — the model's underlying VQA capability is untouched and
deployment does not require re-validation on the un-anchored
distribution. (iii) **No hand-tuning per dataset.** A single strength
choice on n = 200 stratified Phase 1 samples generalises to n = 17,730
on every model in the cluster, with the relative df reduction matching
within 0.3 pp of the Phase 1 prediction on LLaVA and ConvLLaVA.

What free lunch *does not* mean is that the intervention reverses all
anchor damage. The paired anchor-damage analysis on the full set
quantifies this:

| Model | n_paired | em(b) | em(a) at `s = 0` | anchor damage | em(a) at `s*` | recovery | % of damage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llava-1.5-7b` | 17,724 | 0.3696 | 0.3340 | **−3.55 pp** | 0.3417 | +0.77 pp | **21.7 %** |
| `convllava-7b` | 17,722 | 0.4454 | 0.3520 | **−9.34 pp** | 0.3651 | +1.31 pp | **14.0 %** |
| `internvl3-8b` | 11,848 | 0.6325 | 0.5938 | **−3.87 pp** | 0.5977 | +0.40 pp | **10.2 %** |

The intervention recovers between 10 % and 22 % of the anchor's
accuracy cost; it does not push `exact_match(a)` back to `exact_match(b)`.
Two reasons can be separated empirically. First, `direction_follow_rate`
drops faster than `exact_match` rises because direction-follow counts
sub-anchor movement that `exact_match` does not credit until the
prediction crosses the integer-equality boundary. Second, the upper-half
mass that is suppressed is only one of multiple redundant pathways
through which the anchor enters the answer (E1d showed single-layer
ablation null on every model) — closing one pathway saves a fraction
of the damage proportional to the fraction of anchor signal it carried
on each model.

## 7.4.4 Where the result generalises and where it does not

Three claims about scope. The cluster-level claim is that an
**encoder-blind** locus (upper-half attention re-weighting) generalises
across three mid-stack-cluster models with three architecturally
different vision encoders (CLIP-ViT, ConvNeXt, InternViT). The same
single-locus rule, with no per-encoder retuning, fires on all three.
We do not claim it generalises to the SigLIP-Gemma-early or
Qwen-ViT-late or FastVLM-late archetypes from §7.2; their per-layer
profiles peak elsewhere and Phase 1 sweeps on those archetypes are
deferred (§roadmap-P3 — opportunistic).

The dataset-level claim is single-dataset: VQAv2 number subset only.
We do not claim the working point `s*` transfers unchanged to ChartQA,
TallyQA, or MathVista. The mitigation locus *should* generalise (it is
defined on the LLM stack, not on the dataset), but the optimal `s*`
likely shifts with the per-dataset distance distribution and the
graded-tilt magnitude characterised in §3.3 and §5 (MathVista is the
largest single cell in the panel; ChartQA is the smallest). We flag
this as a limitation in §9 and as a tractable follow-up.

The model-level claim is full-cluster: all three Phase 2 results were
collected at full data scale with bootstrap CIs. The relative reduction
ranks LLaVA (−17.7 %) > ConvLLaVA (−10.6 %) > InternVL3 (−5.8 %), in
the same order as the *baseline* `direction_follow_rate` (LLaVA 0.258
> ConvLLaVA 0.228 > InternVL3 0.103). The mitigation effect scales
with the anchor signal available to remove; on the H6 "distraction-
not-anchoring" model (InternVL3) there is less anchor signal in the
full distribution to begin with and the absolute reduction is
correspondingly smaller. This is a feature, not a bug: the
intervention does not "find" anchor pull where there is none.

## 7.4.5 Caveats carried forward

* **ConvLLaVA fluency tail.** `mean_distance_to_anchor` jumps from
  2.99 to 53.54 on ConvLLaVA at `s*` (vs. 3.18 → 3.30 on the Phase 1
  stratified sweep). A small fraction of samples receive predictions
  far from any plausible answer; `exact_match` still rises because the
  bulk of the distribution improves enough to net positive, but the
  *mean* distance is not a robust summary statistic. We report the
  **median** distance and a fluency-degraded fraction count in the
  paper, with the unwinsorised mean only in the supplementary.
* **InternVL3 parse loss.** Roughly 33 % of records on InternVL3 drop
  out of the paired-valid set (`n_paired = 11{,}848` of 17,730) because
  the `max_new_tokens = 32` driver patch was applied during the run
  rather than before it. The dropped items are systematically harder
  (paired-set `em(b) = 0.6325` vs. full-panel `em(b) = 0.5760`); we
  treat the InternVL3 row as "behaviour on the model's parse-tractable
  subset" and explicitly note this scope.
* **Single dataset.** VQAv2 number only. Cross-dataset Phase 2 on
  ChartQA / TallyQA / MathVista is opportunistic (P3); the mitigation
  *locus* is dataset-independent by construction but a per-dataset
  `s*` recalibration remains untested.
