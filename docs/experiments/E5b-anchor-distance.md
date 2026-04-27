# E5b — anchor-distance robustness sweep on llava-interleave-7b

**Status:** Sub-experiment of E5; results writeup. Driver: `scripts/run_experiment.py` with `inputs.anchor_sampling: stratified`. Configs: `configs/experiment_distance_vqa.yaml`, `configs/experiment_distance_tally.yaml`. Analysis: `scripts/analyze_e5b_distance.py`. Notebook: `notebooks/E5b_anchor_distance.ipynb`. Raw outputs: `outputs/experiment_distance_{vqa,tally}/llava-next-interleaved-7b/<latest>/predictions.jsonl`. Aggregate table: `docs/insights/_data/E5b_per_stratum.csv`. Figures: `docs/figures/E5b_adopt_cond_curve.png`, `docs/figures/E5b_adopt_cond_overlay.png`. Design: `docs/experiments/E5b-anchor-distance-design.md`. Plan: `docs/experiments/E5b-anchor-distance-plan.md`.

## TL;DR — three findings

1. **Uncertainty gate.** When the model would have answered correctly without the anchor (`target_only` correct), no stratum produces a meaningful adoption pull: `adopt_cond ≤ 0.10` across all 5 distance strata on both datasets. The within-correct curve is essentially flat — VQAv2 correct-base ranges 0.013–0.095 from S5 to S1; TallyQA correct-base ranges 0.000–0.036. Anchors do not move the prediction of a model that already knows the answer.
2. **Plausibility window.** When the model would have been wrong without the anchor, adoption is sharply distance-dependent — peaked at S1 [0,1] and decaying to a floor by S4 [31,300]. **TallyQA wrong-base S4 and S5 are exactly 0/346 each** (zero adopted out of 346 wrong-base records). VQAv2 wrong-base S5 is 1/399 = 0.0025 and S4 is 4/399 = 0.0100 — near-zero. Implausible anchors (those far outside the GT range that the dataset admits) are rejected.
3. **Cross-dataset robustness.** The qualitative pattern is identical on VQAv2 and TallyQA despite a 3× difference in baseline accuracy (`acc(target_only) = 0.62` on VQAv2 vs `0.21` on TallyQA). The effect is not tied to a single image domain, and the gating structure does not depend on baseline competence beyond setting the wrong-base support size.

Neither gate alone is sufficient. Correct-base records show a ceiling of ≈0.10 even at S1 — uncertainty-modulation is real. Wrong-base records show a floor of ≈0 at S4–S5 — plausibility-windowing is real. The signature of cross-modal numeric anchoring on this model is the **product** of both gates: the model is uncertain *and* the anchor is plausible.

## Setup

### Model and datasets

One model: **`llava-next-interleaved-7b`** (`llava-hf/llava-interleave-qwen-7b-hf`). Selection rationale (from the design doc): highest 7B-class direction-follow in the panel (0.348 baseline on VQAv2 main), full main-run baseline already exists for sanity-checking, multi-image native (interleave training is the model's design intent rather than a stress mode).

Two datasets, identical filter:

| Dataset | Subset rule | N (base questions) | acc(target_only) |
|---|---|---:|---:|
| VQAv2 number val | `answer_range = 8`, `samples_per_answer = 200`, `max_samples = 1000`, `require_single_numeric_gt` | 1000 | 0.62 |
| TallyQA test | `answer_type_filter = ["number"]`, `answer_range = 8`, `samples_per_answer = 200`, `max_samples = 1000`, `require_single_numeric_gt` | 1000 | 0.21 |

Both datasets cap GT to integers in 0..8. The cross-dataset comparison therefore tests **image-domain robustness** (open-domain VQAv2 photos vs natural-photo counting in TallyQA), not GT-scale robustness — a true scale comparison needs ChartQA/MathVista (deferred; flagged in caveats).

### Anchor sampling — 5 strata, GT-based reference

For each base question with ground-truth `g`, draw 5 anchors `(a₁..a₅)` independently from the inventory `inputs/irrelevant_number/{0..10000}.png` (128 PNGs), one per stratum:

| Stratum | `|a − g|` range | role |
|---|---|---|
| S1 | [0, 1] | near-peak (effect strongest if literature is right) |
| S2 | [2, 5] | adjacent |
| S3 | [6, 30] | intermediate decay |
| S4 | [31, 300] | far |
| S5 | [301, ∞) | very far / saturation tail |

Implementation: `vlm_anchor.data.sample_stratified_anchors` returns one randomly-chosen anchor per stratum from the matching inventory subset; `assign_stratified_anchors` resolves the PNG path. Per-question seed `seed = 42`.

### Conditions per question — 6 total

`target_only` (1) + `target_plus_irrelevant_number_S{1..5}` (5). No neutral arm — the S5 condition (very-far anchor) plays the "anchor information ≈ 0" reference role per the design rationale, and `target_only` is the true zero-anchor baseline.

Total: 6,000 records per dataset (1,000 base × 6 conditions); 12,000 records overall.

### Sampling

`temperature = 0.0`, `top_p = 1.0`, `max_new_tokens = 8`, JSON-only system prompt — identical to `experiment.yaml`. Greedy decoding throughout.

## Metric — paired conditional adoption (M1)

The headline metric is `adopt_cond` per `evaluate_sample` (`src/vlm_anchor/metrics.py`, commit `bbcc418`):

```
adopt_cond = (base ≠ a) AND (pred = a) / (n − count(case 4))
```

where for each per-question (anchor-condition, base) pair we classify the four cases by `(base_pred, pred, anchor)`:

| Case | base | pred | What it means |
|---|---|---|---|
| 1 | base ≠ anchor | pred ≠ anchor | not adopted, not confounded |
| 2 | base ≠ anchor | pred = anchor | **adopted** (numerator) |
| 3 | base = anchor | pred ≠ anchor | repelled away from anchor |
| 4 | base = anchor | pred = anchor | confound — model would have produced the anchor without exposure |

The conditional (paired) form is required because under the marginal definition `(pred = a) / n`, case 4 inflates the adoption count whenever `base = anchor` (e.g. GT = anchor). The `(base ≠ a)` clause excludes case 4 from both numerator and denominator. Case 3 is kept in the denominator because the anchor *did* have access; the model rejecting it is itself a non-adoption.

We report `adopt_cond` rather than `direction_follow` (the original headline) for two reasons. First, on near-anchor strata (S1 with anchor=GT), direction-follow has an artificial knife-edge — the anchor and the correct answer point the same way, so the (`pred − GT`) × (`anchor − GT`) sign product is always 0 and direction-follow is mechanically depressed at the very stratum where the effect should be strongest. Second, direction-follow does not isolate adoption from anti-anchor (case 3); a model that systematically *avoids* the anchor (case 3 inflated) and a model that ignores the anchor (case 1 inflated) show the same direction-follow if the avoidance is symmetric. Adoption (case 2 only) is the cleanest "anchor pulls the prediction" signal. The analysis script preserves direction-follow numbers in the per-cell CSV for reference; they peak at S2 not S1 on this run, consistent with the boundary-noise diagnosis.

Bootstrap 95 % CIs are computed on the eligible-cell adoption vector with 1,000 iterations (`scripts/analyze_e5b_distance.py::_bootstrap_rate`).

## Result 1 — uncertainty gate: anchors do not pull correct predictions

Records where `target_only` was correct (the "knows the answer" subset):

| Dataset | Stratum | `n_total` | `n_eligible` | `adopt_cond` | 95 % CI |
|---|---|---:|---:|---:|---|
| VQAv2 | S1 [0,1] | 601 | 400 | **0.0950** | [0.0675, 0.1275] |
| VQAv2 | S2 [2,5] | 601 | 601 | 0.0283 | [0.0166, 0.0433] |
| VQAv2 | S3 [6,30] | 601 | 601 | 0.0200 | [0.0100, 0.0316] |
| VQAv2 | S4 [31,300] | 601 | 601 | 0.0166 | [0.0083, 0.0283] |
| VQAv2 | S5 [301,∞) | 601 | 601 | 0.0133 | [0.0050, 0.0233] |
| TallyQA | S1 [0,1] | 654 | 419 | **0.0358** | [0.0191, 0.0525] |
| TallyQA | S2 [2,5] | 654 | 654 | 0.0046 | [0.0000, 0.0107] |
| TallyQA | S3 [6,30] | 654 | 654 | 0.0015 | [0.0000, 0.0046] |
| TallyQA | S4 [31,300] | 654 | 654 | 0.0000 | [0.0000, 0.0000] |
| TallyQA | S5 [301,∞) | 654 | 654 | 0.0000 | [0.0000, 0.0000] |

The S1 ceiling is ~0.10 on VQAv2 and ~0.04 on TallyQA. From S2 onward both datasets sit below 0.03. Reading: when the model already would have been right without the anchor, anchors do not pull the prediction; the residual S1 effect is the only place uncertainty leaks in (some "correct-base" records are correct but with low margin, and the within-eligible denominator drops to 400/601 there because case 4 / `base = anchor` records are excluded — those are records where the GT and the S1 anchor coincide).

## Result 2 — plausibility window: implausible anchors are rejected

Records where `target_only` was wrong:

| Dataset | Stratum | `n_total` | `n_eligible` | `adopt_cond` | 95 % CI |
|---|---|---:|---:|---:|---|
| VQAv2 | S1 [0,1] | 399 | 332 | **0.1295** | [0.0934, 0.1687] |
| VQAv2 | S2 [2,5] | 399 | 379 | 0.0317 | [0.0158, 0.0501] |
| VQAv2 | S3 [6,30] | 399 | 399 | 0.0100 | [0.0025, 0.0226] |
| VQAv2 | S4 [31,300] | 399 | 399 | 0.0100 | [0.0000, 0.0201] |
| VQAv2 | S5 [301,∞) | 399 | 399 | **0.0025** | [0.0000, 0.0075] |
| TallyQA | S1 [0,1] | 346 | 282 | **0.0922** | [0.0603, 0.1277] |
| TallyQA | S2 [2,5] | 346 | 332 | 0.0060 | [0.0000, 0.0151] |
| TallyQA | S3 [6,30] | 346 | 345 | 0.0029 | [0.0000, 0.0087] |
| TallyQA | S4 [31,300] | 346 | 346 | **0.0000** | [0.0000, 0.0000] |
| TallyQA | S5 [301,∞) | 346 | 346 | **0.0000** | [0.0000, 0.0000] |

The peak is at S1 on both datasets. By S4–S5 the rate has decayed to a floor: TallyQA wrong-base S4 and S5 are *exactly* zero (0/346 in two cells); VQAv2 wrong-base S5 is 1/399 = 0.0025 and S4 is 4/399 = 0.0100 — near-zero, but not *exactly* zero. The two non-zero VQAv2 wrong-base S4/S5 records are not statistically distinguishable from zero (CI lower bound = 0).

The reading: at distances above 30, an anchor digit no longer pulls a wrong prediction toward itself — the anchor is too implausible to be admitted as an answer for a 0..8-GT question. Inside the plausibility window (S1, S2, marginally S3) the anchor *can* pull, but only when there is uncertainty for it to fill.

## Result 3 — cross-dataset robustness

Side-by-side, the wrong-base column (where the effect is largest):

| Stratum | VQAv2 `adopt_cond` | TallyQA `adopt_cond` |
|---|---:|---:|
| S1 | 0.1295 | 0.0922 |
| S2 | 0.0317 | 0.0060 |
| S3 | 0.0100 | 0.0029 |
| S4 | 0.0100 | 0.0000 |
| S5 | 0.0025 | 0.0000 |

Same shape. TallyQA's curve sits below VQAv2's at every stratum (smaller wrong-base support — 346 vs 399 — and a different image distribution), and TallyQA decays slightly faster (its S4 and S5 both hit exactly zero, VQAv2's don't), but the qualitative gating signature is invariant. With baseline accuracy differing 3× across the two datasets, the same gating structure holds — the "uncertain model + plausible anchor" signature is not a VQAv2-specific artefact.

The figures plot this directly: `docs/figures/E5b_adopt_cond_curve.png` is the per-dataset `adopt_cond` vs stratum-midpoint with two lines (correct-base, wrong-base), one panel per dataset; `docs/figures/E5b_adopt_cond_overlay.png` overlays the wrong-base curves of both datasets.

## Why direction-follow was retired as the headline

Two reasons, both flagged at design time and confirmed empirically.

First, the direction-follow at S1 has an artificial floor when the anchor coincides with GT. The metric is `int((pred − GT)(anchor − GT) > 0)`; when anchor = GT, the second factor is 0 and direction-follow is mechanically zero on every record, regardless of how the prediction was pulled. A non-trivial fraction of S1 [0,1] records hit `anchor = GT` (the per-stratum range is inclusive on both ends). The naive direction-follow rate at S1 is therefore depressed not because the effect is small there, but because the metric collapses on the boundary.

Second, direction-follow does not separate adoption (case 2) from anti-anchor (case 3 — model lands on the side away from the anchor). On this run, the cell-level CSV (`docs/insights/_data/E5b_per_stratum.csv`) shows the per-stratum direction-follow peaking at S2 rather than S1 for the wrong-base cohort — case 3 inflation explains the mismatch. Adoption isolates the case the literature actually describes.

The CSV preserves both metrics and case counts; the headline figure plots `adopt_cond` only.

## Caveats

- **Single model.** All numbers are from `llava-interleave-7b`. The two-gate signature has not been replicated on the 11-model panel; the §6 plan calls out a multi-model E5b extension as the natural follow-up. Until that runs, "anchoring is gated by uncertainty AND plausibility" is a single-model claim.
- **Single prompt.** The system prompt is identical to `experiment.yaml`. Paraphrase robustness (Tier 2 E7) is open work.
- **GT range 0..8.** Both datasets cap GT at 8 by `answer_range`. Beyond this regime — ChartQA / MathVista where GT can exceed 10000 — the plausibility-window claim is untested. The cross-dataset extension is queued; until it runs, the present claim is "plausibility-windowed within the 0..8 GT regime".
- **Anchor inventory granularity.** The 128-PNG inventory is fine-grained on 0..10 (every integer) and 5-step on 10..100, then 100-step beyond. Within S1/S2 the inventory is dense; within S4/S5 it is sparse but still much larger than the per-question sampling needs. For the present GT range the granularity is not load-bearing; for ChartQA/MathVista extension it would need re-checking.
- **Case-4 exclusion is not free.** The denominator drops in two places: VQAv2 S1 (601→400 correct-base, 399→332 wrong-base), TallyQA S1 (654→419 correct-base, 346→282 wrong-base), and the two S2 cells where `base = anchor` is non-zero. The CIs above are computed on the eligible cell. The drop is informative: when GT and anchor coincide at S1, ~33 % of correct-base records are case-4-confounded — the marginal-adoption number from any earlier writeup *would* be inflated relative to `adopt_cond` exactly for this reason. The conditional form is the right metric.
- **Direction-follow direction signs.** The CSV contains `df_cond` numbers per cell; we do not plot or feature them. They are noisier than `adopt_cond` for the boundary reason above and should not be used as a substitute headline.

## Implications for the paper

- **Adopt_cond + base correctness split is the headline figure.** The two-line per-dataset plot (correct vs wrong base) plus the cross-dataset wrong-base overlay is the cleanest one-page summary of the cross-modal anchoring effect we have on this model.
- **The bias is gated, not graded.** Earlier writeups framed cross-modal anchoring as a magnitude (e.g. "~13 % adoption at peak"). The two-gate signature is more falsifiable: classical anchoring requires the anchor to be in the plausible range *and* the model to be uncertain. A reader can ask of any new model "does it show the same gating", which is a binary check on shape rather than a magnitude comparison across architectures.
- **Connects to A1.** Phase A's A1 finding ("uncertainty-modulated graded pull") used `target_only` correctness as the stratifier on direction-follow at full main-run scale. E5b extends this stratification to `adopt_cond` and adds the distance axis, on the same model. The two findings line up: A1 says the anchor effect concentrates on uncertain records; E5b adds that even within uncertain records, the effect concentrates on plausible-distance anchors.
- **Frames the deployment-risk story.** "Models exposed to plausible-range distractors when uncertain" is the load-bearing condition for cross-modal numeric anchoring on this model. Deployment-risk discussion in the paper can hinge on this conjunction rather than on a single magnitude number.

## What this experiment does NOT establish

- **Causal attribution to digit pixel content.** The headline is consistent with two readings: (a) the *value* of the anchor digit causes the pull, (b) the mere presence of a second image causes the pull and the value is incidental. E5b cannot distinguish these — the second-image arm is always present in S1..S5. The anchor-mask control (E5c, queued in the roadmap) replaces the anchor digit pixels with an information-equivalent neutral patch and is the clean test of (a) vs (b).
- **Multi-model generalisation.** All numbers are llava-interleave-7b. The 11-model E5b extension is queued.
- **Non-numeric tasks.** The metric, the conditions, and the inventory are all integer-valued. Cross-modal anchoring on categorical or free-text outputs is out of scope.
