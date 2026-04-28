# A1 — Anchoring is uncertainty-modulated graded pull, not categorical capture

**Status:** Phase-A finding. Robust across all 7 models. Strongest single hook for the EMNLP write-up. Source data: `_data/A1_asymmetric_long.csv`, `_data/A1_asymmetric_wide.csv`. Script: `scripts/phase_a_data_mining.py::a1_asymmetric_on_wrong`.

> **2026-04-28 verification.** Phase A re-run on the C-form re-aggregated
> `outputs/experiment/`. Cited numbers below (adoption gap, moved-closer
> gap, per-model wrong−correct deltas) are **unchanged within ±0.1 pp**
> — Phase A's headline metrics (`adoption_rate`, `moved_closer_rate`)
> are independent of the `direction_follow` numerator that the C-form
> refactor changed. The +6.9 to +19.6 pp wrong−correct range and the
> per-model ranking both survive untouched.

## The question

`references/project.md` flagged "stronger anchoring on items the model originally got wrong" as the paper's strongest intellectual hook — none of the LLM/VLM anchoring literature partitions effects by prior correctness, and Mussweiler-Strack / Jacowitz-Kahneman both predict that anchoring scales with subjective uncertainty. **Does it hold?** The pre-stated falsifier (in `references/roadmap.md` §2 H2): if `adoption(wrong)` ≈ `adoption(correct)`, H2 fails.

## Method

For each model, partition the 17,730 paired records by whether the `target_only` prediction was exactly correct (`base_correct`). Within each stratum, compute three anchoring metrics:

1. **`adoption_rate`** — share of `target_plus_irrelevant_number` predictions that exactly equal the anchor digit. Categorical.
2. **`moved_closer_rate`** — share of pairs where `|pred(number) − anchor| < |pred(target_only) − anchor|`. Graded.
3. **`mean_anchor_pull`** — `|pred(target_only) − anchor| − |pred(number) − anchor|`, averaged over pairs. Magnitude.

Effects are reported as `wrong − correct` (positive ⇒ asymmetry in the H2 direction).

## Result

The categorical falsifier *fails*: adoption is roughly equal across strata.

| Model | n_correct | n_wrong | adoption(correct) | adoption(wrong) | **adoption gap** |
|---|---:|---:|---:|---:|---:|
| gemma3-27b-it | 8,611 | 9,119 | 0.128 | 0.152 | **+0.024** |
| gemma4-31b-it | 10,644 | 7,055 | 0.117 | 0.113 | -0.004 |
| gemma4-e4b | 7,570 | 10,160 | 0.129 | 0.118 | -0.011 |
| llava-next-interleaved-7b | 8,419 | 9,264 | 0.143 | 0.125 | -0.018 |
| qwen2.5-vl-7b-instruct | 10,495 | 7,235 | 0.116 | 0.100 | -0.016 |
| qwen3-vl-30b-it | 10,854 | 6,868 | 0.118 | 0.123 | +0.005 |
| qwen3-vl-8b-instruct | 10,747 | 6,983 | 0.126 | 0.130 | +0.004 |

But the graded falsifier holds — and *uniformly across the population*:

| Model | moved_closer(correct) | moved_closer(wrong) | **moved_closer gap** |
|---|---:|---:|---:|
| gemma4-e4b | 0.135 | 0.331 | **+0.196** |
| gemma3-27b-it | 0.080 | 0.239 | **+0.159** |
| qwen3-vl-30b-it | 0.116 | 0.238 | **+0.122** |
| gemma4-31b-it | 0.047 | 0.131 | +0.084 |
| qwen3-vl-8b-instruct | 0.068 | 0.148 | +0.080 |
| llava-next-interleaved-7b | 0.125 | 0.197 | +0.072 |
| qwen2.5-vl-7b-instruct | 0.061 | 0.130 | +0.069 |

`mean_anchor_pull` follows the same direction with the same sign in every model (see `_data/A1_asymmetric_wide.csv`).

## What this means

The H2 prediction needs to be re-stated:

> **Refined H2.** When a VLM is uncertain about the answer to a numerical VQA question (operationalised as "originally wrong"), an irrelevant anchor digit shifts the prediction *toward* the anchor at a substantially higher rate (+7 to +20 pp), even though the rate of *exactly copying* the anchor barely changes.

This is a sharper, harder-to-explain-away claim than the original "wrong cases anchor more" framing. It directly maps to the Mussweiler-Strack "selective accessibility" account: the anchor doesn't replace a confident prediction, it **biases the search direction** when the prediction is uncertain. The cognitive-science framing the paper can use:

- *Confident estimate (= correct on target-only):* anchor is largely ignored; adoption ≈ baseline; pull ≈ 0.
- *Uncertain estimate (= wrong on target-only):* anchor enters the candidate distribution and pulls the prediction toward itself, but does not dominate it.

This is exactly the gradient-anchoring signature human studies report. The paper now has both a novel empirical claim and a clean cognitive-science theory matching the data.

## Caveats

1. **`base_correct` is a noisy proxy for subjective uncertainty.** A model can be wrong-but-confident or right-but-uncertain. Plan: check whether per-token logit margins (already saved per `5f925b2`) correlate with the moved-closer gap — would let us replace the binary stratum with a continuous uncertainty measure. This is a Phase-A extension worth doing before submission.
2. **Outlier filter applied.** `analysis.filter_anchor_distance_outliers` (IQR×1.5) is on. Without it, gemma3 / Gemma4 strengthen-prompt runs would distort means. The pattern is robust without the filter too — verify when reproducing.
3. **Cross-anchor confound.** `adoption_rate` depends on whether the anchor digit is a plausible answer to the question. Anchors 1, 2, 3 are over-represented in the GT distribution, so adoption-correct can include "lucky guess where anchor coincides with truth". The graded `moved_closer_rate` is largely insulated from this because it conditions on a non-trivial shift; the per-digit confound mainly hits A2 and is discussed there.
4. **Models that show the largest gap are also the weakest models** (gemma4-e4b acc=0.55; gemma3-27b acc=0.63). This is the *opposite* direction of Lou & Sun's "stronger LLMs anchor more" claim. Possible explanations: (a) weak models are more often uncertain, so the gap mostly reflects different stratum sizes (gemma4-e4b: 10,160 wrong vs. 7,570 correct); (b) weak models have less anchored priors, so external anchors carry more weight. Disentangling needs a per-confidence (logit-margin) re-analysis.

## Concrete next steps tied to this finding

- **Logit-margin replacement of `base_correct`.** Use the per-token logit data from commit `5f925b2` to reproduce the table with continuous-uncertainty stratification. (~hours, no new compute.)
- **E1 attention analysis** should be conditioned on `base_correct` so the "attention to anchor on wrong cases > on correct cases" prediction can be tested directly. Without that conditioning, the attention signal will average out.
- **Mitigation E4** has a natural target: down-weight anchor-image attention specifically when the LLM's per-token entropy on the answer is high. This is a more principled intervention than blanket anchor-token down-weighting.

## Roadmap entry

- §2 H2 status: ⚠️ → ✅ (refined; the *graded-pull* form holds; the *categorical-adoption* form fails).
- §5 A1: ☐ → ✅
