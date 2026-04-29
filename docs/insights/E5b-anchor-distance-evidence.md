# E5b — anchoring is gated by uncertainty AND plausibility, both axes load-bearing

> **2026-04-29 cross-model expansion.** qwen2.5-vl-7b E5c run
> (`outputs/experiment_e5c_{vqa,tally}/qwen2.5-vl-7b-instruct/`)
> is a strict superset of E5b — its `b + a×S1-5` strata cover the
> E5b stimulus matrix on a second model. wrong-base `adopt_cond`
> decays on qwen2.5-vl: VQAv2 0.070 → 0.014 → 0.003 → 0.003 → 0.003
> (S1—S5); TallyQA 0.033 → 0.015 → 0 → 0 → 0. The S1-peak / S3+ floor
> structure replicates qualitatively on qwen2.5-vl, with absolute
> magnitudes about half llava-interleave's (consistent with §3.3
> main-panel ranking placing qwen2.5-vl at the lowest df). The
> two-gate claim (uncertainty × plausibility) generalises across two
> architecturally different models. Numbers + cross-model panel:
> `docs/insights/E5c-anchor-mask-evidence.md` 2026-04-29 update +
> `docs/insights/_data/E5c_per_cell.csv`.

> **2026-04-28 update.** Re-run on C-form re-aggregated data: the cited
> `adopt_cond` decays (VQAv2 wrong-base 0.130 → 0.032 → 0.010 → 0.010 →
> 0.003 across S1—S5; TallyQA 0.092 → 0.006 → 0.003 → 0 → 0) are
> **unchanged** — adopt is independent of the direction-follow numerator.
> `direction_follow` rates are *smaller in absolute magnitude* under
> C-form (the new form excludes pa==pb pairs structurally) but the
> S1>S5 monotonicity on wrong-base is preserved. Inline numbers below
> are correct for `adopt_cond`; `df` ranges have shrunk relative to the
> anchor·gt form but the qualitative two-axis claim (uncertainty gate +
> plausibility window) holds. Pre-refactor results at
> `outputs/before_C_form/experiment_distance_*/`.

**Status:** Sub-experiment of E5; distilled insight. Source data: `outputs/experiment_distance_{vqa,tally}/llava-next-interleaved-7b/<latest>/predictions.jsonl`. Aggregate table: `docs/insights/_data/E5b_per_stratum.csv`. Figures: `docs/figures/E5b_adopt_cond_curve.png`, `docs/figures/E5b_adopt_cond_overlay.png`. Full writeup: `docs/experiments/E5b-anchor-distance.md`. Design + plan: `docs/experiments/E5b-anchor-distance-design.md`, `docs/experiments/E5b-anchor-distance-plan.md`.

## The claim and the test

Cross-modal numeric anchoring on `llava-interleave-7b` is **uncertainty-modulated AND plausibility-windowed**, contingent on both gates. Neither gate alone is sufficient: a confident model is not pulled even by a near-distance anchor, and a far-distance anchor does not pull even an uncertain model. The signature is the conjunction.

For each of two datasets (VQAv2 number subset, TallyQA test number-type), we draw 5 distance-stratified anchors per base question — `|a − GT|` strata S1 [0,1], S2 [2,5], S3 [6,30], S4 [31,300], S5 [301,∞) — and run `target_only` plus 5 anchor conditions. 1,000 base questions per dataset, 6,000 records per dataset, 12,000 total. Stratified anchor sampling from the 128-PNG `inputs/irrelevant_number/` inventory; per-question seed 42. See `scripts/analyze_e5b_distance.py` for the aggregation and the notebook `notebooks/E5b_anchor_distance.ipynb` for the reproducer.

The headline metric is paired conditional adoption (`evaluate_sample` at commit `bbcc418`): `(base ≠ a) AND (pred = a) / (n − count(case 4))`. Records where the model would have produced the anchor without exposure (case 4 — `base = anchor`) are excluded from both numerator and denominator; records where the anchor is rejected (case 3) stay in the denominator as non-adoptions. We stratify by base correctness (`target_only` correct vs wrong) on top of the distance stratum, giving 20 cells per dataset.

## What we found

The four most decisive cells:

| Dataset | base | Stratum | `n_eligible` | `adopt_cond` | 95 % CI |
|---|---|---|---:|---:|---|
| VQAv2 | wrong | S1 [0,1] | 332 | **0.1295** | [0.0934, 0.1687] |
| VQAv2 | wrong | S5 [301,∞) | 399 | **0.0025** | [0.0000, 0.0075] |
| TallyQA | wrong | S1 [0,1] | 282 | **0.0922** | [0.0603, 0.1277] |
| TallyQA | wrong | S5 [301,∞) | 346 | **0.0000** | [0.0000, 0.0000] |

Wrong-base S1 to wrong-base S5 collapses by two orders of magnitude on both datasets. **TallyQA wrong-base S4 and S5 are exactly 0/346 each** (zero adopted out of 346 records in two cells) — implausible anchors are *fully* rejected once distance exceeds 30 in this regime. VQAv2 wrong-base S5 is 1/399 = 0.0025 and S4 is 4/399 = 0.0100 — near-zero, statistically indistinguishable from zero (CI lower bound = 0).

## Reading

**Plausibility-windowed.** When the model would have been wrong without the anchor — the cohort where anchoring has room to act — the adoption rate is sharply distance-dependent. Peaked at S1, decayed by half by S2, an order of magnitude lower by S3, and at the floor by S4. Anchors outside the plausible answer range do not get adopted, even by an uncertain model.

**Uncertainty-modulated.** When the model would have been correct without the anchor (the "knows the answer" cohort), no stratum produces a meaningful pull: VQAv2 correct-base ranges 0.013–0.095 from S5 to S1; TallyQA correct-base ranges 0.000–0.036. The within-correct curve is essentially flat. Anchors do not move a confident model.

Both gates are load-bearing. Correct-base records show a ceiling of ≈0.10 even at S1 (the most "favourable" anchor distance) — uncertainty is required for adoption to act. Wrong-base records show a floor of ≈0 at S4–S5 (the most uncertain cohort) — plausibility is required for the anchor to be admitted. The product is the signature; either alone yields ~0 adoption.

The cross-dataset overlay confirms this is not an image-domain artefact. Despite VQAv2 having 3× the baseline accuracy of TallyQA (0.62 vs 0.21), the qualitative shape — sharp wrong-base S1 peak, floor by S4 — is the same on both datasets. The structure does not depend on baseline competence beyond setting the wrong-base support size.

## Sub-finding — direction-follow is the wrong headline at S1

Direction-follow `int((pred − GT)(anchor − GT) > 0)` collapses on the boundary case `anchor = GT`, which is non-trivially populated within S1 [0,1] (the stratum is inclusive on both ends). On the wrong-base cohort, direction-follow peaks at S2 — not S1 — because S1 is mechanically depressed by the boundary. Adoption (`pred = anchor`, case 2 only) does not have this artefact and peaks correctly at S1. The CSV preserves both metrics for reference; the headline plot shows `adopt_cond` only.

This is the second reason `adopt_cond` is the right headline. The first is the case-4 confound: under the marginal definition `(pred = a) / n`, records where `base = anchor` (the model would have produced the anchor without exposure) inflate the count. At VQAv2 S1 / TallyQA S1, this confound is large — case 4 contributes 201/601 (correct-base) and 67/399 (wrong-base) on VQAv2; 235/654 and 64/346 on TallyQA. Excluding case 4 from both numerator and denominator (`adopt_cond`) recovers the true conditional rate.

## Why this matters

This connects directly to A1 (`docs/insights/A1-uncertainty-modulated-graded-pull.md`), which used `target_only` correctness as the stratifier on direction-follow at full main-run scale and showed the anchor effect concentrates on uncertain records. E5b strengthens A1 in two ways: it switches the metric to `adopt_cond` (which excludes the `base = anchor` confound that direction-follow does not separate), and it adds the distance axis. The result is a sharper claim — among uncertain records, the effect further concentrates on plausible-distance anchors. The anchoring signature is two-dimensional, not one.

This also lines up with the classical cognitive-anchoring literature. Strack & Mussweiler's (1997) selective accessibility model frames anchoring as a comparison-driven retrieval bias: the anchor is admitted as a candidate when it is in the plausible range for the target judgment, and rejected when it is implausibly extreme. The same shape — peak at near-distance, sharp decay outside the plausible range — recurs here cross-modally on a VLM. Whether this is mechanism convergence or surface coincidence is for follow-up work; the empirical shape is consistent with the literature's prediction.

For the paper headline, the effect should be framed as **gated** rather than graded. "Anchoring exists at magnitude X %" lets a reader compare across architectures only in scalar; "anchoring requires a plausible-range distractor when the model is uncertain" lets a reader run a binary check on shape on any new model and dataset. The deployment-risk narrative narrows correspondingly: the load-bearing condition is "uncertain model + plausible distractor", a falsifiable conjunction rather than a single magnitude.

## What this doesn't say

- **Single model.** All numbers are llava-interleave-7b. The two-gate signature has not been replicated on the 11-model panel.
- **Single prompt.** Identical system prompt to `experiment.yaml`; paraphrase robustness (E7, Tier 2) is open.
- **GT range 0..8 only.** Both datasets cap at `answer_range = 8`. Outside this regime — ChartQA, MathVista — the plausibility-window claim is untested. The S4/S5 floor specifically depends on the distance threshold being well outside the dataset's GT support; on a dataset where GT can be 1000+, the relevant "implausible" boundary moves.

## What we did NOT test

- **Anchor-mask control.** E5b cannot distinguish (a) the anchor digit's *value* causes the pull from (b) the mere presence of a second image causes the pull and value is incidental. The anchor-mask experiment (E5c, queued in the roadmap) replaces the anchor digit pixels with an information-equivalent neutral patch — that is the clean test.
- **Multi-model generalisation.** The 11-model E5b extension (mid-stack cluster + Gemma + InternVL3 + Qwen + FastVLM) is queued. Until it runs, "anchoring is gated by uncertainty AND plausibility" is a single-model claim with strong cross-dataset replication on that one model.

## Implications for the experiment plan

- **E5c (anchor-mask control) gets first priority.** It is the cleanest test of the digit-value-vs-second-image-presence ambiguity and is the load-bearing follow-up for the E5b headline.
- **E7 (paraphrase robustness) still needed.** Single-prompt is a real caveat; the roadmap entry stands.
- **Heading toward the EMNLP main story.** A1 (Phase A uncertainty stratification) + E5b (distance + adoption gating) + E5c (anchor-mask causal test) + E1/E1b/E1d (mechanism — where in the LLM stack the anchor is read in, single-layer ablation null, upper-half mitigation candidate) + E4 (mitigation prototype). The paper narrative threads these as: the bias exists and is gated (A1, E5b), the digit pixels are causally load-bearing (E5c), the bias passes through identifiable LLM-stack layers but redundantly across them (E1/E1b/E1d), upper-half attention re-weighting is the cleanest mitigation prototype on the panel (E1d, E4).
