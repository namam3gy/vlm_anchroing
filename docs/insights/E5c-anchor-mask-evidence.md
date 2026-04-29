# E5c — the digit pixels are the operative cause; the anchor scene is not

> **2026-04-29 update — qwen2.5-vl-7b cross-model expansion.** A second
> model (qwen2.5-vl-7b-instruct) ran on the same E5c stimulus matrix
> (n=12,000 records / dataset). The **digit-pixel gap is at floor on
> qwen2.5-vl** on both datasets (VQAv2 wrong-base S1 a−m = +0.4 pp,
> TallyQA = −0.5 pp), consistent with §3.3 main panel placing
> qwen2.5-vl as the most-anchor-resistant model (`adopt(a) = 0.021`,
> `df(a) = 0.094` on the 17,730-sample VQAv2 panel). The
> direction-consistent picture — "where the pull is large enough to
> detect, the gap is positive; where the pull is at floor, the gap is
> also at floor" — preserves the §5.4 digit-pixel-causality claim on
> the model where the effect is detectable but qualifies the
> cross-model generalisation. Pending gemma3-27b-it E5c VQAv2 cell
> (~5–6h H200) will arbitrate whether mid-panel models track the
> llava-style detectable gap or the qwen-style floor.

> **2026-04-28 update.** Re-run on C-form re-aggregated data: the cited
> `adopt_cond` digit-pixel gaps (VQAv2 wrong-base S1 +6.1 pp,
> TallyQA wrong-base S1 +2.5 pp) are **unchanged** — adopt is
> independent of the direction-follow numerator. Under C-form,
> `df_cond` for anchor on VQAv2 wrong-base S1 is 0.208 (was ~0.520
> under anchor·gt form) and for masked is 0.155, giving a +5.3 pp
> df-gap (under anchor·gt form, df_cond was nearly identical between
> anchor and masked, masking the digit-pixel df contribution; under
> C-form a df-gap also emerges). Pre-refactor results archived at
> `outputs/before_C_form/experiment_e5c_*/`.

**Status:** Sub-experiment of E5; distilled insight. Source data: `outputs/experiment_e5c_{vqa,tally}/{llava-next-interleaved-7b,qwen2.5-vl-7b-instruct}/<latest>/predictions.jsonl`. Aggregate table: `docs/insights/_data/E5c_per_cell.csv` (now keyed by `model`). Figures: `docs/figures/E5c_anchor_vs_masked_adopt[_<model>].png`, `docs/figures/E5c_anchor_vs_masked_df[_<model>].png`, `docs/figures/E5c_acc_drop_3way[_<model>].png`, `docs/figures/E5c_correct_vs_wrong_adopt[_<model>].png` (default model = llava-next-interleaved-7b; non-default models append `_<model>` to the filename). Full writeup: `docs/experiments/E5c-anchor-mask-control.md`.

## The claim and the test

The anchor digit pixels themselves are the cause of paired adoption on `llava-interleave-7b`. The anchor image's *background scene* is not. Replacing the digit pixel region of each anchor image with a content-preserving inpaint (Telea, OpenCV) — same scene, digit removed — reduces the anchor's effect on `adopt_cond` measurably (wrong-base × S1 paired adoption drops from 0.1292 to 0.0681 on VQAv2, from 0.1095 to 0.0842 on TallyQA), and reduces the anchor's effect on `acc_drop` to the level of a generic neutral distractor (correct-base masked S1 acc_drop = 0.0754, neutral = 0.0882, gap 1.3 pp on VQAv2; 0.0250 vs 0.0449 on TallyQA, gap 2.0 pp). The pure digit-pixel contribution to paired conditional adoption is **+6.1 pp on VQAv2 wrong-base S1** and decays monotonically to ~0 by S5.

The test is a 12-condition design per question: `target_only` (1) + `target_plus_irrelevant_number_S{1..5}` (5 anchor strata, digit visible) + `target_plus_irrelevant_number_masked_S{1..5}` (5 strata, digit pixels inpainted, scene preserved) + `target_plus_irrelevant_neutral` (1 generic neutral distractor). 1000 base questions per dataset (VQAv2 number subset + TallyQA test number-type), 12,000 records per dataset, 24,000 records overall. Same model and same stratified sampling as E5b; same per-question seed; same JSON-only system prompt; greedy decoding.

The masked anchor inventory is generated once and re-used: `inputs/irrelevant_number_masked/{0..10000}.png`, 128 PNGs, OCR-validated to confirm the digit is no longer detectable post-inpaint. The mask region is the dilated bounding box of the digit (PaddleOCR-detected, with synthetic-bbox fallback for items where OCR fails). The scene background is preserved by Telea inpainting; only the digit pixels are replaced.

Three metrics are reported per cell: paired conditional `adopt_cond` (M1; case 4 excluded from denominator), `df_cond` (direction-follow with the same case-4 exclusion plus `anchor != gt`), and `acc_drop` (target_only baseline minus cell accuracy on the same base subset). The full per-cell table is `docs/insights/_data/E5c_per_cell.csv` (66 rows: 2 datasets × 3 base subsets × 11 cells per base).

## What we found

The most decisive cells:

| Dataset | base | Cell | `n` | `adopt_cond` | `acc_drop` |
|---|---|---|---:|---:|---:|
| VQAv2 | wrong | anchor S1 | 399 | **0.1292** | -0.1053 (regression-to-mean) |
| VQAv2 | wrong | masked S1 | 399 | **0.0681** | -0.0852 |
| VQAv2 | correct | anchor S1 | 601 | 0.0898 | 0.0932 |
| VQAv2 | correct | masked S1 | 601 | 0.0500 | 0.0754 |
| VQAv2 | correct | neutral | 601 | n/a (no anchor) | **0.0882** |
| TallyQA | wrong | anchor S1 | 346 | **0.1095** | -0.0462 |
| TallyQA | wrong | masked S1 | 346 | **0.0842** | -0.0414 |

Anchor − masked `adopt_cond` gap on wrong-base × S1 = **+0.0611 (VQAv2)** and **+0.0253 (TallyQA)**. The gap decays monotonically with distance: VQAv2 wrong-base gap is 0.0611 → 0.0158 → 0.0126 → 0.0125 → 0.0075 across S1→S5; TallyQA wrong-base gap is essentially 0 by S2.

`acc_drop` comparison on correct-base S1: masked (0.0754 / 0.0250) sits within 2 pp of neutral (0.0882 / 0.0449) on both datasets; anchor (0.0932 / 0.0270) is at most 1.5 pp above masked. The diffusion-generated background of the anchor scene contributes nothing measurable beyond what a generic neutral image would contribute.

## Reading

**Reading 1 — digit-pixel causality.** The (1, 2, 3) comparison is `target_only`, `target+anchor`, `target+masked`. Because the masked image is the same scene with only the digit pixels removed, the anchor − masked `adopt_cond` gap quantifies the pure digit-pixel contribution to paired adoption. On wrong-base × S1 — the cohort and stratum where E5b shows the peak anchor pull — this gap is +6.1 pp on VQAv2 and +2.5 pp on TallyQA, with both gaps decaying as the anchor moves further from GT. The digit pixels are operative.

**Reading 2 — anchor background ≈ generic neutral.** The (1, 3, 4) comparison is `target_only`, `target+masked`, `target+neutral`. Both treatments preserve "second image present" and remove "digit visible". They differ only in the *content* of the background — masked is the digit-removed anchor scene, neutral is an unrelated photograph. On the correct-base subset (where `acc_drop` is interpretable), masked and neutral hurt accuracy by indistinguishable amounts. The anchor image's background is doing *no work* beyond what a generic 2-image distractor does.

Both readings hold. The first one rules out "second image presence is enough"; the second one rules out "the anchor scene's background is the carrier". By elimination, the digit pixels themselves are the causal pathway for paired adoption.

## Sub-finding — direction-follow does not isolate the digit

`df_cond` on wrong-base records is essentially identical between anchor and masked at every stratum. VQAv2 wrong-base S2: anchor 0.5198, masked 0.5223 (gap −0.25 pp, masked actually slightly higher). TallyQA wrong-base S1: anchor 0.2840, masked 0.2683 (gap +1.6 pp, smaller than the `adopt_cond` gap of +2.5 pp on the same cell). The model's prediction drifts toward the anchor's directional half-plane whenever a second image is present and the model is uncertain — the digit pixels do not noticeably amplify this directional drift.

This is the second nail in direction-follow's coffin as a headline metric on this dataset and model. E5b retired direction-follow because of the boundary-noise issue at S1 (`anchor = GT` mechanically zeros direction-follow). E5c adds a content-causality argument: even at strata where boundary noise is not a problem, direction-follow tracks generic uncertainty-driven perturbation, not digit-specific signal. `adopt_cond` is the right metric for digit-pixel causality.

## Why this matters

This is the load-bearing follow-up to E5b. E5b established that paired anchor adoption is uncertainty-modulated AND plausibility-windowed; E5c establishes that the operative cause within the plausibility window is the digit pixels themselves. Without E5c, the E5b headline is consistent with "the second image perturbs the prediction at a plausible distance regardless of content"; with E5c, the second-image-presence reading is ruled out and the digit-value reading is established.

For the paper, this lets the deployment-risk story narrow further. The load-bearing condition for cross-modal numeric anchoring on this model is the conjunction "uncertain model + plausible distance + visible digit pixels". A reader can ask of any new model whether all three gates are required, and each is a binary check on a per-condition cell rather than a magnitude comparison across architectures.

This connects to two upcoming experiments:

- **E1d's upper-half attention re-weighting target on the mid-stack cluster.** E5c says the digit pixels are causally load-bearing; E1d says the mid-stack attention pathway is causally load-bearing for the direction-follow signal. The next mechanistic question is whether the same pathway carries the digit-specific paired-adoption signal that E5c isolates. A region-aware attention re-analysis (already flagged in roadmap §10 as a follow-up to E5 ChartQA) — measuring attention to the digit-pixel patches specifically vs the rest of the anchor image — would close this loop.
- **The 11-model E5c extension.** E5c is currently a 2-model finding (llava-interleave-7b detectable, qwen2.5-vl-7b at floor) with gemma3-27b-it pending. Once gemma lands, the 3-model E5e panel + the qwen-vs-llava direction-consistent reading suggest the cross-model story will be "the digit-pixel gap tracks the §3.3 main-panel anchor-pull magnitude" rather than "the digit-pixel mechanism flips along encoder-architecture axes". Replicating across the full 11-model panel would harden this and let us bin models by whether they sit above the noise-floor cutoff.

## What this doesn't say

- **Single model.** All numbers are llava-interleave-7b. The two-axis (digit-pixel + anchor-scene) decomposition has not been replicated on the 11-model panel.
- **Single prompt.** Paraphrase robustness (E7) is open.
- **Anchor scene is FLUX-generated with digit-mentioning prompts.** The masked images preserve the scene; if FLUX systematically encodes digit-specific scene priors (e.g. specific colours per digit), the masked condition would carry residual digit signal. The conservative reading "anchor scene ≈ neutral on accuracy" is robust to this concern; a stronger reading would need a scene-decorrelated regenerate.

## What we did NOT test

- **Scene-decorrelated regenerate.** The masked anchor inventory is the digit-visible inventory with digit pixels removed. A stronger control would re-generate the anchor scene from a digit-free prompt entirely. This would tighten the (1, 3, 4) "background carries no information" claim further. Not done in this experiment; flagged as a caveat.
- **Multi-model E5c.** Queued. Would tighten the cross-architecture pure-digit-pixel claim.

## Implications for the experiment plan

- **E6 (closed-model subset) gets clearer framing.** With E5b's gating and E5c's digit-pixel causality both established on the open-model side, the closed-model question narrows to "does GPT-4o / Gemini-2.5 show the same gating + digit-pixel causality, or only one of them?". The two-test decomposition is more falsifiable than a single magnitude check.
- **E7 (paraphrase robustness) still needed.** Single-prompt is a real caveat. Both E5b and E5c results would be stronger with paraphrase replication.
- **Multi-model E5c extension is the natural next step on the bias-mechanism arc.** The 11-model panel for E5c (running the same 12-condition design) would let us check whether the digit-pixel causality holds across the 4 archetypes from E1b (SigLIP-Gemma early, mid-stack cluster, Qwen-late, FastVLM-FastViT). The mid-stack cluster is the highest-leverage target — three encoders, one shared mechanism candidate from E1d.
