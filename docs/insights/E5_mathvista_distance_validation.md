# E5d — MathVista distance-validation: cutoff decision

**Status:** Per-dataset validation per `docs/insights/E5_anchor_distance_judgment.md` step 4. Source data: `outputs/experiment_e5d_mathvista_validation/llava-next-interleaved-7b/20260427-162235/predictions.jsonl`. Aggregate table: `docs/insights/_data/E5d_mathvista_per_stratum.csv`. Figure: `docs/figures/E5d_mathvista_decay.png`. Driver / config / analysis script committed in this commit.

## TL;DR

MathVista wrong-base paired conditional adoption stays roughly flat from `0.110` at S1 (≤10 % · GT) to `0.051` at S5 (>300 % · GT). Two of three acceptance criteria pass; **C3 fails decisively** — the noise floor is never reached. The qualitative pattern diverges from ChartQA: instead of decay-to-zero in a "plausibility window," MathVista shows diffuse low-magnitude adoption at every distance.

- **C1 (monotonic decay):** PASS — one soft inversion (S2 → S3, +0.039) with bootstrap-CI overlap; no hard inversions.
- **C2 (S1 effect size ≥ 0.05):** PASS — S1 wrong-base adopt_cond = **0.110** (above ChartQA S1 = 0.056).
- **C3 (S4 or S5 noise floor ≤ 0.01):** **FAIL** — S4 = **0.076**, S5 = **0.051**. Both 5×–8× above the threshold.

A GT-floor diagnostic (re-stratify to GT ≥ 50; n_base = 37) confirms the diffuse pattern is **not** a small-GT artefact: S5 stays at 0.038 (still 4× the noise threshold). C3 failure is a real property of the model's behaviour on MathVista, not a stratification glitch.

**Cutoff decision: do not adopt the ChartQA "S1 only" rule for MathVista.** No anchor-distance cutoff in [0, ∞) gives a clean isolated effect on this dataset. The recommendation is to **scope MathVista out of paper-canonical anchor claims for now** and queue a follow-up at n = 500 with a tighter dataset filter (GT ≥ 10 plus stricter `require_single_numeric_gt`), or document the divergent pattern as the finding itself.

## Setup

- Dataset: MathVista testmini, restricted to **integer-answer items with single-numeric GT in [1, 1000]** (`answer_type_filter = ["integer"]`, `answer_range = 1000`, `samples_per_answer = 5`, `max_samples = 200`, `require_single_numeric_gt = true`). Loader's integer-GT gate further filters; final base set = **153 questions** (cap is loader-bound, not the `max_samples = 200` setting).
- Unique GT values: 61 (vs ChartQA's 200 with the same `samples_per_answer = 5` cap binding less tightly). MathVista's integer-answer pool is small and skewed: median GT = 16, 55 / 153 questions have GT ≤ 10.
- Records: 900 = 153 base + 747 stratified anchor rows. (Expected 153 × 6 = 918; 18 stratified rows dropped where the FLUX inventory had no anchor in the per-question stratum range — concentrated in S1 for tiny GTs where `[0, 1]` only has anchors {0, 1}.)
- Model: llava-interleave-7b (matches E5b/E5c/E5d-ChartQA).
- Anchor sampling: 5-stratum hybrid scale-relative cutoff via `compute_strata(gt, scheme="relative")` (same as ChartQA).
- Sampling: `temperature = 0.0`, `top_p = 1.0`, `max_new_tokens = 16`.
- Wall: ~2 minutes on H200 (1.2 it/s for 153 base questions). MathVista images render small — wall is dominated by encode/generate, not image preprocessing. Far cheaper per question than ChartQA (~5.4 s/it).

## Decay curve (wrong-base subset)

| Stratum | range (gt-relative) | n_total | case1 | case2 | case3 | case4 | n_elig | adopt_cond | 95 % CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | ≤10 % | 105 | 87 | 11 | 2 | 5 | 100 | **0.1100** | [0.050, 0.170] |
| S2 | ≤30 % | 117 | 104 | 7 | 0 | 6 | 111 | 0.0631 | [0.027, 0.117] |
| S3 | ≤100 % | 118 | 104 | 12 | 1 | 1 | 117 | 0.1026 | [0.051, 0.162] |
| S4 | ≤300 % | 118 | 109 | 9 | 0 | 0 | 118 | **0.0763** | [0.034, 0.127] |
| S5 | >300 % | 118 | 112 | 6 | 0 | 0 | 118 | **0.0508** | [0.017, 0.093] |

Correct-base subset (n_total = 31–35 per cell): adopt_cond ranges 0.000 (S5) to 0.095 (S1) — also non-zero, but small numbers per cell (n_eligible 21–35) make this noisy. The qualitative wrong-base-only effect from E5b/E5c is muddier on MathVista.

### GT-floor diagnostic (sanity check on the C3 failure)

Re-stratify the wrong-base subset by minimum GT, asking whether the diffuse pattern is driven by very small GTs whose absolute distance bands are tiny:

| GT floor | n_base_questions | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| (no floor) | 153 | 0.110 | 0.063 | 0.103 | 0.076 | 0.051 |
| ≥ 10 | 103 | 0.145 | 0.073 | 0.105 | 0.079 | 0.053 |
| ≥ 30 | 50 | 0.167 | 0.121 | 0.132 | 0.158 | 0.079 |
| ≥ 50 | 37 | 0.167 | 0.087 | 0.115 | 0.115 | 0.038 |
| ≥ 100 | 16 | 0.200 | 0.100 | 0.167 | 0.167 | 0.083 |

**Verdict: the diffuse pattern is robust.** Filtering to GT ≥ 50 still produces S5 = 0.038, which would still fail C3 (0.038 > 0.01). The wide-distance adoption is genuine model behaviour, not a small-GT artefact. ChartQA's clean decay does not generalise.

## Acceptance criteria — verdicts

- **C1 — monotonic decay (wrong-base): PASS.**
  - Direction: S1 (0.110) > S5 (0.051). Endpoints decrease.
  - One soft inversion: S2 (0.063) → S3 (0.103), +0.039. Bootstrap CIs overlap. Zero hard inversions. Within the analysis script's "≤ 1 inversion + CI overlap" tolerance.
  - Interpretation: weakly monotonic, but the slope is shallow (S1/S5 ≈ 2.2× vs ChartQA S1/S5 ≈ 3.6× with a true zero in between).
- **C2 — S1 effect size ≥ 0.05: PASS.** S1 = 0.110, more than 2× the threshold and ~2× ChartQA's S1 = 0.056. Effect is present and detectable.
- **C3 — S4 or S5 ≤ 0.01: FAIL.** S4 = 0.076, S5 = 0.051. Both are 5×–8× above the noise floor. Re-checking under GT ≥ 50 still gives S5 = 0.038 (4× threshold). No anchor-distance regime within `[0, ∞)` reaches the noise floor.

## Cutoff decision

**No clean cutoff exists for MathVista.** The wrong-base anchor effect on MathVista is qualitatively different from ChartQA:

- ChartQA: concentrated effect (S1 = 0.056) decaying to zero at S4 (0.000). Bounded plausibility window. S1-only sampling isolates the effect.
- MathVista: shallow gradient (S1 = 0.110 → S5 = 0.051), with non-trivial adoption all the way to >300 % · GT. No plausibility window — the effect tails into the far field.

Possible drivers (not separated here):

1. **Base prediction is often wildly wrong.** With target_only `accuracy_exact = 0.229`, llava-interleave-7b is under-confident and frequently produces predictions far from GT. When `base_pred` itself is order-of-magnitude wrong, the "anchor in the model's plausibility window" framing breaks down — the model's window is huge.
2. **Far-field anchors land at structurally-significant absolute values.** S5 examples include `gt=3, anchor=1400 → pred=1400` and `gt=4, anchor=10000 → pred=10000`. The model is picking up "the only specific number in context" rather than evaluating distance from a (poorly-formed) prior.
3. **Cross-anchor contamination via image-modality priors.** Possible but not tested: MathVista images include charts, diagrams, and natural-image scenes; the anchor patch may be processed differently than in ChartQA's chart-only context.

**Concrete operational recommendation:**
- For the paper's headline cross-dataset claim, **scope MathVista out for now**. Use VQAv2 (E5b) + TallyQA (E5b) + ChartQA (E5d-γ) as the three datasets where the cutoff rule and its plausibility-window interpretation hold.
- Queue a follow-up MathVista run at **n = 500 with stricter filter** (GT ≥ 10, integer-only, single-image only, llava-interleave-7b) before any paper claim about MathVista. Or extend the panel to a stronger model (Qwen2.5-VL-7B) where target_only accuracy is higher and the diffuse adoption may be tighter.
- Alternatively, publish **MathVista as a contrast case**: "the plausibility-window principle holds on integer-VQA, counting, and chart datasets; MathVista's mixed visual reasoning shows diffuse anchor susceptibility regardless of distance, which we attribute to the higher base-prediction error rate."

## What this doesn't cover

- **Float / decimal GT** — filtered out by `require_single_numeric_gt`. MathVista has 540 multi_choice + 460 free_form items; only 153 integer-GT items satisfy our [1, 1000] gate. Full coverage would need decimal-aware stratification (already partial via `compute_strata` rounding).
- **GT > 1000 (~4 % of MathVista)** — excluded by `answer_range`. Anchor inventory caps at 10000.
- **n = 200 statistical power** — actually n_base = 153 (loader-bound). Wrong-base cells have n_eligible 100–118, so S1 95 % CI = [0.050, 0.170] is wide. C3 failure is robust enough to survive a doubling of n, but not C1's slope.
- **Single-model panel** — single model (llava-interleave-7b). The diffuse pattern may flatten further or sharpen on stronger / weaker VLMs.

## What we did NOT test

- **Anchor below GT vs above GT asymmetry** — same as E5b/E5c.
- **`max_new_tokens = 16` vs 8 baseline interaction** — bumped to 16 for MathVista; not ablated.
- **Inverse-relative cutoff (e.g., `|a/GT − 1| ≤ 0.10`)** vs. additive-relative cutoff — we used `|a − GT| / GT`, equivalent only for `a > 0` and `GT > 0`.
- **Image-vs-text anchor delivery** — the anchor here is always a separate FLUX-rendered image. Effect of textual anchor delivery on MathVista is unknown.
- **Per-task-type breakdown** — MathVista has multiple task types (numerical, geometric, scientific reasoning); the diffuse pattern may concentrate in one sub-task rather than uniformly.

## Implications for paper

1. **For MathVista paper-canonical experiments**: do not assert a cross-modal anchoring effect bounded by distance on MathVista, at least not at this n / model. Either drop MathVista from the cross-dataset table or document it explicitly as the contrast case.
2. **For ChartQA paper-canonical experiments**: anchor sampling rule remains "S1 only relative" (`|a − GT| ≤ max(1, 0.10 · |round(GT)|)`).
3. **For E5b/E5c (VQAv2 + TallyQA)**: existing rule `anchor ∈ {0..9}` stays unchanged.
4. **Cross-dataset narrative**: the plausibility-window principle holds on counting and integer-VQA tasks (VQAv2, TallyQA, ChartQA-integer subset); MathVista's mixed visual reasoning shows diffuse anchor susceptibility, suggesting the principle is conditioned on the model having a tight prior over the answer.

## Implications for the experiment plan

- **§6 Tier 2 row E5d (MathVista)** — flip from `☐` to `⚠ landed but C3 FAIL (commits 6131e8a..<this commit>; MathVista n=153 hybrid relative-cutoff; diffuse pattern; not adopted as paper-canonical)`.
- **MathVista follow-up** — queue with n = 500, stricter GT-floor filter (GT ≥ 10), and ideally a second model (Qwen2.5-VL-7B) before any MathVista claim in the paper.
- **Inventory extension** — not the bottleneck here. The current S5 anchor pool [301, 10000] is large enough; the issue is model behaviour, not coverage.
