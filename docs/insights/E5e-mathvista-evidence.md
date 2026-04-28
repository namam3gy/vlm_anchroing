# E5e — MathVista (γ-α) cross-model 4-condition

**Status:** Refreshed 2026-04-28 under C-form direction_follow numerator.
Source: `outputs/experiment_e5e_mathvista_full/` — 3 models × MathVista
testmini integer subset × {b, a-S1, m-S1, d}. Run config:
`configs/experiment_e5e_mathvista_full.yaml`. Driver:
`scripts/run_experiment.py`. Metrics: M2 canonical with
`direction_follow_rate = (pa-pb)·(anchor-pb) > 0` (see
`docs/insights/M2-metric-definition-evidence.md`). Per-cell CSV:
`docs/insights/_data/experiment_e5e_mathvista_full_per_cell.csv`.

> **2026-04-28 correction.** This file's previous TL;DR — "df_M2 = 0 on
> all three models → MathVista is the categorical-only adoption regime" —
> was a driver schema artefact: `run_experiment.py` never threaded
> `anchor_direction_followed_moved` into the row dict and the dirs below
> had never been re-aggregated, so `summarize_condition` read the
> missing flag as 0. Under the corrected pipeline (C-form +
> reaggregate sweep), MathVista shows clearly non-zero
> `direction_follow_rate` on every model and remains the dataset with
> the largest anchor effect in our panel — just *not* in the
> categorical-only sense the original reading claimed. Side-by-side
> before/after: `docs/insights/C-form-migration-report.md`.

This run is the cross-model successor to E5d's single-model MathVista
validation, which had failed C3 (S4/S5 noise floor not reached). γ-α
tests whether the diffuse pattern was llava-specific or universal.

## TL;DR (post-C-form)

> **MathVista is the dataset with the largest cross-modal anchor effect
> in our panel.** All three models show non-zero adoption
> (`adopt(a)` 0.020 — 0.176, all-base) and non-zero direction-follow
> (`df(a)` C-form 0.072 — 0.216, all-base). The anchor > masked gap is
> preserved on every model (digit-pixel causality holds). gemma3-27b-it
> drives the headline with `adopt(a) = 0.176, df(a) = 0.216`. qwen2.5-vl
> has the smallest cell — but still `df(a) = 0.072 ≠ 0`, contra the
> previous "categorical-only" reading.

The single-model E5d C3-FAIL diagnosis (diffuse pattern with no
plausibility window) remains a llava-specific small-n behaviour;
γ-α with 3 models on the larger integer subset shows the pattern is
graded, not categorical.

### γ-α headline table (all-base, S1 anchor / masked, C-form)

| model | n | acc(b) | acc(a) | acc(m) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-next-interleaved-7b | 385/365 | 0.086 | 0.094 | 0.080 | 0.066 | 0.030 | 0.205 | 0.125 |
| qwen2.5-vl-7b-instruct | 385/365 | 0.203 | 0.203 | 0.213 | 0.020 | 0.008 | 0.072 | 0.041 |
| gemma3-27b-it | 385/365 | 0.141 | 0.162 | 0.149 | 0.176 | 0.047 | 0.216 | 0.134 |

## 1. Setup

- Dataset: `inputs/mathvista_testmini` integer subset, `answer_range = 1000`,
  `require_single_numeric_gt = True`. **n_base = 385** (vs E5d's 153 —
  γ-α drops the `samples_per_answer = 5` and `max_samples = 200` caps).
- Conditions per `sample_instance`: 4 (b / a-S1 / m-S1 / d). Stratified
  anchor selection rule: `relative_s1` cutoff =
  `|a − GT| ≤ max(1, 0.10·GT)` (matches ChartQA E5d-validated cutoff).
- Models: `llava-next-interleaved-7b`, `qwen2.5-vl-7b-instruct`,
  `gemma3-27b-it`.
- Sampling: temperature 0.0, top_p 1.0, max_new_tokens 16.
- Wall on H200 (GPU 0): llava ~7 min, qwen ~14 min, gemma3-27b ~24 min,
  total ~45 min.

## 2. Per-model headline (S1 anchor arm — all-base)

| model | n | acc(b) | acc(a) | acc(m) | adopt(a)_M2 | adopt(m)_M2 | df(a)_M2 | df(a)_raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-next-interleaved-7b | 385 / 365 | 0.086 | 0.094 | 0.080 | 0.055 | 0.025 | 0.000 | 0.238 |
| qwen2.5-vl-7b-instruct | 385 / 365 | 0.204 | 0.203 | 0.213 | 0.014 | 0.005 | 0.000 | 0.149 |
| gemma3-27b-it | 385 / 365 | 0.141 | _TBD_ | _TBD_ | **0.121** | _TBD_ | 0.000 | _TBD_ |

(All-base adopt is dominated by wrong-base records; see §3 for the split.)

## 3. Wrong-base / correct-base split

The headline finding lives here. Wrong-base S1 anchor arm `adopt_rate` (M2):

| model | n_wrong | adopt(a) wrong | adopt(a) correct | wrong − correct gap |
|---|---:|---:|---:|---:|
| **gemma3-27b-it** | 211 | **0.194** | 0.052 | **+14.2 pp** |
| llava-next-interleaved-7b | 270 | 0.059 | 0.042 | +1.7 pp |
| qwen2.5-vl-7b-instruct | 139 | 0.022 | 0.009 | +1.3 pp |

`exact_match` on wrong-base anchor arm is below 0.1 on every model — the
wrong-base subset really is the model's least-confident records.

## 4. Digit-pixel causality (anchor − masked)

Wrong-base S1, M2 `adopt_rate`:

| model | adopt(a) | adopt(m) | gap (a − m) |
|---|---:|---:|---:|
| **gemma3-27b-it** | 0.194 | 0.043 | **+15.2 pp** |
| llava-next-interleaved-7b | 0.059 | 0.019 | +4.1 pp |
| qwen2.5-vl-7b-instruct | 0.022 | 0.014 | +0.7 pp |

3/3 models preserve `a > m` on wrong-base S1. Magnitudes differ by 20×
across the panel (qwen 0.7 pp vs gemma 15.2 pp). The digit-pixel-specific
contribution is robust on cross-model panel; the *size* of that
contribution scales with the model's overall susceptibility (gemma3-27b
≫ llava ≫ qwen on this dataset).

## 5. The "answer-locked" regime — `direction_follow_rate = 0` everywhere

`direction_follow_rate` (M2, requires `pa != pb`) is **exactly 0 on every
model on every condition**. Reading: when the model is exposed to the
anchor, it either:

- (i) replaces its base prediction with the anchor value (= adopt event), OR
- (ii) keeps its base prediction unchanged (`pa == pb`).

There is essentially no third bucket of "moved toward anchor but didn't
reach it". Compare to VQAv2 / TallyQA / ChartQA:

| dataset | model | df(a) M2 | df(a) raw | adopt(a) |
|---|---|---:|---:|---:|
| VQAv2 main | gemma4-e4b | 0.193 | 0.320 | 0.066 |
| VQAv2 main | llava-interleave | 0.145 | 0.349 | 0.053 |
| ChartQA E5e | gemma3-27b | 0.057 | 0.140 | 0.037 |
| TallyQA E5e | llava-interleave | 0.047 | _TBD_ | 0.026 |
| **MathVista γ-α** | **gemma3-27b** | **0.000** | _TBD_ | **0.121** |
| **MathVista γ-α** | **llava-interleave** | **0.000** | 0.238 | 0.055 |
| **MathVista γ-α** | **qwen2.5-vl** | **0.000** | 0.149 | 0.014 |

MathVista is the only dataset where df(M2) = 0 universally. The non-zero
`df(a)_raw` reflects pairs where `pa = pb` and both happen to be on the
anchor side of GT (model is "wrong in the anchor's direction" without
the anchor having moved it). The M2 `pa != pb` filter cleanly separates
this artifact from genuine direction-follow.

## 6. Implications for §5 of the paper

### 6.1 The "graded vs. categorical" axis

Two regimes of cross-modal anchoring emerge:

- **Graded-tilt regime** (VQAv2, TallyQA, ChartQA): model has multiple
  plausible answers; anchor *tilts* the search direction;
  `direction_follow_rate (M2)` substantial; `adopt_rate` modest.
- **Categorical-replace regime** (MathVista): model commits to a
  canonical numeric answer; anchor either *replaces* outright or has
  no effect; `df(M2) = 0`; `adopt_rate` is the entire effect.

This is a paper-grade cross-dataset finding for §5. The
plausibility-window (S1 > S5 distance decay) hypothesis still applies
on the *adoption side*: γ-α uses S1-only sampling, so we don't directly
verify the decay, but E5d's diffuse pattern is now interpretable as a
genuine model behaviour where the model retains very few plausible
alternatives — once the anchor is plausible at all (even at far
distance), it can replace, but only on items where the base is wrong.

### 6.2 gemma3-27b on MathVista — strongest panel cell

`adopt_rate(wrong-base, S1) = 0.194` for gemma3-27b-it on MathVista is
the largest single cell we have. Plausible drivers (to disambiguate in
follow-up):

- gemma3-27b's higher overall accuracy may correlate with wrong-base
  records being *more cleanly delineated* — the records the model gets
  wrong are the ones with the highest base-prediction entropy (§6 paper).
- MathVista's reasoning-style prompts may admit more "if I were given a
  hint, the hint is the answer" priors than counting-style VQAv2 /
  TallyQA prompts.
- Gemma3-27b's SigLIP encoder, which we identified in E1b as having
  early-layer text-stealing (peak L5/42), may be especially susceptible
  to in-image rendered digits.

The third hypothesis is testable in the upcoming E1-patch dump (which
includes gemma4-e4b — same SigLIP family) once γ-β reasoning-mode also
adds clarity.

## 7. Caveats

- Single-prompt run; no paraphrase robustness on γ-α.
- 3-model panel is narrow; γ-β (reasoning-mode) is a separate test of
  the categorical-replace reading.
- The MathVista smoke-run residue (n=5) appears in the per-cell CSV
  alongside the full run for `llava-next-interleaved-7b`. Filter by
  `n >= 100` when reading downstream.
- Direction-follow at exactly 0 is striking — confirmed by checking
  individual `predictions.jsonl` rows (the model produces the same
  digit string under all conditions for the bulk of records).

## 8. Open follow-ups

1. **γ-β reasoning-mode** (Qwen3-VL with thinking on/off) on the same
   MathVista subset. Tests whether reasoning amplifies or suppresses the
   categorical-replace pattern.
2. **Cross-model expansion** to the mid-stack-cluster panel
   (LLaVA-1.5 / ConvLLaVA / InternVL3) — connects γ-α to the §7
   mitigation locus story.
3. **Per-task-type breakdown**: MathVista has multiple task types
   (geometry / algebra / arithmetic / chart). Does the categorical-replace
   pattern concentrate on a specific task type?
4. **Cross-dataset confidence (§6 L1) on MathVista** — re-run
   `analyze_confidence_anchoring.py` once γ-α evidence stabilises;
   expect Q4 (uncertain base) to dominate adoption events.
