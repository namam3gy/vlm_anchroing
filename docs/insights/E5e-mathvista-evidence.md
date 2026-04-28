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

## 2. Per-model headline (S1 anchor arm — all-base, C-form)

| model | n | acc(b) | acc(a) | acc(m) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-next-interleaved-7b | 385 / 365 | 0.086 | 0.094 | 0.080 | 0.066 | 0.030 | 0.205 | 0.125 |
| qwen2.5-vl-7b-instruct | 385 / 365 | 0.203 | 0.203 | 0.213 | 0.020 | 0.008 | 0.072 | 0.041 |
| **gemma3-27b-it** | 385 / 365 | 0.141 | 0.162 | 0.149 | **0.176** | 0.047 | **0.216** | 0.134 |

(All-base adopt is dominated by wrong-base records; see §3 for the split.)

## 3. Wrong-base / correct-base split (C-form)

The headline finding lives here. Wrong-base S1 anchor arm rates from
`docs/insights/_data/experiment_e5e_mathvista_full_per_cell.csv`:

| model | n_wrong | adopt(a) wrong | adopt(a) correct | wrong − correct gap | df(a) wrong C-form |
|---|---:|---:|---:|---:|---:|
| **gemma3-27b-it** | 211 | **0.230** | 0.079 | **+15.1 pp** | **0.332** |
| llava-next-interleaved-7b | 270 | 0.065 | 0.070 | −0.5 pp | 0.266 |
| qwen2.5-vl-7b-instruct | 139 | 0.026 | 0.014 | +1.2 pp | 0.162 |

`exact_match` on wrong-base anchor arm: 0.05–0.18 — the wrong-base
subset is the model's least-confident records.

## 4. Digit-pixel causality (anchor − masked, wrong-base S1, C-form)

| model | adopt(a) wrong | adopt(m) wrong | gap (a − m) |
|---|---:|---:|---:|
| **gemma3-27b-it** | 0.230 | 0.051 | **+17.9 pp** |
| llava-next-interleaved-7b | 0.065 | 0.020 | +4.5 pp |
| qwen2.5-vl-7b-instruct | 0.026 | 0.014 | +1.2 pp |

3/3 models preserve `a > m` on wrong-base S1. The digit-pixel-specific
contribution scales with the model's overall susceptibility
(gemma3-27b ≫ llava ≫ qwen on this dataset).

## 5. Graded movement is real on MathVista — earlier "categorical-only" was a driver-bug artefact

The 2026-04-29 first pass reported `direction_follow_rate (M2) = 0` on every
model on every condition and concluded MathVista was the
"categorical-replace" regime — anchor either replaces outright or doesn't
move pa at all. **That conclusion was wrong**: it followed from the
driver schema gap (`run_experiment.py` not threading
`anchor_direction_followed_moved` into the row dict; `summarize_condition`
defaulting the missing flag to 0). After the C-form refactor + reaggregate
sweep, MathVista shows clearly non-zero direction-follow with a clean
graded structure:

| dataset | model | df(a) C-form | adopt(a) |
|---|---|---:|---:|
| VQAv2 main | gemma4-e4b | 0.274 | 0.066 |
| VQAv2 main | llava-interleave | 0.172 | 0.053 |
| ChartQA E5e | gemma3-27b | 0.073 | 0.037 |
| TallyQA E5e | llava-interleave | 0.078 | 0.020 |
| **MathVista γ-α** | **gemma3-27b** | **0.216** | **0.176** |
| **MathVista γ-α** | **llava-interleave** | 0.205 | 0.066 |
| **MathVista γ-α** | **qwen2.5-vl** | 0.072 | 0.020 |

**MathVista is the strongest single dataset for adopt(a) AND a top-tier
dataset for df(a)**, simultaneously. The graded-movement layer co-exists
with the adoption layer; both are present.

## 6. γ-β: reasoning amplifies anchor pull (Qwen3-VL-8B Instruct vs. Thinking)

γ-β tests whether reasoning-mode (chain-of-thought) suppresses or
amplifies the cross-modal anchor effect. The pair is `Qwen3-VL-8B-Instruct`
(thinking-off) vs. `Qwen3-VL-8B-Thinking` (separately-trained
reasoning-mode checkpoint, Apache-2.0). Same architecture, same chat
template, same 4-condition stimuli. Source:
`outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-{instruct,thinking}/20260428-114421/`,
post-C-form reaggregate.

| model | acc(b) | acc(a) | acc(m) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-vl-8b-instruct | 0.216 | 0.218 | 0.220 | 0.074 | 0.030 | 0.102 | 0.069 |
| **qwen3-vl-8b-thinking** | 0.196 | 0.202 | 0.196 | **0.117** | 0.032 | **0.291** | 0.237 |

**Thinking amplifies anchor pull on every metric**: adopt ×1.6,
df ×2.9 (thinking ÷ instruct). Anchor > masked digit-pixel causality
preserved on both models. acc(b) is *lower* on thinking (0.196 vs 0.216)
— reasoning trace doesn't gain accuracy on this panel; it just loses
anchor robustness.

This is the strongest single reasoning-amplifies-bias finding in our
panel and an empirical confirmation of the VLMBias / Wang
LRM-judging direction (text-only reasoning models can be *more* biased
than their non-reasoning counterparts; H4 lands on the *amplification*
side, not the *suppression* side).

## 7. Implications for §5 / §6 / §8 of the paper

### 7.1 §5 — graded-movement claim survives on MathVista

The earlier "categorical-replace" reading is retracted. MathVista
contributes both a strong adoption signal AND a strong direction-follow
signal — the dataset where anchoring is largest, not where
graded-movement is absent. The §5 cross-dataset cross-model panel is
strengthened, not split.

### 7.2 gemma3-27b on MathVista — strongest panel cell

Wrong-base S1 `adopt(a) = 0.230` and `df(a) = 0.332` for gemma3-27b-it
on MathVista are the largest single cells we have. Plausible drivers
(testable in follow-up):

- gemma3-27b's higher overall accuracy may correlate with wrong-base
  records being *more cleanly delineated* — the records the model gets
  wrong are the ones with the highest base-prediction entropy (§6 paper).
- MathVista's reasoning-style prompts may admit more "if I were given a
  hint, the hint is the answer" priors than counting-style VQAv2 /
  TallyQA prompts.
- Gemma3-27b's SigLIP encoder, which E1b identified as having early-layer
  text-stealing (peak L5/42), may be especially susceptible to in-image
  rendered digits — testable via E1-patch (gemma4-e4b is the same
  SigLIP family).

### 7.3 §6 — graded-tilt vs adoption decomposed

L1's confidence-quartile result (entropy_top_k Q4-Q1 mean +0.152 on
df(a) under C-form, 23/35 cells fully monotone) explains why MathVista
shows substantial direction-follow alongside adoption: MathVista's
wrong-base subset is the highest-entropy cohort the panel has, and §6
predicts highest-entropy cohorts get the largest graded pull. γ-β
strengthens this: thinking lowers acc(b) (= raises mean entropy) and
df(a) rises in lockstep (×2.9).

### 7.4 §8 — reasoning-mode VLMs as future work

γ-β closes the H4 question (does reasoning suppress or amplify
anchoring?) on the *amplification* side. §8 should keep the broader
"reasoning-mode VLM at scale" direction open as future work — γ-β is
a single-pair single-dataset answer, the broader claim needs more
models and datasets.

## 8. Caveats

- Single-prompt run; no paraphrase robustness on γ-α / γ-β.
- 3-model panel is narrow on γ-α; γ-β is a 2-model pair.
- The MathVista smoke-run residues (n=5 / n=3) appear in the per-cell
  CSV alongside the full runs. Filter by `n >= 100` when reading
  downstream.
- Pre-refactor (driver-bug 0) γ-α + γ-β results preserved at
  `outputs/before_C_form/experiment_e5e_mathvista_full/` and
  `outputs/before_C_form/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-thinking_postlanding/`
  for audit. **DO NOT reference those for analysis** — see backup
  README for the full notice.

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
