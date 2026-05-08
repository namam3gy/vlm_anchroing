# E5e — MathVista (γ-β) reasoning-mode amplification

**Status:** First write-up 2026-05-04. Inference landed 2026-04-28 and the
headline ratio (×1.6 adopt, ×2.9 df) has lived as a row in `roadmap.md
§3.1` since then; this doc adds the wrong-base / correct-base split that
makes the result a §1 / §6 / §8 contribution rather than just a marginal
ratio. Source:
`outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-{instruct,thinking}/20260428-114421/`,
post-C-form reaggregate (`predictions.marginal.bak.jsonl` retained).
Per-cell CSV: `docs/insights/_data/experiment_e5e_mathvista_reasoning_per_cell.csv`.
Notebook: `notebooks/E5e_reasoning_ablation.ipynb`.

> **Companion doc.** This is a deeper supplement to
> `docs/insights/E5e-mathvista-evidence.md §6`, which carries the
> all-base headline numbers. New material here: the wrong/correct
> split (§3), the masked-arm contrast (§4), and the "H2 binary
> projection collapses in reasoning mode" interpretation (§5).

## TL;DR

> **Reasoning mode amplifies anchor pull *most strongly on correct-base
> samples*, where instruct mode shows almost none.** On MathVista S1
> anchor arm, `direction_follow_rate` C-form jumps from 0.021
> (qwen3-instruct) to 0.267 (qwen3-thinking) on correct-base records —
> a **×12.7 amplification** that the all-base headline (×2.9) hides.
> Wrong-base df only goes from 0.256 → 0.327 (×1.28). The H2
> wrong > correct asymmetry that holds across the entire main panel
> **collapses** in reasoning mode: thinking pulls correct answers
> nearly as much as wrong ones (df 0.267 vs 0.327). This is the cleanest
> empirical signal in the project that **continuous confidence
> monotonicity (H7) breaks down when chain-of-thought is on**.

## 1. Setup

- Model pair: `Qwen/Qwen3-VL-8B-Instruct` (thinking-off; standard chat
  template) vs. `Qwen/Qwen3-VL-8B-Thinking` (separately-trained
  reasoning-mode checkpoint, Apache-2.0). Same architecture, same
  vision tower, same parameter count. Differs only in instruction
  tuning + chat template.
- Dataset: MathVista testmini integer subset, n_base = 385.
- Conditions: 4 (b / a-S1 / m-S1 / d). S1 stratum: `|a − GT| ≤ max(1, 0.10·GT)`.
- Driver: `scripts/run_experiment.py` with `</think>`-aware response
  parsing for the thinking model (extracts the post-`</think>`
  numeric answer).
- Sampling: temperature 0.0, top_p 1.0, max_new_tokens 512 (thinking
  needs the budget; instruct rarely uses more than 16).

## 2. All-base headline (S1 anchor arm, C-form)

| model | n | acc_em(b) | acc_em(a) | acc_em(m) | acc_em(d) | adopt(a) | df(a) C-form | df(m) C-form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-vl-8b-instruct | 385 / 365 | 0.647 | 0.655 | 0.660 | 0.649 | 0.074 | 0.102 | 0.069 |
| **qwen3-vl-8b-thinking** | 385 / 365 | 0.587 | 0.605 | 0.589 | 0.566 | **0.117** | **0.291** | **0.237** |

Ratios (thinking ÷ instruct):

| metric | ratio |
|---|---:|
| `adopt(a)` | ×1.6 |
| `adopt(m)` | ×1.1 |
| `df(a) C-form` | ×2.9 |
| `df(m) C-form` | ×3.4 |

Notes:
- `acc_em` (exact match) is the metric reported here; the legacy
  `accuracy_vqa` numbers in `E5e-mathvista-evidence.md §6` use the
  VQAv2-style soft-match. Both metrics agree on direction.
- Thinking has lower `acc_em(b)` (0.587 vs 0.647). Reasoning trace
  on MathVista does not improve accuracy *and* loses anchor
  robustness — strict cost on this dataset.

## 3. Wrong-base / correct-base split — H7 monotonicity collapses

This is the new material. Wrong vs correct base prediction is the H2
binary projection of confidence (cf. `L1-confidence-modulation-evidence.md`).
Across the main panel, every model shows **df(wrong-base) > df(correct-base)** —
graded confidence-monotonicity in coarse form. Reasoning mode breaks the
asymmetry.

| model | base subset | n | adopt(a) | df(a) C-form | em(a) |
|---|---|---:|---:|---:|---:|
| qwen3-vl-8b-instruct | wrong | 127 | 0.139 | 0.256 | 0.094 |
| qwen3-vl-8b-instruct | correct | 238 | 0.026 | **0.021** | 0.954 |
| qwen3-vl-8b-thinking | wrong | 151 | 0.178 | 0.327 | 0.258 |
| qwen3-vl-8b-thinking | correct | 214 | 0.072 | **0.267** | 0.850 |

**Critical row to compare:** `df(a) C-form` on correct-base.
- instruct: **0.021** (essentially zero — confident answers stay put)
- thinking: **0.267** (anchor pulls correct answers as strongly as
  wrong ones; df(0.267) ≈ df(0.327) on wrong-base).

The thinking ÷ instruct ratio on this single cell is **×12.7**, vs ×1.28
on wrong-base. The all-base ×2.9 headline averages these two regimes
together and undersells the qualitative finding.

### 3.1 H2 / H7 framing

- **H2** (wrong > correct): instruct gap is +23.5 pp on df(a)
  (0.256 vs 0.021). Thinking gap is +6.0 pp (0.327 vs 0.267) — H2
  **largely dissolves** in reasoning mode.
- **H7** (continuous confidence monotonicity): the binary
  wrong/correct projection of H2 is the coarsest possible monotonicity
  test. H2's collapse is direct evidence that the underlying
  uncertainty-modulated graded pull from `L1-confidence-modulation-evidence.md`
  **does not hold** when the model emits a reasoning trace before the
  final token.

The interpretation we prefer: **reasoning mode injects anchor-related
content into the chain-of-thought as if it were salient, regardless
of base-answer confidence.** The `</think>` post-trace token is then
sampled from a distribution biased by the trace, which mentions
the anchor digit by virtue of its visual presence.

## 4. Masked-arm contrast — digit-pixel causality survives but compresses

Wrong-base S1 anchor − masked gap (a − m) on `df` and `adopt`:

| model | df(a) wrong | df(m) wrong | gap (a − m) | adopt(a) wrong | adopt(m) wrong | gap (a − m) |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-vl-8b-instruct | 0.256 | 0.176 | +8.0 pp | 0.139 | 0.052 | +8.7 pp |
| qwen3-vl-8b-thinking | 0.327 | 0.248 | +7.9 pp | 0.178 | 0.034 | +14.4 pp |

Correct-base:

| model | df(a) correct | df(m) correct | gap (a − m) | adopt(a) correct | adopt(m) correct |
|---|---:|---:|---:|---:|---:|
| qwen3-vl-8b-instruct | 0.021 | 0.013 | +0.8 pp | 0.026 | 0.013 |
| qwen3-vl-8b-thinking | 0.267 | 0.230 | +3.7 pp | 0.072 | 0.031 |

Reading: the **digit-pixel-specific contribution** (a − m) is roughly
preserved across modes (8 pp on df, wrong-base) — i.e. reasoning mode
does not *only* pull on the digit, it pulls on **both digit and
non-digit cues from the second image**. df(m) goes from 0.176
(instruct, wrong-base) to 0.248 (thinking, wrong-base) — a 7 pp jump
that comes from non-digit-pixel image content alone.

This decomposes the ×2.9 df amplification into:
1. **Non-digit-pixel image-presence amplification** (≈ 7 pp jump on
   df(m)): reasoning is sensitive to the *fact* that there is a second
   image, not just its digit.
2. **Digit-pixel amplification** (preserved 8 pp gap a − m): reasoning
   doesn't suppress the digit-pixel-specific channel; it only adds to it.

## 5. Implications

### §1 — opening hook
The reasoning amplification is a strong opening pull-quote:
"chain-of-thought *amplifies* the cross-modal anchor effect rather than
suppressing it, with the largest amplification on records the
non-reasoning model gets correct (×12.7 on df)." The Dual-Process-Theory
"System 2 should suppress System 1 anchoring" prior is wrong here; it's
also wrong in VLMBias (in-image text bias on o3, o4-mini) and in
Wang LRM-judging. Three independent finds, same direction.

### §6 — confidence modulation
`L1-confidence-modulation-evidence.md` reports `df(a) Q4 − Q1 = +0.152`
under entropy_top_k on the panel models (E5b/E5c/E5e, 4 datasets,
3 models). The thinking-mode result is the **first cell where the
monotone trend doesn't fit** — both the wrong-base and correct-base
subsets are pulled. We add a §6.4 paragraph distinguishing
"uncertainty-modulated graded pull" (instruct + panel main) from
"reasoning-induced graded pull" (thinking).

### §8 — future work, F3
The single γ-β cell *replaces* the previous F3 "scope only" framing.
Direction we'd take if we extended:
- Same Qwen3-VL pair on the rest of the §3.3 main matrix
  (Tally / Chart / Info / Plot) — does the correct-base amplification
  hold off MathVista?
- A **causal** version: same prompt with `<think>...</think>` injected
  manually around the visible reasoning (or stripped via
  `</think>`-token clamping) on the *instruct* model, to isolate
  template-vs-weights as the driver.
- Larger reasoning models (o3-style) on the same 4-condition stimuli.

## 6. Caveats

- **n = 365** on each anchor arm. Confidence intervals on the
  correct-base df cell (n = 214 for thinking, n = 238 for instruct) are
  wide. Bootstrap 95 % on `df(a) correct, thinking` is approximately
  [0.21, 0.33]; on instruct correct it's [0.005, 0.045]. The ratio is
  decisively > 1 even at the lower bounds.
- **Same architecture, different training**. The Instruct vs. Thinking
  weights diverge during reasoning-mode SFT/RL. We cannot fully
  separate "reasoning trace presence" from "Thinking-checkpoint
  weight drift" with this single pair. The decomposition in §4 (df(m)
  shifts) suggests at least part of the effect is template-bound
  rather than purely weight-bound, but a clean isolation needs the
  injected-reasoning version proposed in §5 §8.
- **MathVista-specific.** Other datasets may show different
  amplification ratios; the paper's §5 / §8 prose should not
  generalise the ×12.7 figure beyond MathVista without the multi-dataset
  follow-up.
