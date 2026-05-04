# E7 — PlotQA + InfoVQA full panel (Phase 1 P0 v3 chart-stack)

**Status:** First standalone write-up 2026-05-04. Inference landed 2026-05-02.
The 7-model × 2-dataset matrix (Phase 1 P0 v3 chart-stack expansion) was
folded into the §3.3 main panel headline via `phase1-p0-v3-summary.md` and
`docs/insights/_data/main_panel_5dataset_summary.md`; this doc surfaces
the per-dataset findings that the umbrella summary doesn't show.

Source:
- `outputs/experiment_e7_plotqa_full/<model>/<run>/`, n_base = 5000.
- `outputs/experiment_e7_infographicvqa_full/<model>/<run>/`, n_base = 1147.
- Per-cell CSVs (gitignored):
  `docs/insights/_data/experiment_e7_plotqa_full_per_cell.csv`,
  `docs/insights/_data/experiment_e7_infographicvqa_full_per_cell.csv`.
- Models (7-panel): `gemma3-4b-it`, `gemma3-27b-it`, `internvl3-8b`,
  `llava-next-interleaved-7b`, `llava-onevision-qwen2-7b-ov` (Main),
  `qwen2.5-vl-7b-instruct`, `qwen2.5-vl-32b-instruct`. (gemma3-12b-it
  was queued but never produced predictions; empty dir under
  `experiment_e7_plotqa_full/gemma3-12b-it/`.)

## TL;DR

> **Anti-scaling is dataset-bound, not architecture-bound.** Within the
> Gemma3 family, 4B is *more* anchor-pulled than 27B on PlotQA
> (df 0.395 vs 0.227, wrong-base) but the ordering reverses on InfoVQA
> (4B 0.324 vs 27B 0.350). Within Qwen2.5-VL, the 7B-vs-32B gap is
> small (≤ 4 pp) on both datasets. **internvl3-8b is the most robust
> model on PlotQA but only mid-pack on InfoVQA.** The H2 wrong > correct
> asymmetry holds on every model except internvl3, which has a
> near-zero wrong−correct gap (PlotQA 0.008 pp; InfoVQA 0.024 pp) —
> a panel-side hint of the same H2-collapse pattern that thinking
> mode shows on MathVista (`E5e-mathvista-reasoning-evidence.md`).
>
> **Anchor presence sometimes *improves* exact-match on PlotQA.** 6/7
> models post `em(a) > em(b)` on the all-base PlotQA cell — a baseline
> "free-lunch" pattern that motivates E6 §7.4.5 mitigation. On InfoVQA
> the picture is mixed.

## 1. Setup

| field | PlotQA | InfoVQA |
|---|---|---|
| n_base (target_only) | 5,000 | 1,147 |
| Conditions per sid | 4 (b / a-S1 / m-S1 / d) | 4 |
| Stratum cutoff S1 | `|a − GT| ≤ max(1, 0.10·GT)` | `|a − GT| ≤ max(1, 0.10·GT)` |
| Driver config | `configs/experiment_e7_plotqa_full.yaml` | `configs/experiment_e7_infographicvqa_full.yaml` |

Same M2 + C-form metrics as §3.3 main panel; rows split by
`base_correct` via `scripts/analyze_e5e_wrong_correct.py`.

> **PlotQA heavy-tail caveat (§9 of roadmap).** PlotQA emits raw chart
> values (e.g. `"10000000"` for `gt = 22`) on a small minority of
> samples, inflating `mean_distance_to_anchor`. We report
> `median_distance_to_anchor` in the live metrics and use df / adopt
> (categorical metrics) for cross-model comparison; the heavy-tail
> only affects the legacy mean-distance field.

## 2. Cross-model susceptibility ranking (df wrong-base, S1 anchor)

### 2.1 PlotQA

| model | df(a) wrong | df(a) correct | wrong − correct | adopt(a) wrong | em(a) wrong |
|---|---:|---:|---:|---:|---:|
| **gemma3-4b-it** | **0.395** | 0.057 | +0.338 | 0.184 | 0.123 |
| llava-next-interleaved-7b | 0.295 | 0.122 | +0.173 | 0.082 | 0.031 |
| gemma3-27b-it | 0.227 | 0.024 | +0.203 | 0.099 | 0.063 |
| llava-onevision-qwen2-7b-ov *(Main)* | 0.206 | 0.021 | +0.185 | 0.090 | 0.044 |
| qwen2.5-vl-7b-instruct | 0.174 | 0.015 | +0.159 | 0.024 | 0.119 |
| qwen2.5-vl-32b-instruct | 0.163 | 0.009 | +0.154 | 0.023 | 0.091 |
| **internvl3-8b** | **0.095** | 0.087 | +0.008 | 0.002 | 0.021 |

### 2.2 InfoVQA

| model | df(a) wrong | df(a) correct | wrong − correct | adopt(a) wrong | em(a) wrong |
|---|---:|---:|---:|---:|---:|
| **gemma3-27b-it** | **0.350** | 0.035 | +0.315 | 0.163 | 0.114 |
| gemma3-4b-it | 0.324 | 0.075 | +0.249 | 0.133 | 0.094 |
| llava-next-interleaved-7b | 0.244 | 0.115 | +0.129 | 0.061 | 0.047 |
| llava-onevision-qwen2-7b-ov *(Main)* | 0.190 | 0.059 | +0.131 | 0.020 | 0.079 |
| internvl3-8b | 0.161 | 0.137 | +0.024 | 0.005 | 0.084 |
| qwen2.5-vl-32b-instruct | 0.156 | 0.009 | +0.147 | 0.098 | 0.069 |
| qwen2.5-vl-7b-instruct | 0.123 | 0.008 | +0.115 | 0.025 | 0.068 |

## 3. Anti-scaling within Gemma3 — dataset-dependent

| dataset | gemma3-4b | gemma3-27b | direction |
|---|---:|---:|---|
| PlotQA | 0.395 | 0.227 | **4B more pulled** (anti-scaling) |
| InfoVQA | 0.324 | 0.350 | 27B slightly more pulled (positive scaling) |

The §3.3 panel summary highlights "anti-scaling within Gemma3" as a
panel-wide pattern, but it's **PlotQA-driven**. On InfoVQA the ordering
reverses. Likely reading: PlotQA ground-truth values are themselves
chart-numeric and 4B's reduced visual reasoning capacity makes it lean
harder on the visible second-image digit; InfoVQA questions can be
text-heavy ("How many cities are marked in red?"), where 27B's
strength on free-form text questions doesn't help robustness.

Within Qwen2.5-VL the 7B-vs-32B gap is ≤ 4 pp on both datasets —
**Qwen scales smoothly** in robustness, Gemma doesn't.

## 4. internvl3-8b — H2 weakest in panel

InternVL3-8b is the most-robust model on PlotQA (df_wrong = 0.095).
But its wrong − correct gap is **near zero** on both datasets:

| dataset | df wrong | df correct | gap |
|---|---:|---:|---:|
| PlotQA | 0.095 | 0.087 | +0.008 pp |
| InfoVQA | 0.161 | 0.137 | +0.024 pp |

Compare other models: gap = +0.10 to +0.34 pp. This is a panel-side
analogue of the H2 collapse we observed in **thinking mode** on MathVista
(`E5e-mathvista-reasoning-evidence.md` §3.1): the binary
wrong/correct projection of confidence breaks down. For internvl3 it's
particularly notable because the model isn't a reasoning-mode VLM. Two
plausible interpretations:

1. **InternVL3's robustness is global, not confidence-modulated.** It
   resists anchor pull uniformly across base-correctness, possibly
   because its chat template / JSON-strict prompting locks the answer
   format more than other models. (Roadmap §9 caveat: InternVL3 emits
   prose despite the JSON-strict prompt; ~30 % parse-failure on E4
   Phase 1.)
2. **The wrong-base subset is too small to measure.** On PlotQA
   `em(b) = 0.019` for InternVL3 — almost all 5000 sids land in the
   wrong-base subset, and the few correct-base sids may be noise.
   But on InfoVQA `em(b) = 0.194` (wrong = 924, correct = 223) —
   the small-correct-n explanation is weaker.

Worth a follow-up §6 confidence-quartile analysis (Phase 2 P1) on
InternVL3-8b specifically to see whether the L1 entropy_top_k Q4 − Q1
gap is also compressed.

## 5. Anchor sometimes *improves* em — PlotQA "free-lunch" pattern

All-base `accuracy_exact` per arm:

| model | em(b) | em(d) | em(a-S1) | em(m-S1) | em(a) − em(b) |
|---|---:|---:|---:|---:|---:|
| gemma3-27b-it | 0.513 | 0.515 | 0.546 | 0.547 | **+3.3 pp** |
| gemma3-4b-it | 0.300 | 0.311 | 0.350 | 0.323 | **+5.0 pp** |
| llava-onevision-qwen2-7b-ov | 0.481 | 0.470 | 0.501 | 0.497 | **+2.0 pp** |
| qwen2.5-vl-32b-instruct | 0.729 | 0.731 | 0.757 | 0.753 | **+2.8 pp** |
| qwen2.5-vl-7b-instruct | 0.783 | 0.784 | 0.804 | 0.804 | **+2.1 pp** |
| internvl3-8b | 0.019 | 0.015 | 0.025 | 0.026 | +0.6 pp |
| llava-next-interleaved-7b | 0.116 | 0.107 | 0.112 | 0.113 | −0.4 pp |

6/7 models show `em(a) ≥ em(b)` on PlotQA — anchor presence is
**accuracy-positive** at the all-base level. This isn't a random
fluke: the S1 anchor cutoff `|a − GT| ≤ max(1, 0.10·GT)` means the
anchor digit is *by construction* close to the gt, so models that pick
it up as a "good guess" cue gain accuracy on samples they would
otherwise have miscounted.

This is the same un-mitigated baseline pattern that **§7.4.5 E6 Subspace
mitigation** turns into a recovery story (Stage 4-final eval shows
**Δem(a) +3.9 pp** on the calibrated cell vs the un-calibrated
baseline; see `paper-section-7-4-mitigation-free-lunch.md` and
`phase1-p0-v3-summary.md` for the strict free-lunch numbers). The E7
panel adds the side-evidence that the un-mitigated baseline is *already*
accuracy-positive on PlotQA — recovery isn't fighting an em-loss but
amplifying an em-gain.

InfoVQA shows a more mixed picture (gemma3 +2.7 pp; llava-onevision
−3.1 pp; internvl3 −5.5 pp), so the free-lunch finding **does not
generalise to chart-text-heavy datasets**. The §7.4.5 prose should
note this dataset-bound aspect.

## 6. Digit-pixel causality — (a − m) gap on df, wrong-base

| model | PlotQA (a − m) | InfoVQA (a − m) |
|---|---:|---:|
| gemma3-4b-it | +0.139 | +0.084 |
| llava-next-interleaved-7b | +0.110 | +0.073 |
| llava-onevision-qwen2-7b-ov | +0.080 | +0.016 |
| gemma3-27b-it | +0.062 | +0.086 |
| qwen2.5-vl-32b-instruct | +0.035 | +0.057 |
| qwen2.5-vl-7b-instruct | +0.019 | +0.005 |
| internvl3-8b | +0.014 | +0.026 |

7/7 models preserve `df(a) > df(m)` on both datasets — digit-pixel
causality replicates beyond the 3-model E5e MathVista panel onto a
larger (n=5000) PlotQA sample and a different chart-text axis.
The (a − m) gap correlates loosely with overall susceptibility:
high-susc models have larger digit-pixel-specific contribution.

## 7. Caveats

- **Sample size asymmetry.** PlotQA n=5000 vs InfoVQA n=1147. CIs on
  InfoVQA per-cell numbers are ~2× wider. Cross-dataset comparisons
  use point estimates only.
- **gemma3-12b-it dropped.** Empty run dir
  `experiment_e7_plotqa_full/gemma3-12b-it/20260502-115102/`. The
  4-12-27B Gemma scaling curve is missing the middle point; the
  anti-scaling claim is built on the 4B-vs-27B endpoints only. Any
  follow-up should include 12B to nail down the curve shape.
- **InfoVQA panel is 7-model on inference but the Gemma3-12B gap is
  identical** (no 12B run). Section §3 anti-scaling reading should
  not over-claim a "U-shape" without the 12B data.
- **InternVL3 prose-leak parse failure** (~30 % on E4 Phase 1, see
  roadmap §9). Some of InternVL3's apparent robustness on these
  datasets may be parse-dropout rather than genuine resistance —
  a `--max-new-tokens 8` truncation drops "based on..." prose. Check
  the per-cell `n_pb_ne_anchor` columns before quoting; the wrong-base
  ranking should survive but absolute numbers may be noisier.

## 8. Cross-link to other insight docs

- `phase1-p0-v3-summary.md` — umbrella headline panel (5-dataset, 6-7 models).
- `docs/insights/_data/main_panel_5dataset_summary.md` (gitignored) —
  numeric backing for §3.3.
- `docs/insights/L1-confidence-modulation-evidence.md` — H2/H7 monotonicity
  on E5b/E5c/E5e (4-dataset, 3-model). The Phase 2 P1 follow-up should
  re-run the L1 analyzer on this 7-model × 2-dataset E7 matrix.
- `docs/insights/E5e-mathvista-reasoning-evidence.md` — H2 collapse in
  thinking mode. The InternVL3 panel-side analogue (§4 above) is a
  different mechanism for the same surface symptom.
- `docs/insights/paper-section-7-4-mitigation-free-lunch.md` — §7.4.5
  E6 Subspace mitigation that turns the PlotQA accuracy-positive
  baseline into a strict free-lunch story.
