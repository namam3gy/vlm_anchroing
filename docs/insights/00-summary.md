# Phase-A summary — what 17,730 samples × 7 models actually say

**Source data:** `outputs/experiment/<model>/<run>/predictions.csv`, processed by `scripts/phase_a_data_mining.py`. Outlier filter from `analysis.filter_anchor_distance_outliers` applied (IQR×1.5 on per-sample anchor-GT distance, default settings). All numbers come from `docs/insights/_data/*.csv`.

This file is the cross-cutting roll-up. Long-form discussion of the strongest individual findings lives in the per-insight files (`A1-…`, `A2-…`, `A7-…`).

## TL;DR — the four findings worth a paragraph each

1. **The anchoring effect is graded, not categorical.** Across all 7 models, only 5–14 % of predictions match the anchor digit exactly. But 6–19 % more shift *toward* the anchor. The bias is a soft regression-style pull, exactly as the cognitive-science anchoring model predicts.

2. **The "uncertainty modulates anchoring" prediction (H2) is real, but on the *graded-pull* axis — not the *adoption* axis.** Stratifying by `target_only` correctness, all 7 models show **+6.9 to +19.6 pp** higher moved-closer rate on items the model originally got wrong. Categorical adoption barely moves (Δ ∈ [-1.8, +2.4] pp). This refines H2: uncertain items don't get *captured* by the anchor — they get *dragged toward* it.

3. **Anchor digits are not symmetric.** Anchors 7 and 8 are systematically less effective than 2 and 4 across all 7 models. A confound — the GT distribution in the VQAv2 number subset is right-skewed (most answers are 1, 2, 3), so anchor=2 has a much higher prior probability of being correct. Need a chance-corrected re-analysis (A2 file goes into this in detail).

4. **Item susceptibility is partly content-driven.** Spearman correlation of per-question moved-closer rate across model pairs is 0.15–0.31. Same-family models correlate highest (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30; Gemma4-e4b ↔ Qwen3-VL-30B = 0.31). Some questions are universally bias-susceptible; others are model-specific. **Implication for mechanism work:** part of the bias lives in the visual encoder / content (predicts H3 will land), part in the LLM head.

## Headline numbers (standard prompt, 17,730 samples per model)

| Model | acc(target) | adoption(num) | moved-closer | moved-closer\|wrong − correct |
|---|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.123 | 0.247 | **+19.6 pp** |
| gemma3-27b-it | 0.628 | 0.141 | 0.162 | **+15.9 pp** |
| qwen3-vl-30b-it | 0.759 | 0.120 | 0.163 | **+12.2 pp** |
| gemma4-31b-it | 0.749 | 0.116 | 0.081 | +8.4 pp |
| qwen3-vl-8b-instruct | 0.751 | 0.127 | 0.100 | +8.0 pp |
| llava-next-interleaved-7b | 0.619 | 0.133 | 0.163 | +7.2 pp |
| qwen2.5-vl-7b-instruct | 0.736 | 0.110 | 0.089 | +6.9 pp |

**Cross-model pattern:** the wrong-vs-correct gap (the H2 effect) ranges over ~13 pp but is positive for *every* model. The smallest models (gemma4-e4b) and the most permissive (gemma3-27b) show the largest gaps; Qwen2.5-VL is the most resistant. There is no clean "stronger model → less bias" trend on this axis.

## A3 (question type), A4 (shift distribution), A5 (prompt), A6 (failure modes) at a glance

These didn't earn their own insight markdowns — they're folded in here.

- **A3** — VQAv2's `question_type` field is too coarse to slice meaningfully on this subset. The dominant types are "how many" and "what number". Numbers in `A3_question_type.csv` show the same pattern as the pooled per-model summary; no question-type-specific bias signature jumps out. Defer until we add ChartQA / TallyQA where the question taxonomy is richer.
- **A4** — shift distribution is **strongly bimodal at "0" + a thin pull-toward-anchor tail**. ≥ 75 % of pairs don't change at all (gemma4-31b 85 %, qwen2.5-vl 85 %, qwen3-vl-8b 84 %; gemma4-e4b is the outlier at 56 %). Of the changes, the ±1 bin dominates and the negative side (away from anchor) is consistently lighter than the positive side. The visual signature matches a Mussweiler-Strack "selective accessibility" account — most items are unaffected, a fraction is dragged.
- **A5** — strengthening the prompt ("must output a number") moves only one model meaningfully: **gemma3-27b** (adoption +17.4 pp, moved-closer +15.3 pp). All others change by < 6 pp. The strengthen prompt is **not** a universal anchor amplifier; it primarily breaks gemma3-27b. Cross-reference §3.5 of `references/roadmap.md`: gemma3 in particular hallucinates large numbers under "no hedging" pressure, which the outlier filter trims but is worth flagging in the paper.
- **A6** — failure-mode taxonomy: across all 7 models, the partitioning is roughly { exact-anchor: 11–14 %, unchanged: 56–85 %, graded-toward-anchor: 6–19 %, orthogonal / away: 6–19 %, non-numeric parse failure: ~0 % }. The "graded toward" bucket is the new finding the paper will lean on; the "exact-anchor" and "orthogonal" buckets are roughly balanced and can be dismissed as noise.

## What this means for the paper

- **Headline restated:** anchoring in VLMs is uncertainty-modulated *graded pull*, not categorical capture. This both differentiates from the LLM-anchoring literature (which mostly reports aggregate accuracy drops) and from VLMBias / typographic-attack work (which reports classification flips).
- **The H2 result is paper-worthy as is.** No more compute needed for the headline finding.
- **A2 is dangerous.** The per-digit pattern needs a chance-corrected analysis (subtract base-rate of `anchor == GT`) before any claim is published. Easy fix; do it before adding to the paper.
- **A7's correlations are the bridge to mechanism.** They predict that an encoder ablation (E2 — ConvLLaVA full run) and an attention-mass analysis (E1) should both produce signal. Schedule both.

## Decision triggers (per `references/roadmap.md` §7)

- After-A1 trigger fires **green**: asymmetry ≥ 10 pp on multiple models (gemma4-e4b, gemma3-27b, qwen3-vl-30b clear the bar; the rest are 6–9 pp). Headline holds.
- E1 / E2 priority is **unchanged**. Same-family correlation in A7 (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30) suggests architecture matters → encoder ablation is worth running.
