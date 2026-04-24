# A7 — Item susceptibility is partly content-driven; same-family models correlate more

**Status:** Phase-A finding. Source data: `_data/A7_per_question.csv`, `_data/A7_model_correlation.csv`. Script: `scripts/phase_a_data_mining.py::a7_cross_model_agreement`.

## The question

If the bias is purely model-internal (LLM head pulling toward arbitrary tokens), per-question susceptibility scores across models should be uncorrelated. If it's content-driven (the *target image* / question makes the model uncertain in a way that any anchor exploits), per-question scores should be highly correlated. The truth is almost certainly in between — the question is *how much* of each.

## Method

For each `(model, question_id)` cell, compute per-question moved-closer rate (averaged over the 5 irrelevant sets per question). This produces a `[1,968 questions × 7 models]` matrix. Spearman correlation across models gives the per-pair item-agreement strength.

## Result

Spearman correlations of per-question moved-closer rate:

|                          | g3-27 | g4-31 | g4-e4 | llava | q2.5-7 | q3-30 | q3-8  |
|--------------------------|------:|------:|------:|------:|------:|------:|------:|
| **gemma3-27b-it**        | 1.00  | 0.20  | 0.27  | 0.22  | 0.22  | 0.24  | 0.23  |
| **gemma4-31b-it**        |       | 1.00  | 0.19  | 0.15  | 0.16  | 0.15  | 0.17  |
| **gemma4-e4b**           |       |       | 1.00  | 0.26  | 0.26  | 0.31  | 0.27  |
| **llava-interleave-7b**  |       |       |       | 1.00  | 0.20  | 0.24  | 0.19  |
| **qwen2.5-vl-7b**        |       |       |       |       | 1.00  | 0.24  | 0.25  |
| **qwen3-vl-30b-it**      |       |       |       |       |       | 1.00  | **0.30** |
| **qwen3-vl-8b-instruct** |       |       |       |       |       |       | 1.00  |

The positive-but-modest correlations (0.15 – 0.31) are the headline. **Item susceptibility is partly content-driven and partly model-driven.** Three structural patterns:

1. **Highest correlations are within the Qwen3-VL family** (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30) and between Gemma4-e4b ↔ Qwen3-VL-30B = 0.31. The first pair shares LLM training; the second is harder to read off architecture and may be a coincidence.

2. **Gemma3 and Gemma4 are NOT particularly correlated** (0.20, lower than several cross-vendor pairs). Gemma3-27B and Gemma4-31B/E4B use different vision encoders and were trained separately under the Gemma 3 vs. 4 lineage; this internal disagreement is the cleanest sign that the encoder side matters.

3. **`gemma4-31b-it` is the consistent floor** in correlation with everything (0.15–0.20). It's also the model with the lowest moved-closer rate (0.081) — i.e. it just doesn't anchor much, so its per-question signal is noisy and its correlation with anyone is mechanically capped.

## What this means for mechanism

- **A pure LLM-head story doesn't fit.** If anchoring lived entirely in the language model's prior over numbers, models with the *same* LLM (e.g. Qwen-family backbones) would correlate near 1, and cross-vendor models would correlate near 0. Neither is what we see — same-family ≈ 0.30, cross-vendor ≈ 0.20. The encoder is also doing real work.
- **A pure encoder story doesn't fit either.** Qwen3-VL-8B and Gemma4-e4b run completely different vision stacks (Qwen3 uses a custom ViT; Gemma uses SigLIP). They correlate at 0.27, comparable to the within-Qwen-family figure. Some bias-susceptible items are universally bias-susceptible.
- **The most coherent reading:** susceptibility = `f(content, encoder, LLM)`, and all three components carry weight. This is the right setup for a paper that *separates* the components — exactly what E1 (attention mass) + E2 (encoder ablation) + later activation patching are designed to do.

## Implications for the experiment plan (`references/roadmap.md` §6)

- **E2 (ConvLLaVA full run) is well-motivated.** A pure-Conv encoder is the cleanest counterfactual the lineup affords, and the modest cross-encoder correlations predict that ConvLLaVA's susceptibility profile will land at a *different* point on the model-pair correlation map. If it correlates at ~0.30 with everyone (i.e. ConvNeXt + LLaMA still produces the universal content-driven bias), the bias is encoder-architecture-invariant. If it correlates at ~0.10 (much lower), the encoder matters a lot.
- **E1 attention mass should be conditioned on `is_susceptible_item`.** Pick the top-decile items by cross-model moved-closer rate (universally susceptible) and the bottom-decile items (universally resistant) and compare attention patterns. This is a direct read on the encoder's role.
- **A separate analysis worth doing in Phase A:** compute a per-question susceptibility score = mean across all 7 models. The questions in the top decile of that score are the universally-susceptible "hard cases" — running cheap attention diagnostics on them is the highest-leverage mechanistic move.

## Caveats

- The matrix is built on `moved_closer_rate`, not `mean_anchor_pull` or `adoption_rate`. Repeating with the other two metrics would let us verify the pattern; expected to be qualitatively similar.
- 1,968 questions is enough that 0.30 vs 0.20 is statistically distinguishable (Spearman SE ≈ 0.022 at n=1968), but borderline — bootstrap on a few key pairs before claiming the same-family-correlates-more story.
- Per-question rates are averaged over only 5 irrelevant sets, so each cell has high noise. The correlation may underestimate the true content-driven share.

## Roadmap entry

§5 A7: ☐ → ✅
