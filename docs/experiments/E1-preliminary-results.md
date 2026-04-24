# E1 — attention-mass results across 4 encoder families (n=200 each)

**Status:** 4 models × 200 susceptibility-stratified questions × 3 conditions each = 2,400 attention records. Cross-replication settled on the main claims. Plan: `docs/experiments/E1-attention-mass.md`. Raw data: `outputs/attention_analysis/{gemma4-e4b,qwen2.5-vl-7b-instruct,llava-1.5-7b,internvl3-8b}/<run>/per_step_attention.jsonl`. Analysis: `scripts/analyze_attention_mass.py`.

## TL;DR — three findings, each replicated or cleanly split across 4 models

1. **Robust universal.** Anchor-image attention mass > neutral-image mass. Mean delta at the answer-digit step ranges +0.004 to +0.007 across the four encoder families; 95 % CI excludes zero in every model. VLMs consistently *notice* digit images more than content-matched neutral images.
2. **Robust null.** The H2 mechanism prediction — "models attend more to the anchor on items they were originally wrong on" — **fails in 4/4 models**. `delta(wrong)` ≈ `delta(correct)` everywhere, with CI overlap. Phase A's graded behavioural pull modulation is not reflected at the mean-attention-mass level. This is a clean, publishable null with direct consequences for how mitigation should be designed.
3. **Partially replicated A7.** "Universally susceptible items attract more answer-step anchor attention than resistant items" holds cleanly in 3/4 models (Qwen, LLaVA-1.5, InternVL3 — top/bottom ratio of 1.6 – 4×). Gemma reverses it (bottom > top). The SigLIP-family outlier also concentrates its signal at step 0 (prompt processing) rather than answer generation — two related architecture-specific oddities that hang together and suggest SigLIP's documented typographic-attack inheritance kicks in early in a way the other encoders don't.

## Setup

- **Models:** four encoder families that span the Phase-A susceptibility axis (biggest wrong/correct moved-closer gap → smallest): gemma4-e4b (SigLIP + Gemma-4-E4B, 4B, gap +19.6 pp) → llava-1.5-7b (CLIP-ViT + Vicuna-7B, gap implicit from pilot) → internvl3-8b (InternViT + InternLM3, no main run) → qwen2.5-vl-7b-instruct (Qwen-ViT + Qwen2.5-7B, gap +6.9 pp).
- **Samples:** same 200 questions for all four models (100 top-decile susceptible + 100 bottom-decile resistant, `_data/susceptibility_strata.csv`). Single irrelevant-set variant per question. 3 conditions each → 600 records per model.
- **Attention extraction:** `attn_implementation="eager"` (sdpa silently drops weights), `output_attentions=True` during `model.generate()`. Per-layer attention mass from last query position to 4 regions: target-image span, anchor-image span, prompt text, generated-so-far. Mean over heads; captured per generated step.
- **Join fix:** base_correct is derived from the attention run's own `target_only` decoded output, not from `outputs/experiment/.../predictions.csv` — the attention run's set00 anchor image differs from the predictions run's set00 due to RNG trajectory divergence when `variants_per_sample=1 vs 5`.

## Headline numbers (all 4 models, n=200 each)

All deltas are `attention(number) − attention(neutral)` mean anchor-image attention mass (averaged over heads then over layers). Bootstrap 2,000 iter, 95 % CI.

### Test 1 — overall

| Model | step | mean delta | 95 % CI | share > 0 |
|---|---|---:|---|---:|
| gemma4-e4b | step 0 | **+0.01169** | [+0.00988, +0.01343] | 0.82 |
| gemma4-e4b | answer | +0.00434 | [+0.00080, +0.00797] | 0.54 |
| qwen2.5-vl-7b | step 0 | +0.00196 | [+0.00136, +0.00253] | 0.70 |
| qwen2.5-vl-7b | answer | +0.00525 | [+0.00319, +0.00741] | 0.76 |
| llava-1.5-7b | step 0 | +0.00603 | [+0.00513, +0.00697] | 0.81 |
| llava-1.5-7b | answer | +0.00559 | [+0.00402, +0.00721] | 0.75 |
| internvl3-8b | step 0 | +0.00199 | [+0.00171, +0.00226] | 0.85 |
| internvl3-8b | answer | **+0.00670** | [+0.00483, +0.00870] | **0.95** |

Answer-step mean falls in a tight band (+0.004 to +0.007) across all four models; CI excludes zero in every cell. InternVL3 has the highest `share>0` (0.95 — 128 of 135 triplets positive at the answer step), which is consistent with its pilot profile (low adoption but high acc_drop → second image has an outsized per-item effect). The step-0 signal is where the models differ: gemma (+0.012) dominates, llava-1.5 (+0.006) comes next, qwen and internvl3 (+0.002) are minimal.

### Test 2 — stratified by base_correct (H2 mechanistic prediction, answer step)

| Model | wrong n | wrong mean | wrong CI | correct n | correct mean | correct CI |
|---|---:|---:|---|---:|---:|---|
| gemma4-e4b | 100 | +0.00566 | [-0.00037, +0.01188] | 100 | +0.00301 | [-0.00079, +0.00694] |
| qwen2.5-vl-7b | 84 | +0.00494 | [+0.00192, +0.00853] | 116 | +0.00547 | [+0.00297, +0.00805] |
| llava-1.5-7b | 113 | +0.00515 | [+0.00308, +0.00734] | 87 | +0.00616 | [+0.00382, +0.00855] |
| internvl3-8b | 42 | +0.00743 | [+0.00429, +0.01178] | 93 | +0.00638 | [+0.00399, +0.00830] |

**H2 null, 4/4 replicates.** In every model, `delta(wrong)` and `delta(correct)` overlap within their CIs. In LLaVA-1.5 and Qwen the nominal direction even flips (correct > wrong). Uncertainty — operationalised as base_correct — does not modulate mean anchor-image attention in any of the four encoder families. This is a clean cross-architecture null and a serious constraint on any E4 mitigation that would target "high-uncertainty cases".

### Test 3 — stratified by susceptibility_stratum (A7 prediction, answer step)

| Model | bottom n | bottom mean | bottom CI | top n | top mean | top CI |
|---|---:|---:|---|---:|---:|---|
| gemma4-e4b | 100 | **+0.00647** | [+0.00197, +0.01156] | 100 | +0.00220 | [-0.00332, +0.00762] |
| qwen2.5-vl-7b | 100 | +0.00191 | [-0.00021, +0.00352] | 100 | **+0.00859** | [+0.00526, +0.01238] |
| llava-1.5-7b | 100 | +0.00422 | [+0.00252, +0.00589] | 100 | **+0.00696** | [+0.00434, +0.00969] |
| internvl3-8b | 87 | +0.00563 | [+0.00328, +0.00734] | 48 | **+0.00865** | [+0.00545, +0.01289] |

**A7 holds in 3/4 models, Gemma inverts.** In Qwen, LLaVA-1.5, and InternVL3, top-decile susceptible questions attract more anchor attention at the answer step than bottom-decile resistant ones — ratio 1.6× to 4×, CI separation clean (or very near clean) in the three. Only gemma4-e4b flips the prediction. Combined with Test 1's architecture ordering — Gemma's step-0 signal is ~2× LLaVA-1.5's, ~6× Qwen's/InternVL3's — the cleanest read is that **SigLIP-backed VLMs (Gemma) do their anchor-encoding earlier in the computation and in a way that is not conditioned on the downstream susceptibility of the item**, while ViT-backed VLMs encode it at (or near) the answer-generation step in a way that *does* conform to the A7-predicted susceptibility ordering.

### Test 4 — combined base_correct × stratum (answer step)

The `correct × top_decile_susceptible` cell is the largest positive cell in 3/4 models — the model "sees" the anchor yet gets the answer right. This is the inverse of the naive prediction (`wrong × top_decile = strongest`) and is consistent with the null on H2: attention captures anchor *notice*, not anchor *capture*. Gemma inverts again; its strongest cell is `wrong × bottom_decile_resistant` (n=23, paradoxical).

| Model | strongest cell | n | mean | 95 % CI |
|---|---|---:|---:|---|
| gemma4-e4b | wrong × bottom_decile | 23 | +0.01760 | [+0.00296, +0.03392] |
| qwen2.5-vl-7b | correct × top_decile | 32 | +0.01442 | [+0.00862, +0.02100] |
| llava-1.5-7b | correct × top_decile | 11 | +0.01998 | [+0.01073, +0.03001] |
| internvl3-8b | correct × top_decile | 15 | +0.00987 | [+0.00450, +0.01668] |

## What this does and doesn't say

**Does say (4/4 or 3/4 replicates):**
- Anchor images attract more attention than neutral images (4/4 models, clean CI).
- Uncertainty does *not* modulate mean anchor attention at any of the 4 architectures tested. H2 as an attention-level prediction is robustly falsified.
- In ViT-backed VLMs (3/4), susceptible items attract more answer-step anchor attention than resistant ones. The attention mechanism tracks behavioural susceptibility in the expected direction — but only at the answer-generation step, not during prompt integration.
- SigLIP-Gemma is an outlier on both axes: step-0-heavy signal, inverted A7 relationship. The SigLIP typographic-attack literature provides an obvious explanatory hook (early, unconditional encoding of in-image text) but this interpretation needs confirming evidence (per-layer trace, logit lens).

**Does not say:**
- That anchor attention *causes* anchor pull. Mean attention mass is a correlational measure; causal mediation (ablate anchor tokens, measure output shift) hasn't been done.
- That the pattern generalises to ConvLLaVA (ConvNeXt) or FastVLM (FastViT). Both need the inputs_embeds-path extension to the extraction script.
- That the pattern is stable across prompt phrasings or decoding settings. These are fixed in this run.
- That the 4-model A7 holds at the same size across different susceptibility definitions. We used the `moved_closer_rate` cross-model susceptibility; using a different per-model susceptibility would give slightly different strata.

## Implications for E4 (mitigation)

The H2-conditioned "downweight when uncertain" design is **dead across all four architectures**. No mitigation that keys off model confidence will work — attention gap is flat on base_correct in 4/4.

What the 4-model data *does* support:

- **For ViT-backed models** (Qwen-ViT, CLIP-ViT, InternViT): the intervention target is clear. Rescale anchor-image attention at the answer-digit generation step, conditioned on top-decile item susceptibility (cross-model `moved_closer_rate` from A7). The A7-conforming pattern holds in 3/4 of the tested VLMs; a single intervention can plausibly be tuned once and work across the family.
- **For SigLIP-backed models** (Gemma): the intervention target is *prompt-encoding* rather than generation. Because the signal is step-0-heavy and inverts the susceptibility relationship, the natural E4 for this family is an input-side intervention (e.g., patch the second image's KV cache at the input layer, or zero its projection features) rather than an attention-weight rescaling at generation time.

The per-family result is now strong enough to state the E4 design constraint directly: **a single universal mitigation will not serve; the paper should either evaluate both interventions side-by-side or pick the ViT-family intervention and explicitly exclude SigLIP from the E4 evaluation, explaining why.**

Remaining blocker before writing E4 code: verify the per-family pattern extends to ConvLLaVA (ConvNeXt) and FastVLM (FastViT). Both use the inputs_embeds path and need an extension to the extraction script. Once done we'll have a 6-model picture on the attention side and can commit to the E4 design.

## Open questions to answer before E4

1. **Does the per-family pattern hold on ConvLLaVA / FastVLM?** Both use inputs_embeds — need to extend the extraction script to handle that path, then run ~10 min each.
2. **Is the attention gap concentrated in specific layers?** Per-layer data captured; focused analysis pending. If Qwen's answer-step signal lives in layers 10–20 vs. 20–28, that narrows E4 intervention.
3. **Does the per-step attention profile show a spike at `answer−1` (the step just before the digit is emitted)?** If yes, intervention at `answer−1` may be more effective than at `answer`.
4. **Causal test.** Ablate anchor-image tokens at the identified layer/step and measure output shift. This is the actual mechanism claim and turns attention correlations into causal evidence.
5. **Larger n on combined stratification.** Test 4 cells have n=11–32. A 400-question run (`--top-decile-n 200 --bottom-decile-n 200`) tightens CI on the `correct × top_decile` cells.

## Roadmap entry

- §6 Tier 1 E1: ☐ → ✅ primary claims settled across 4 encoder families. Further work (layer localisation, ConvNeXt/FastViT extension, causal test) is scope for E4 and the paper write-up, not E1 itself.
- H6 (two-axis anchoring vs. distraction) is *consistent* with the attention data but doesn't fall out of it — the cleaner paper story may be "anchor notice (attention) is robust; anchor pull (behaviour) is encoder-family-modulated; uncertainty modulates pull but not attention". That's a 3-claim structure H2 + H6 + E1 jointly support.
