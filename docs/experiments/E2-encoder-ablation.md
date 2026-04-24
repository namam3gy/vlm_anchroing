# E2 — Vision-encoder ablation: ConvLLaVA vs CLIP-ViT VLMs

**Status:** Pilot launched 2026-04-24 (`configs/experiment_encoder_pilot.yaml`, 4 models × 25 samples-per-answer = ~1,100 sample-instances each). Full run blocked on pilot success. Source: `references/roadmap.md` §6 Tier 1 E2 + H3.

## Hypothesis under test

H3 from `references/roadmap.md` §2: vision-encoder family modulates anchoring susceptibility. CLIP / SigLIP-ViT VLMs inherit a documented typographic-attack weakness (arXiv:2508.20570) — text in pixels activates concept neurons. ConvNeXt-encoder VLMs (and encoder-free models) lack the same training signal and should show a smaller cross-modal anchoring effect.

**Falsifier:** if ConvLLaVA-7B's `moved_closer_rate` and `mean_anchor_pull` are statistically equivalent to CLIP/SigLIP-ViT VLMs at matched compute scale (±10 % relative on either metric, 95 % bootstrap CI), H3 fails and the encoder-architecture story drops out of the paper.

## Experimental design

### Pilot (this run)

- Models: `convllava-7b` (ConvNeXt encoder), `llava-1.5-7b` (CLIP-ViT, vanilla baseline), `internvl3-8b` (InternViT), `fastvlm-7b` (Apple FastViT — different ViT lineage). Same prompt + decoding as `experiment.yaml`.
- 25 samples per answer × 9 answer values × 5 irrelevant sets × 3 conditions ≈ 3,375 generations per model.
- Goal: (a) sanity-check pipeline runs end-to-end on each newly integrated runner; (b) get a fast read on the cross-encoder direction-follow ordering. Pilot does **not** need bootstrap CIs — it's a feasibility / direction read.

### Full run (after pilot signs off)

- Same 4 models, `samples_per_answer=400` → 17,730 sample-instances each, matching the existing 7 main runs.
- Adds H3 evidence to the existing 7-model panel (now 11 models total spanning 4–5 vision-encoder families).
- Also captures `token_info` (logits) on every record — opens the door to the deferred A1 logit-margin re-analysis without needing a separate run. *(See `docs/insights/A1-asymmetric-on-wrong.md` "Concrete next steps".)*

### Comparison metric

Headline metric: `moved_closer_rate` stratified by `base_correct == False` (the H2 "uncertain items" subset, where Phase A showed the cleanest signal). Secondary: `adoption_rate`, `mean_anchor_pull`. All with bootstrap 95 % CIs.

Side-by-side test for H3:

```
For each model in {convllava, llava-1.5, internvl3, fastvlm} ∪ existing 7:
    compute moved_closer_rate(wrong) with 95% CI
Group by encoder family:
    ConvNeXt:  {convllava-7b}
    CLIP-ViT:  {llava-1.5-7b, llava-next-interleave-7b}
    SigLIP:    {gemma3-27b, gemma4-31b, gemma4-e4b}
    Qwen-ViT:  {qwen2.5-vl-7b, qwen3-vl-8b, qwen3-vl-30b}
    InternViT: {internvl3-8b}
    FastViT:   {fastvlm-7b}
Test: ConvNeXt mean significantly lower than CLIP-ViT and SigLIP means.
```

The lineup is unbalanced (ConvNeXt has only 1 model), so the strongest claim possible is "ConvLLaVA's susceptibility falls *outside* the 95 % CI of the CLIP/SigLIP cluster" — a directional rather than population claim. This is consistent with the precedent (Weng et al. EMNLP 2024 Main used a similar rhetorical structure for their causal-mediation ablation).

## Risks / threats to validity

- **ConvLLaVA-7B is undertrained relative to the SigLIP-Gemma models.** Lower baseline accuracy (target_only) means more "wrong" items, mechanically inflating the Phase-A asymmetry metric. Need to control for `base_acc` when comparing.
- **Encoder family vs. LLM family confound.** ConvLLaVA uses a Vicuna LLM; LLaVA-1.5 uses a different Vicuna; Qwen3-VL uses Qwen3. Differences cannot be cleanly attributed to encoder alone. Mitigation: in the writeup, present results as "encoder family + LLM family co-vary" rather than overclaiming a clean encoder effect. The within-family same-LLM Qwen3-VL pair (Qwen3-VL-30B vs Qwen3-VL-8B) gives one "LLM-only varies" comparison; ConvLLaVA vs. LLaVA-1.5 gives one "encoder-only varies" comparison (both use Vicuna-7B).
- **FastVLM emits prose despite the JSON-only prompt** (per the deleted integration notes — model characteristic, not pipeline bug). Parser rescue rate is < 100 %. Report `is_numeric_prediction` rate and exclude FastVLM from any metric computation if rate < 90 %.
- **OpenGVLab InternVL3-8B requires the `-hf` repo suffix** — already wired in the model dispatch.

## Success criteria

- Pilot completes for all 4 models with ≥ 90 % numeric-parse rate and no pipeline crashes.
- Either (a) ConvLLaVA's `moved_closer_rate(wrong)` lies clearly below the CLIP-ViT cluster (H3 supported, schedule full run with high priority) or (b) it doesn't (H3 falls — drop encoder-ablation framing, refocus on attention mass / mitigation).
- Either way: write a 1-page result note and update `references/roadmap.md` §3 status table.

## Result

*(populated when pilot finishes — see roadmap §10 changelog for date)*
