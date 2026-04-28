# §4. Datasets and model panel

## 4.1 Datasets

We evaluate cross-modal numerical anchoring on four numeric VQA datasets:

| Dataset | What | GT range | Filtered subset | Why |
|---|---|---|---|---|
| VQAv2 number | natural-image counting / numeric VQA | 0-8 | 17,730 (range-restricted, single-numeric-GT) | Largest, most-studied counting VQA benchmark; primary panel |
| TallyQA | counting on natural images, harder than VQAv2 | 0-15 (we cap to ≤ 8) | ~38,000 (test number-type) | Probes counting under occlusion / ambiguity |
| ChartQA | chart QA with numeric answers | 1-1000+ (we filter to integer) | 5,390 (single-numeric-GT, integer) | Tests anchoring when target image *contains a legible answer number* |
| MathVista | math reasoning over diagrams + charts | 1-1000 (integer subset) | 385 (testmini integer) | Tests anchoring on math-reasoning-style prompts |

All datasets are filtered to integer-GT, single-numeric-answer subsets
(`require_single_numeric_gt = True` in our loader) so the
`adopt_rate` numerator (`pa == anchor`) is well-defined.

Per-dataset distance cutoffs were validated empirically on a single
reference model (E5d, llava-interleave) before the cross-model panel:

| Dataset | Cutoff rule | Status |
|---|---|---|
| VQAv2 | absolute `\|a − gt\| ≤ 5`, anchor ∈ {0..9} (range-overlap) | ✅ adopted |
| TallyQA | absolute `[0,5]` | ✅ adopted |
| ChartQA | relative `\|a − gt\| ≤ max(1, 0.10·gt)` | ✅ E5d C3-validated |
| MathVista | relative_s1 (same as ChartQA) | ✅ adopted (single-stratum design — γ-α / γ-β) |

## 4.2 Stimulus inventories

Three image inventories drive the four conditions:

- **Anchor inventory** (`a`): 128 FLUX-generated rendered-digit images
  (1024 × 1024). Filename = digit value; image content = single
  Arabic numeral on a generated scene.
- **Mask inventory** (`m`): 128 corresponding inpaints. Each PNG is
  the same scene as the anchor with the digit pixel region replaced
  by a Telea inpaint (OpenCV). The mask region is the dilated
  bounding box of the digit, detected via PaddleOCR with a
  synthetic-bbox fallback. OCR-validated post-inpaint (no detectable
  digit).
- **Neutral inventory** (`d`): 128 digit-free FLUX renders with a
  scene-stylistic distribution matched to the anchor inventory.

All three inventories are released alongside the code.

## 4.3 Model panel

Seven models on the main VQAv2 panel; subsets on each follow-up:

| Model | Params | Encoder | Role |
|---|---|---|---|
| LLaVA-1.5-7b | 7B | CLIP-ViT-L/14 | Mechanism panel (E1/E1b/E1d/E4) + Phase-A pilot |
| LLaVA-Next-Interleaved-7b | 7B | CLIP-ViT-L/14 | Main panel + E5b/E5c distance/mask reference |
| InternVL3-8b | 8B | InternViT-300M | Mechanism + E4 mid-stack-cluster |
| ConvLLaVA-7b | 7B | ConvNeXt | Mechanism + E4 mid-stack-cluster |
| FastVLM-7b | 7B | FastViT | Mechanism (4th archetype) |
| Gemma3-27b-it | 27B | SigLIP-So-400m | Main + E5e |
| Gemma4-31b-it | 31B | (Gemma4 multimodal) | Main |
| Gemma4-e4b | 4B | SigLIP | Main + E1 SigLIP archetype |
| Qwen2.5-VL-7b-Instruct | 7B | Qwen-ViT | Main + E5e |
| Qwen3-VL-8b-Instruct | 8B | (Qwen3-VL multimodal) | Main + γ-β instruct arm |
| Qwen3-VL-30b-A3B-Instruct | 30B (3B active, MoE) | (Qwen3-VL MoE) | Main |
| **Qwen3-VL-8b-Thinking** | 8B | (Qwen3-VL multimodal) | γ-β reasoning-mode arm |

The **mechanism panel** (E1 / E1b / E1d) covers six models spanning four
encoder-family archetypes (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b,
internvl3-8b, convllava-7b, fastvlm-7b), n=200 stratified per model.

The **mitigation panel** (E4 Phase 2, full validation) covers three
mid-stack-cluster models (llava-1.5-7b, convllava-7b, internvl3-8b),
17,730 base questions × 3 conditions = ~53,190 records per model.

The **γ-β reasoning-mode pair** (Qwen3-VL-8b Instruct vs. Thinking) is
single-architecture, two-checkpoint — designed for a clean
reasoning-on / reasoning-off contrast at fixed model family and
parameter count.

## 4.4 Sample sizes per experiment

| Experiment | Dataset | Models | n per model | Total records |
|---|---|---|---|---|
| Main panel (b/a/d) | VQAv2 | 7 | 17,730 × 3 cond | 372,330 |
| Strengthen-prompt | VQAv2 | 7 | 17,730 × 3 cond | 372,330 |
| E5b distance sweep | VQAv2 + TallyQA | 1 | 1,000 × 6 cond × 2 datasets | 12,000 |
| E5c digit-mask | VQAv2 + TallyQA | 1 | 1,000 × 12 cond × 2 datasets | 24,000 |
| E5 ChartQA original (3-cond) | ChartQA | 3 | 5,390 × 3 cond | 48,510 |
| E5e ChartQA full (4-cond) | ChartQA | 3 | 705 × 4 cond | 8,460 |
| E5e TallyQA full (4-cond) | TallyQA | 2 done + 1 in flight | 38,245 × 4 cond | 458,940 |
| E5e MathVista (γ-α) | MathVista | 3 | 385 × 4 cond | 4,620 |
| E5e MathVista (γ-β) | MathVista | 2 (instruct + thinking) | 385 × 4 cond | 3,080 |
| E1 attention-mass | VQAv2 stratified | 6 | 200 × 3 cond | 3,600 |
| E1d causal ablation | VQAv2 stratified | 6 | 200 × 3 cond × 7 modes | 25,200 |
| E4 Phase 1 sweep | VQAv2 stratified | 3 | 200 × 3 cond × 7 strengths | 12,600 |
| E4 Phase 2 full | VQAv2 | 3 | 17,730 × 5 cond (target_only-skip optimisation: 1 b + 2 modes × 2 anchor-arms) | 265,950 |

Total ~1.6 M model generations across all experiments at target
completion (verified column-sum on the table above; the in-flight
TallyQA × gemma3-27b cell is included in the target count).

## 4.5 Compute

All experiments run on shared 8 × NVIDIA H200 (one GPU shared with a
vLLM Qwen2.5-32B server reserving ~55 % VRAM, leaving ~60 GB usable).
Each model is loaded sequentially per experiment dir, with
`del runner; gc.collect(); torch.cuda.empty_cache()` between models so
back-to-back 8B BF16 loads do not OOM. Wall-clock for the seven-model
VQAv2 main panel is ~6 hours; full mitigation Phase 2 is ~16 hours per
mid-stack-cluster model; γ-β thinking is ~45 minutes for 1,540
generations on H200 (20.8 tok/s on Qwen3-VL-8b-Thinking).

Total compute envelope: ~5,760 GPU-hours over the project lifetime.
