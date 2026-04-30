# E6 — anchor-agnostic steering-vector mitigation (Phase 1 PoC results)

**Status (2026-04-29):** Phase 0 + Phase 0.5 + Phase 1 complete on
`llava-next-interleaved-7b`. PoC clears the design-doc selection rule
on **8 of 42 steered cells**. Chosen cell **L30 / α=1.0 / v_wrong** —
df −14.2 % rel with all em metrics within ±0.5 pp of baseline.

**Source data:**
- `outputs/e6_steering/llava-next-interleaved-7b/calibration/v.pt`
  (1000 sids, 399 wrong-base, 32 layers × 4096 d, 230 s wall)
- `outputs/e6_steering/llava-next-interleaved-7b/sweep_n200/predictions.jsonl`
  (200 sids × 4 conditions × 43 cells = 34,400 records, 6,455 s wall = 1.8 h)
- `outputs/e6_steering/_summary/sweep_pareto.csv` (per-cell metrics)
- `outputs/e6_steering/_summary/chosen_cell.json`

**Driver:** `scripts/e6_steering_vector.py --phase {calibrate,smoke,sweep}`.
**Analysis:** `scripts/analyze_e6_steering.py --model llava-next-interleaved-7b`.
**Design spec:** `docs/experiments/E6-steering-vector-design.md` (superseded
by this writeup for results, retained for design rationale).

## Headline numbers (chosen cell on n=200 stratified)

| metric | baseline (no steering) | chosen (L30, α=1.0, v_wrong) | Δ |
|---|---:|---:|---:|
| `df_a` (C-form, anchor-arm) | 0.1915 | **0.1643** | **−14.2 % rel** |
| `adopt_a` | 0.1206 | 0.1214 | +0.08 pp (≈ noise) |
| `em_b` (target_only) | 0.585 | 0.580 | −0.5 pp ✓ deployable |
| `em_d` (irrelevant_neutral) | 0.570 | 0.565 | −0.5 pp ✓ deployable |
| `em_a` (a-S1) | 0.540 | 0.540 | **invariant** |
| `em_m` (m-S1, sanity) | 0.555 | 0.560 | +0.5 pp |
| `mdist` (mean dist to anchor) | 1.105 | 1.095 | −0.01 (fluency clean) |

**Inference-time anchor labels needed: none.** `−α · v_{L=30}` is a
fixed offset added to the residual stream at the last input token of
layer 30 on every forward pass, regardless of input content.

## How this compares to E4

E4 (LLaVA-1.5-7b mid-stack-cluster panel) and E6
(llava-next-interleaved-7b, §3.3 main panel) are on **different models**,
so the table below is a side-by-side narrative reference rather than a
within-model controlled comparison. The §7.4.5 paper claim is the
inference-label axis (last row), not a same-model effect-size race.

| | E4 (current) | E6 (this work) |
|---|---|---|
| Model | LLaVA-1.5-7b (CLIP-ViT mid-stack) | llava-next-interleaved-7b (Qwen-7B + SigLIP) |
| Mechanism | upper-half attention re-weighting | single-layer residual offset |
| Layer scope | 16 layers (upper half of 32) | **1 layer (L30)** |
| Calibration | none (just s* sweep) | (a, m) S1 wrong-base pairs (free from E5c) |
| **Inference-time anchor label** | **required (anchor token span)** | **none (fixed offset)** |
| df reduction | −14.6 % rel | −14.2 % rel |
| em(target_only) | invariant | invariant |
| em(target_plus_neutral) | not reported | invariant |
| em(anchor arm) | +0.8 pp | invariant |

E6 matches E4's effect size with a single-layer scope and zero inference
labels — exactly the §7.4.5 deployable-mitigation claim.

## Phase 1 sweep — full Pareto

42 steered cells × 1 baseline = 43 cells; 7 layers × 3 alphas × 2 v-vars.

**Cells passing all selection criteria** (`df ≤ 0.9·df₀`, em_b/em_d/em_a
≥ baseline − 0.02, mdist ≤ baseline + 1.0) — sorted by df reduction:

| cell | df_a | rel% | em_b | em_d | em_a | mdist |
|---|---:|---:|---:|---:|---:|---:|
| L24_a2.0_v_all | 0.1631 | −14.8 % | 0.590 | 0.570 | 0.535 | 1.12 |
| L30_a4.0_v_wrong | 0.1631 | −14.8 % | 0.585 | 0.565 | 0.545 | 1.11 |
| **L30_a1.0_v_wrong** ⭐ | 0.1643 | **−14.2 %** | 0.580 | 0.565 | 0.540 | 1.09 |
| L08_a2.0_v_wrong | 0.1702 | −11.1 % | 0.585 | 0.565 | 0.525 | 1.11 |
| L16_a2.0_v_wrong | 0.1702 | −11.1 % | 0.585 | 0.560 | 0.525 | 1.14 |
| L18_a1.0_v_all | 0.1702 | −11.1 % | 0.585 | 0.555 | 0.525 | 1.12 |
| L30_a1.0_v_all | 0.1702 | −11.1 % | 0.585 | 0.570 | 0.540 | 1.09 |
| L30_a2.0_v_all | 0.1702 | −11.1 % | 0.585 | 0.555 | 0.535 | 1.10 |

**Tiebreaker selection** (smallest |α| → v_wrong over v_all → mid-stack):
L30_a1.0_v_wrong wins because (i) α=1.0 is the smallest |α| among
−14 %-class passers; (ii) its v_wrong sibling has equal-or-better em_a
than the v_all variants; (iii) at L30 (deepest layer in the grid) the
offset's downstream propagation is minimal — a desirable narrowness
property for the deployability claim.

**Cells that achieve > −14 % df reduction but FAIL deployability:**

| cell | df_a | rel% | em_b | em_d | em_a | mdist | failed |
|---|---:|---:|---:|---:|---:|---:|---|
| L24_a2.0_v_wrong | 0.1594 | −16.7 % | 0.536 | 0.542 | 0.526 | 1.03 | em_b −4.9 pp |

L24_a2.0_v_wrong achieves the *single largest* df reduction in the
sweep but at the cost of em_b dropping 4.9 pp — exactly the failure
mode the deployability check is designed to catch. The mitigation
would damage anchor-free inputs in deployment.

**Cells that BACKFIRE (df increases under steering):**

| cell | df_a | rel% |
|---|---:|---:|
| L24_a4.0_v_wrong | 0.2183 | **+14.0 %** |
| L02_a4.0_v_wrong | 0.1986 | +3.7 % |
| L14_a1.0_v_wrong | 0.1986 | +3.7 % |
| L14_a2.0_v_all | 0.1986 | +3.7 % |
| L16_a4.0_v_all | 0.1972 | +3.0 % |
| L14_a4.0_v_wrong | 0.1972 | +3.0 % |

The α=4 / wrong-v at L24 is the most pronounced backfire: high-α
overshoot at the depth where the residual norm is large drives the
anchor-pull direction *past* zero into the opposite regime. Reading
this cell as "the wrong sign of α=4" is the canonical failure mode of
unsweet ActAdd α tuning.

## Phase 0 — calibration findings

`v[L]` ℓ₂-norm rises smoothly from 0.03 (L=0) through 1.5 (L=15) to
**peak 7.26 at L=30**, then drops slightly to 5.5 at L=31. The mid-
stack layer where E1b's CLIP-ViT panel models peaked (L≈16) is *not*
where the residual-stream signal is strongest on llava-next-interleaved
— the residual encoding of "anchor present" accumulates roughly
monotonically through the LLM stack and is most concentrated near the
final layer.

`cos(v_wrong[L], v_all[L]) ∈ [0.96, 0.99]` across all L. The wrong-
base filter does not change the steering direction qualitatively; it
mostly scales the magnitude. Phase 1's empirical confirmation: cells
with the same (L, α) and different v-var land in the same df-reduction
band (within ±3 pp), and the chosen cell happens to use v_wrong but
v_all variants are nearly equivalent. This generalises the design-doc
hypothesis "wrong-base concentration is signal-amplifying but not
qualitatively different".

## Implications for §7.4.5 prose

1. **The deployability gap E4 has — closed.** E4 reduces df ~ −15 % rel
   on LLaVA-1.5 but needs anchor token span at inference. E6 reduces df
   ~ −14 % rel on llava-next-interleaved with **no inference-time
   labels**. Same effect size, vastly different deployment story.

2. **Single-layer residual offset suffices on this model.** E1d showed
   single-layer *attention* ablation is null on 6/6 mechanism-panel
   models, motivating E4's multi-layer choice. Phase 1 confirms the
   E1d null is attention-pathway-specific: a single-layer residual-
   stream offset at L30 reduces df by 14.2 % on llava-next-interleaved.
   The residual stream and the attention pathway are causally distinct
   single-layer intervention sites.

3. **Layer locus is encoder-family-specific.** llava-next-interleaved
   (Qwen-7B + SigLIP) has its peak ‖v‖ at L30, near the final layer.
   This is *not* the E1b mid-stack peak L16 of the CLIP-ViT mechanism
   panel. Phase 2c (LLaVA-1.5-7b head-to-head) will reveal whether the
   chosen L\* on a CLIP-ViT model lands at L16 (matches E1b attention
   peak) or at a later layer (residual story decoupled from attention
   story).

4. **adopt_a barely moves; df_a does.** Phase 1 baseline `adopt_a` is
   0.1206; chosen-cell `adopt_a` is 0.1214 — basically equal. The
   chosen cell reduces *graded pull* (df) without changing *categorical
   adoption*. This matches the project's headline "graded pull, not
   categorical capture" framing — and shows the mitigation operates on
   the gradient, not on the rare events of literal anchor copying.

## Caveats

- **n=200 stratified evaluation.** The Phase 1 numbers carry the
  bootstrap uncertainty of a 200-sid evaluation set (df_a denominator
  ≈ 141 paired-valid sids on this model — 30 % wrong-base, lower than
  the §3.3 main-panel numbers suggest because target_only is the
  baseline cell and parse-failure rate matters here). Phase 2a at full
  n=17,730 tightens these CIs by ~9× and gives the §7.4.5 paper-grade
  numbers.
- **Single model.** llava-next-interleaved-7b only. Phase 2b uses the
  pre-existing E5c TallyQA + E5e ChartQA + E5e MathVista assets on
  this same model for a free cross-dataset deployability check (no
  re-calibration). Phase 2c (optional) ports to LLaVA-1.5-7b for
  same-model E4 head-to-head.
- **mdist baseline 1.10 → chosen 1.09 — fluency unchanged**, but
  baseline mdist on llava-next is already at the floor (most predictions
  are within 1 unit of the anchor digit). The fluency-degradation
  failure mode that ConvLLaVA showed in E4 Phase 2 (mdist 2.99 → 53.5)
  is not visible at this scale; full-scale Phase 2a may show it.
- **Sign asymmetry.** L24_a4.0_v_wrong at +14.0 % df (backfire) is the
  α-overshoot regime. Production deployment would need a hard guard
  against |α| > some threshold determined per layer; the chosen
  α=1.0 sits well inside the safe region.

## Next steps

1. **Phase 2a** — full VQAv2 (n=17,730) at chosen cell. ~20–40 h H200.
   Tightens CI; lands the §7.4.5 paper headline number.
2. **Phase 2b** — VQAv2-calibrated `v` deployed on TallyQA / ChartQA /
   MathVista E5* data without re-calibration. Free cross-dataset
   deployability story.
3. **Phase 2c (optional)** — port to LLaVA-1.5-7b for direct E4
   head-to-head. ~1 day. Triggered only if reviewer pushback on
   "different model panels" framing.

## Cross-dataset tiebreaker (2026-04-29 — VQAv2 → TallyQA + ChartQA)

Phase 2b miniature: VQAv2-calibrated `v` (chosen cell L30/α=1/v_wrong)
applied unchanged to TallyQA E5c (n=346 wrong-base S1) and ChartQA E5e
(n=416 wrong-base S1). **Sign of df change reverses on cross-dataset.**

| dataset | baseline df_a | L30/α=1/v_wrong df_a | rel% | L30/α=4/v_wrong df_a | rel% |
|---|---:|---:|---:|---:|---:|
| VQAv2 (Phase 1) | 0.1915 | 0.1643 | **−14.2 %** | 0.1631 | −14.8 % |
| TallyQA | 0.2342 | 0.2472 | **+5.5 %** | 0.2388 | +2.0 % |
| ChartQA | 0.2178 | 0.2206 | +1.3 % | 0.2277 | **+4.5 %** |

Em metrics remain invariant on cross-dataset (em_b within ±0.6 pp on
all 3 datasets at both cells), so the **deployable safety claim
holds** — applying the steering universally does not damage anchor-
free inputs on any dataset. The **mitigation claim**, however, does
not transfer: VQAv2-calibrated `v` does *not* reduce anchor pull on
TallyQA or ChartQA.

Statistical caveats:
- VQAv2 paired-valid n = 141; TallyQA n = 269; ChartQA n = 349.
  Binomial SE on baseline df ≈ 3–4 pp on each. The −14 % Phase 1
  effect = −2.7 pp absolute is ≈ 1 SE; the cross-dataset +1 to +5 %
  changes are also ≈ 1 SE. **All Phase 1 effects are at the noise
  floor of n=200 stratified evaluation; Phase 2a (n=17,730) is
  required to land statistically clean numbers.**
- α=1 vs α=4 on the same dataset: differences within the same SE
  band on every dataset. Tiebreaker is essentially under-determined
  at this scale.

**Implication.** Single-direction VQAv2-calibrated `v` is not a
universal cross-domain deployable mitigation. The §7.4.5 paper
headline "calibrate once, deploy anywhere" claim does not survive.
Two failure modes are possible:

1. **Calibration source dependency.** VQAv2 wrong-base may capture a
   direction that's specific to VQAv2's question/answer distribution.
   Reverse calibration (TallyQA → VQAv2, ChartQA → VQAv2) would tell
   us if the issue is "VQAv2-specific direction" or "all per-dataset
   directions are dataset-bound".
2. **Fundamental method limitation.** Single-direction subtraction may
   simply not have enough degrees of freedom to capture the anchor-
   pull subspace as it manifests in different question distributions.
   Multi-direction PCA / LEACE / decode-time intervention could be
   structurally better fits.

Reverse-calibration test (Phase 1.5) and method-pivot decisions are
captured below.

## Multi-method search frame (2026-04-29 onwards)

The work below this point is one method in a **multi-method
mitigation search** spanning multiple sessions. The canonical tracker
is the plan file at
`~/.claude/plans/task-notification-task-id-bugsfzyep-tas-lively-dongarra.md`
(read it first when resuming). Each method's per-dataset result table
appends here as it lands, in the same format. Methods so far:

- **Method 0a — VQAv2-cal single-dir ActAdd ❌** (Phase 1 PoC above;
  works on VQAv2 in isolation, fails cross-dataset)
- **Method 0b — TallyQA-cal single-dir ActAdd ❌** (this section,
  cross-direction transfer to VQAv2 works one direction; self-test
  fails at α=1)
- **Method 0c — ChartQA-cal single-dir ActAdd** (calibration extracted
  but cross-dataset sweep deferred — low-priority sanity)
- **Method 1 — multi-direction subspace projection (next)** —
  CIPHER/VCE/RepE family; replaces single mean v with top-K SVD basis.
  Implementation queued for next session.
- **Method 2 / 3 / 4+ — fallbacks** documented in plan file and the
  "Three-method pivot plan" section above.

**Experiment policy (effective 2026-04-29).** Every new method tested
first on **TallyQA + ChartQA SUBSETS (n=100–200 wrong-base sids)**.
Only graduate to VQAv2 after cross-dataset proves out. Reverses the
prior VQAv2-first default — cross-dataset failure is the binding
problem.

**Selection rule (universal).** ≥ 5 % rel df reduction on ≥ 2 of 3
datasets, em(b) / em(d) / em(a) within ± 2 pp of baseline.

## Phase 1.5 — reverse-direction calibration results (2026-04-29)

User-driven hypothesis: VQAv2 may be the noisier calibration source;
TallyQA or ChartQA may yield a more transferable `v`. Three calibration
tensors extracted at L=30 wrong-base S1 pairs:

| source | n_wrong | n_all | ‖v_wrong[L=30]‖ | wall |
|---|---:|---:|---:|---:|
| VQAv2 (Phase 0 baseline) | 399 | 1000 | 7.27 | 230 s |
| TallyQA | 346 | 1000 | 7.24 | 298 s |
| ChartQA | 416 | 632 | 6.32 | 408 s |

**Cosine similarity between calibration directions at L=30:**

```
cos(v_VQA, v_tally)    = 0.9792   ← almost identical direction
cos(v_VQA, v_chartqa)  = 0.5594   ← partially aligned
cos(v_tally, v_chartqa) = 0.5405
```

VQAv2 and TallyQA point essentially the same residual-stream direction
(cos > 0.97 across all layers), with similar magnitudes. ChartQA
points a substantially different direction. **The ‖v‖ magnitudes are
within 15 % across all three sources** — direction-quality, not
magnitude, is the differentiator.

**v_tally tiebreaker results** (n=346 TallyQA wrong-base self-test,
n=399 VQAv2 wrong-base cross-test, baseline + L30/α=1/v_wrong +
L30/α=4/v_wrong):

| Test | df_a baseline | L30/α=1 | rel% | L30/α=4 | rel% |
|---|---:|---:|---:|---:|---:|
| v_tally → VQAv2 (cross-direction transfer) | 0.2492 | 0.2337 | **−6.2 %** | 0.2347 | −5.8 % |
| v_tally → TallyQA (self-test) | 0.2342 | 0.2444 | **+4.3 %** | 0.2293 | −2.1 % |

**Two unexpected observations.**

1. **Asymmetric cross-direction transfer.** v_tally → VQAv2 reduces
   df (cross-direction transfer works), but v_VQA → TallyQA backfires
   (cross-direction transfer fails). Combined with the cos≈0.98
   between v_VQA and v_tally, this means the *direction* is
   essentially the same but its *effect on each dataset* differs —
   the failure is in how the residual offset interacts with the
   target distribution, not in the calibrated direction itself.
2. **Self-test fails at α=1.** TallyQA-calibrated v *backfires on its
   own calibration source* at α=1 (+4.3 %), barely works at α=4
   (−2.1 %). This is the canonical signature of a noisy
   single-direction estimate — at small α the offset is dominated
   by noise (random walk, sometimes adverse direction), at larger α
   the true signal partially dominates.

**Implication.** Single-direction residual-stream subtraction is
**structurally limited** for this task. It is not a calibration-source
issue; even the best per-dataset calibration is fragile at the n=346–
399 / d=4096 SNR regime. Need to pivot to a method that either (a)
uses more degrees of freedom (multi-direction subspace), (b) adapts
per input (query-conditional), or (c) operates in a different
intervention space (decode-time, training-time).

**v_chartqa tiebreakers (deferred).** Calibration tensor extracted
(2026-04-29 wall ~7 min) but cross-dataset tiebreaker chain not yet
run; deferred to optional background task during the Method 1
implementation. Incremental evidence — does not block plan.

## Literature survey (2026-04-29 deep, paper-search-mcp across 14 axes)

22 candidate methods grouped by intervention family. Each evaluated
against the 4 hard constraints (no inference labels, cross-dataset
robustness, accuracy preservation on anchor arm, implementation cost).
Fit score 0–10. **Method ranking informs the 3-method pivot plan
(see §"Three-method pivot plan" below).**

### A. Residual-stream multi-direction & subspace (most directly applicable)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **CIPHER** (Counterfactual Image Perturbations for Hallucination Extraction & Removal) | 2603.10470 | 2026-03 | Diffusion-edited counterfactuals → contrast hidden-state shifts → low-rank subspace → project away at inference | ✓ idempotent on non-anchor | 9 |
| **VCE** (Visual Contrastive Editing) | 2604.19412 | 2026-04 | SVD-decompose contrastive activation patterns to isolate hallucination subspace; targeted parameter edits | ✓ label-free | 9 |
| **MSRS** (Multi-Subspace Representation Steering) | 2508.10599 | 2025-08 | Orthogonal subspace per attribute + dynamic weighting + token-level mechanism | ✓ no labels at inference | 8 |
| **RepE** (Representation Engineering) | 2310.01405 | 2023-10 | Top-K PCA over difference matrix; multi-direction subspace projection | ✓ | 7 |
| **LEACE** (LEAst-squares Concept Erasure) | 2306.03819 | 2023-06 (196 cites) | Closed-form affine projection; provably erases all linearly-decodable concept info while minimum-norm | ✓ idempotent | 7 |
| **Closed-Form Concept Erasure via Double Projections** | 2604.10032 | 2026-04 | Two sequential closed-form steps; constrained left-null-space transform | ✓ | 6 |

### B. Single-direction extensions (incremental over current method)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **Spherical Steering** | 2602.08169 | 2026-02 | Activation rotation on geodesic instead of addition; norm-preserving; +10 % on TruthfulQA | ✓ + confidence gate | 7 |
| **Depth-Wise Activation Steering** | 2512.07667 | 2025-12 | Gaussian schedule across layer depth; outperforms single-layer baselines on 6/7 LLaMA/Qwen/Mistral models | ✓ | 6 |
| **One-shot Optimized Steering Vectors** | 2502.18862 | 2025-02 | Optimize SVs via gradient descent on single training example; transfer via vector arithmetic | ✓ at inference | 5 |
| **CAST** (Cross-task Activation Steering Transfer) | 2507.13236 | 2025-07 | Latent-space steering for cross-task transfer; contrastive enhanced activations from high-resource to low-resource | ✓ | 6 |
| **DSO** (Direct Steering Optimization for Bias Mitigation) | 2512.15926 | 2025-12 (1 cite) | RL-optimized linear transformations for steering activations; tunable fairness/capability tradeoff; tested on **VLMs and LLMs** | ✓ | 7 |

### C. Query / input-adaptive (per-input correction)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **AFTER** (Adaptive Factual-guided Visual-Textual Editing) | 2601.01957 | 2026-01 | FAS (general steering vector) + QAO (query-adaptive offset estimator); 16.3 % reduction on AMBER on 3 LVLMs | ✓ | 8 |
| **Dual Steering** | 2602.15293 | 2026-02 | Linear-probe-based steering that optimally modifies target concept while minimizing off-target changes | ✓ | 7 |
| **CogBias** | 2604.01366 | 2026-04 | Linear probes detect cognitive bias direction; activation steering achieves 26-32 % reduction on 8 cognitive biases including **anchoring** while preserving downstream capability on 25 benchmarks; cross-architecture (Llama, Qwen) | ✓ | 8 |
| **CBMAS** (Cognitive Behavioral Modeling via Activation Steering) | 2601.06109 | 2026-01 | Continuous α-sweep + logit-lens-based bias curves + layer-site sensitivity; diagnostic tool for steering tipping points | (diagnostic) | 5 |

### D. Decode-time / contrastive decoding (different intervention paradigm)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **VCD** (Visual Contrastive Decoding) | 2311.16922 | 2023-11 | Contrast logits(I_clean) vs logits(I_distorted) at decode time; suppresses prior-driven tokens | ✓ (image distortion) | 6 (sign-issue for our anchor) |
| **Mask What Matters** | 2602.11737 | 2026-02 | VCD-extension: object-aligned auxiliary view by removing salient evidence | ✓ | 6 |
| **VGS-Decoding** (Visual Grounding Score) | 2603.20314 | 2026-03 | Per-token visual dependency via comparing distributions from clean vs distorted images; Med-VLM, +9.12 % | ✓ | 6 |
| **PAI** / **AIR** / **Modality-Bias** attention rebalancing | 2407.21771 / 2603.24058 / 2508.02419 | 2024-07 — 2026-03 | Up-weight image-token attention; AIR reports 35.1 % object-hallucination reduction | ✓ | 6 |
| **PSRD** (Phase-wise Self-Reward Decoding) | 2604.17982 | 2026-04 | Lightweight reward model distilled from LVLM; phase-wise correction during decode; LLaVA-1.5 −50 % hallucination | ✓ | 5 (heavier infra) |
| **EAZY** (Eliminate hallucinations by Zeroing image tokens) | 2503.07772 | 2025-03 | 1.5 % of image tokens with high attention drive hallucination; zero them out; +15 % on detection | ✓ | 5 (token-zeroing without anchor labels = guess) |
| **HIME** (Hallucination Insensitivity Model Editing) | 2602.18711 | 2026-02 | Per-layer Hallucination Insensitivity Score guides selective weight edits; 61.8 % reduction | (model edit) | 5 |

### E. Lightweight training (LoRA/DPO with preference pairs)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **MIA-DPO** (Multi-Image Augmented DPO) | 2410.17637 | 2024-10 | Multi-image preference alignment for LVLMs; attention-aware chosen/rejected selection; LLaVA-v1.5 +3.0 %, InternLM-XC2.5 +4.3 % on **5 multi-image benchmarks** | ✓ at deploy (LoRA adapter) | 8 |
| **Antidote** | 2504.20468 | 2025-04 | Synthetic-data preference optimization for counterfactual presupposition + object-perception hallucination; +50 % CP-Bench, +30-50 % CHAIR | ✓ | 6 |
| **TIS-DPO** (Token-level Importance Sampling DPO) | 2410.04350 | 2024-10 | Per-token reward weighting via contrastive LLM probability differences | ✓ | 5 |

### F. SAE-based feature steering (heaviest infra)

| Method | arXiv | Year | 1-line | No inf labels? | Fit |
|---|---|---|---|---|---:|
| **SCAR** (Sparse Conditioned Autoencoder) | 2411.07122 | 2024-11 | Single trained SAE module extending LLM; bidirectional concept steering | ✓ | 4 (no public SAE for llava-next) |
| **CorrSteer** | 2508.12535 | 2025-08 | SAE feature selection via correlation with sample correctness at inference | ✓ | 4 |
| **SDCV** (Sparse Autoencoder-Denoised Concept Vectors) | 2505.15038 | 2025-05 | SAE denoises noisy concept vectors from limited data; +4-16 % steering success | ✓ | 5 (would solve our limited-data noise BUT needs SAE) |
| **SAE-RSV** (SAE Refinement of Steering Vectors) | 2509.23799 | 2025-09 | SAE semantically denoises + augments steering vectors from small training data | ✓ | 5 (same issue) |

### G. Negative-result reference (cautionary)

| Method | arXiv | Year | 1-line |
|---|---|---|---|
| **No Free Lunch in Bias Mitigation** | 2511.18635 | 2025-11 | Systematic eval of 4 mitigation techniques (logit steering, activation patching, BiasEdit, prompt debiasing) across 7 model families: 31.5 % of cases show collateral degradation on **untargeted** dimensions. Warns that single-target steering can worsen unrelated biases. |

### Failure-mode mapping table

Mapping our 4 observed failure modes (single-direction noise floor,
dataset-bound direction effect, sign-reversal cross-dataset,
asymmetric cross-direction transfer) to which surveyed methods
structurally address each:

| Failure mode (E6 PoC) | Methods that structurally address |
|---|---|
| n=200 noise-floor effect (single direction noisy) | SDCV / SAE-RSV (denoise) ; multi-direction PCA / VCE / CIPHER (more DOF) |
| Dataset-bound direction (cross-dataset fails) | MSRS (orthogonal subspaces) ; CIPHER / VCE (low-rank subspace) ; AFTER QAO (per-input adaptation) ; MIA-DPO (preference signal) |
| Sign-reversal cross-dataset | AFTER QAO (input-adaptive offset) ; Dual Steering (linear-probe-guided); CogBias (cognitive-bias-specific steering) |
| Asymmetric cross-direction transfer | RepE / Multi-direction PCA pooled across datasets ; LEACE (binary concept erasure) |

### Three-method pivot plan (selected from this survey)

Decision criteria: (a) closed-form / lightest dev cost first, (b) most
directly addresses the noise-floor and dataset-bound failure modes,
(c) reuses our existing residual-capture infrastructure.

1. **Method 1 (PRIMARY) — Multi-direction subspace projection**
   (CIPHER + VCE + RepE family). Single-direction → top-K SVD basis;
   project residual orthogonal to the anchor subspace at L=30. Drop-
   in for the existing offset hook; uses our wrong-base pair
   residuals (re-extracted per-pair instead of mean).
2. **Method 2 (FALLBACK 1) — Query-adaptive offset** (AFTER QAO).
   Tiny probe estimates per-input correction direction; runs only
   if Method 1 fails.
3. **Method 3 (FALLBACK 2) — MIA-DPO LoRA**. Multi-image preference
   tuning with LoRA; runs only if Method 2 fails. Most likely to
   land statistically clean cross-dataset reductions; deployment via
   LoRA adapter.

Method-1-first rationale: cheapest implementation (½ day), directly
attacks the structural single-direction limitation, label-free at
both calibration and inference. Method 2 / 3 reserved for if Method 1
fails the cross-dataset selection rule on ≥ 2 of 3 datasets.

References (verified via paper-search-mcp on arXiv + Semantic Scholar):

- Multi-direction subspace: CIPHER [2603.10470](https://arxiv.org/abs/2603.10470) · VCE [2604.19412](https://arxiv.org/abs/2604.19412) · MSRS [2508.10599](https://arxiv.org/abs/2508.10599) · RepE [2310.01405](https://arxiv.org/abs/2310.01405) · LEACE [2306.03819](https://arxiv.org/abs/2306.03819) · Closed-Form Erasure [2604.10032](https://arxiv.org/abs/2604.10032)
- Single-direction extensions: Spherical Steering [2602.08169](https://arxiv.org/abs/2602.08169) · Depth-Wise [2512.07667](https://arxiv.org/abs/2512.07667) · One-shot SVs [2502.18862](https://arxiv.org/abs/2502.18862) · CAST [2507.13236](https://arxiv.org/abs/2507.13236) · DSO [2512.15926](https://arxiv.org/abs/2512.15926)
- Query-adaptive: AFTER [2601.01957](https://arxiv.org/abs/2601.01957) · Dual Steering [2602.15293](https://arxiv.org/abs/2602.15293) · CogBias [2604.01366](https://arxiv.org/abs/2604.01366) · CBMAS [2601.06109](https://arxiv.org/abs/2601.06109)
- Decode-time: VCD [2311.16922](https://arxiv.org/abs/2311.16922) · Mask What Matters [2602.11737](https://arxiv.org/abs/2602.11737) · VGS-Decoding [2603.20314](https://arxiv.org/abs/2603.20314) · PAI [2407.21771](https://arxiv.org/abs/2407.21771) · AIR [2603.24058](https://arxiv.org/abs/2603.24058) · Modality-Bias [2508.02419](https://arxiv.org/abs/2508.02419) · PSRD [2604.17982](https://arxiv.org/abs/2604.17982) · EAZY [2503.07772](https://arxiv.org/abs/2503.07772) · HIME [2602.18711](https://arxiv.org/abs/2602.18711) · Med-VCD [2512.01922](https://arxiv.org/abs/2512.01922) · 3D-VCD [2604.08645](https://arxiv.org/abs/2604.08645) · ConVis [2408.13906](https://arxiv.org/abs/2408.13906) · CGD [2402.15300](https://arxiv.org/abs/2402.15300) · PM [2503.10183](https://arxiv.org/abs/2503.10183) · AVCD [2505.20862](https://arxiv.org/abs/2505.20862)
- Lightweight training: MIA-DPO [2410.17637](https://arxiv.org/abs/2410.17637) · Antidote [2504.20468](https://arxiv.org/abs/2504.20468) · TIS-DPO [2410.04350](https://arxiv.org/abs/2410.04350)
- SAE-based: SCAR [2411.07122](https://arxiv.org/abs/2411.07122) · CorrSteer [2508.12535](https://arxiv.org/abs/2508.12535) · SDCV [2505.15038](https://arxiv.org/abs/2505.15038) · SAE-RSV [2509.23799](https://arxiv.org/abs/2509.23799)
- Cautionary: No Free Lunch [2511.18635](https://arxiv.org/abs/2511.18635)
- Awesome list index: [showlab/Awesome-MLLM-Hallucination](https://github.com/showlab/Awesome-MLLM-Hallucination)

---

## Method 1 — Multi-direction subspace projection (CIPHER/VCE/RepE)

**Status (2026-04-30): ❌ FAILED cross-dataset — TallyQA ✅ 11/81, ChartQA ✅ 3/81, overlap = 0**

**Hypothesis.** Replace the noisy single mean direction with a richer
K-dim subspace obtained by SVD of the pooled per-pair diff matrix
D[i,L] = h_a[i,L] − h_m[i,L]. At inference: `h ← h − α·V_K(V_K^T h)`.
Label-free at inference; closed-form calibration.

**Calibration** (2026-04-30):
- TallyQA: `calibration_tally/D_wrong.pt` — shape (400, 32, 4096), wall ~204 s
- ChartQA: `calibration_chartqa/D_wrong.pt` — shape (400, 32, 4096), wall ~204 s
- VQAv2: `calibration_vqa/D_wrong.pt` — shape (399, 32, 4096)
- Pooled SVD: `_subspace/subspace_pooled_K16.pt` — shape (32, 16, 4096)

**Sweep grid (81 cells):**
- L ∈ {16, 22, 28, 30, 31} × K ∈ {2, 4, 8, 16} × α ∈ {0.5, 1.0, 2.0, 4.0}
- Baseline (α=0) always included; 5×4×4+1 = 81 total

**Implementation:**
- `scripts/e6_steering_vector.py --phase {calibrate-subspace,smoke-subspace,sweep-subspace}`
- `scripts/e6_compute_subspace.py --scope pooled` — SVD, saves `_subspace/subspace_pooled_K16.pt`
- `scripts/analyze_e6_subspace.py` — 5 % rel-df threshold, ±2 pp em tolerance
- Branch: `e6-method1-subspace-projection`

**Selection rule:** ≥ 5 % rel df reduction AND em within ±2 pp on ≥ 2/3 datasets.

### TallyQA subset results (n=100 wrong-base sids, 2026-04-30)

- Baseline: df=0.1414, em_a=0.1616
- **11/81 cells pass** (≥5% rel df drop, em ±2 pp)
- Best cell: `L28_K02_a4.0` — df=0.0800 (−43.4% rel), em_a=0.1700 (+0.84 pp) ✅

Top passing cells (sorted by df_rel_change):

| cell | L | K | α | df_a | Δ% rel | em_a | Δpp | pass |
|---|---|---|---|---:|---:|---:|---:|---|
| L28_K02_a4.0 | 28 | 2 | 4.0 | 0.0800 | **−43.4%** | 0.1700 | +0.84 | ✅ |
| L28_K04_a2.0 | 28 | 4 | 2.0 | 0.0808 | −42.9% | 0.1616 | 0.00 | ✅ |
| L31_K08_a1.0 | 31 | 8 | 1.0 | 0.0808 | −42.9% | 0.1717 | +1.01 | ✅ |
| L31_K04_a1.0 | 31 | 4 | 1.0 | 0.0909 | −35.7% | 0.1616 | 0.00 | ✅ |
| L16_K08_a0.5 | 16 | 8 | 0.5 | 0.0909 | −35.7% | 0.1717 | +1.01 | ✅ |

Artifact: `outputs/e6_steering/llava-next-interleaved-7b/sweep_subspace_tally_pooled/_analysis/cell_summary.csv`

### ChartQA subset results (n=100 wrong-base sids, 2026-04-30)

- Baseline: df=0.1515, em_a=0.0404
- **3/81 cells pass** (≥5% rel df drop, em ±2 pp)
- Best cell: `L31_K08_a4.0` — df=0.1200 (−20.8% rel), em_a=0.0526 (+1.22 pp) ✅

Passing cells: `L16_K02_a0.5` (−6.7%), `L16_K02_a4.0` (−13.3%), `L31_K08_a4.0` (−20.8%)

Artifact: `outputs/e6_steering/llava-next-interleaved-7b/sweep_subspace_chartqa_pooled/_analysis/cell_summary.csv`

### Cross-dataset verdict — ❌ FAILED

**Zero cells pass on both TallyQA and ChartQA simultaneously.** The cells that work on
TallyQA actively hurt ChartQA and vice versa:

| cell | TallyQA df Δ | ChartQA df Δ | verdict |
|---|---:|---:|---|
| L28_K02_a4.0 (Tally best) | −43.4% | +73.3% ❌ | cross-dataset failure |
| L28_K04_a2.0 | −42.9% | +66.7% ❌ | cross-dataset failure |
| L31_K08_a4.0 (Chart best) | −67.5% (em −3.9 pp ❌) | −20.8% | em damaged on Tally |

Same failure mode as Method 0: the projection direction that suppresses anchor pull
in TallyQA's residual distribution amplifies it in ChartQA's. Pooled calibration
across 3 datasets does not resolve the cross-dataset incompatibility.

### Per-dataset result tracker

| Dataset | n | best (L, K, α) | df_a baseline | steered | Δ% rel | em_a | pass? |
|---|---:|---|---:|---:|---:|---:|---|
| TallyQA subset (n=100) | 100 | L28_K02_a4.0 | 0.1414 | 0.0800 | −43.4% | 0.1700 (+0.84pp) | ✅ 11/81 |
| ChartQA subset (n=100) | 100 | L31_K08_a4.0 | 0.1515 | 0.1200 | −20.8% | 0.0526 (+1.22pp) | ✅ 3/81 |
| Cross-dataset (≥2/3) | — | — | — | — | — | — | ❌ 0 overlap |
| VQAv2 (gated) | ✗ blocked | — | — | — | — | — | not attempted |

---

## Method 1 — Pre-Method-2 Diagnostics (2026-04-30)

Before escalating to Method 2, ran three diagnostics to characterise the failure:

### 1. Bidirectional df diagnostic

Counted cells with df↓ on BOTH TallyQA and ChartQA simultaneously (81 cells total):

| cell | T df Δ% | T em pp | T pass | C df Δ% | C em pp | C pass |
|---|---:|---:|---|---:|---:|---|
| L31_K08_a4.0 | −67.5% | −3.94pp | False | −20.8% | +1.22pp | True |
| L31_K16_a1.0 | −64.3% | +27.27pp | False | −1.0% | +1.96pp | False |
| L31_K04_a2.0 | −49.5% | −3.92pp | False | −1.0% | +0.61pp | False |
| L16_K02_a0.5 | −21.4% | −2.02pp | False | −6.7% | −0.00pp | True |
| L16_K02_a1.0 | −21.4% | +1.01pp | True | −0.0% | −0.00pp | False |

Only 5/81 cells reduce df on BOTH datasets. The near-miss is **L31_K08_a4.0**, blocked by Tally em −3.94pp (tolerance ±2pp).

**Grid refinement ruled out:** L31_K08 alpha trajectory shows ChartQA df stays POSITIVE until α≈4.0, but at α=4.0 Tally em = −3.94pp. The feasible regions for (T em ±2pp) and (C df ≤ −5%) do not overlap — no alpha ∈ [0.5, 4.0] satisfies both simultaneously.

### 2. Direction cosine similarity

Per-layer `cos(v_wrong_tally, v_wrong_chartqa)` at key layers:

| Layer | cos(T,C) | cos(T,Pooled) | cos(C,Pooled) |
|---|---:|---:|---:|
| L16 | +0.623 | +0.961 | +0.791 |
| L28 | +0.469 | +0.788 | +0.747 |
| L30 | +0.549 | +0.966 | +0.701 |
| L31 | +0.619 | +0.627 | +0.568 |

Directions differ substantially (cos ≈ 0.47–0.62 at key layers), confirming the failure is **direction mismatch**, not just alpha sensitivity. The "cos ≈ 0.98" from Phase 0 was for L=30 within-dataset directions; across-dataset the alignment is much lower.

### 3. Dataset discriminability

d' (Cohen's d) along the between-group mean direction of D_wrong residuals:

| Layer | d'(between-mean) | Interpretation |
|---|---:|---|
| L16 | 3.58 | TallyQA and ChartQA hidden states are highly separable |
| L28 | 2.39 | Separable |
| L30 | 2.63 | Separable |
| L31 | 1.74 | Moderately separable |

Datasets occupy distinct regions in residual space at all key layers. This motivates **per-input adaptation** (Method 2): a probe can distinguish TallyQA-type from ChartQA-type queries and apply the appropriate correction direction.

---

## Method 2 — Query-Adaptive Offset (AFTER QAO)

**Status (2026-04-30): ❌ FAILED — full-set validation confirms no cross-dataset overlap**

### Methodology

`h ← h − α · (v_general[L] + δ)` where:
- `v_general[L]` = mean of v_wrong across VQA + TallyQA + ChartQA calibration at layer L
- `δ = probe(q)` — linear probe mapping query representation to per-input correction
- `q` = b-arm (target_only) hidden state at last token, layer L_q
- No anchor labels at inference: probe operates only on the question + target image

**Probe training:** PCA (100 components) on b-arm reprs Q [N, d_model], then Ridge
regression (λ=1e3) from Q_pca → D_wrong per (L_q, L_target) pair. Trained on
pooled VQA + TallyQA + ChartQA calibration (N ≈ 1200 wrong-base pairs total).

**Source:** AFTER [arXiv:2601.01957, 2026-01] — 16.3% hallucination reduction on
AMBER; adapted here for cross-dataset numerical anchoring.

**Implementation:**
- `scripts/e6_query_adaptive_offset.py` — phases: `calibrate-qao`, `train-probe`,
  `smoke-qao`, `sweep-qao`
- `scripts/analyze_e6_qao.py` — M2/C-form metrics per (L_q, L_target, α) cell

**Sweep grid:** L_q ∈ {20,25,30,31} × L_target ∈ {28,30,31} × α ∈ {0.5,1.0,2.0,4.0}
= 48 steered cells + 1 baseline = 49 total.

### Per-dataset result tracker

| Dataset | n | best cell | df baseline | steered df | Δ% rel | em_a | n pass/48 |
|---|---:|---|---:|---:|---:|---:|---|
| TallyQA subset (n=100) | 100 | Lq25_Lt28_a1.0 | 0.0800 (8/100) | 0.0600 | −25.0% | 0.0800 (0.00pp) | 15/48 |
| ChartQA subset (n=100) | 100 | Lq25_Lt28_a1.0 | 0.2222 (22/99) | 0.1939 | −12.8% | 0.0505 (−1.01pp) | 15/48 |
| Cross-dataset overlap (n=100) | — | — | — | — | — | — | **⚠ 4/48 overlap — tentative pass** |
| TallyQA subset (n=100) | 100 | Lq25_Lt28_a1.0 | 0.0800 | 0.0600 | −25.0% | 0.0800 (+0.00pp) | 15/48 |
| ChartQA subset (n=100) | 100 | Lq25_Lt28_a1.0 | 0.2222 | 0.1939 | −12.8% | 0.0505 (−1.01pp) | 15/48 |
| Cross-dataset (n=100) | — | — | — | — | — | — | ⚠ 4/48 tentative overlap |
| **TallyQA full (n=346)** | **346** | **Lq30_Lt28_a0.5** | **0.1503** | **0.1358** | **−9.6%** | **0.1503 (+0.29pp)** | **1/4 pass** |
| **ChartQA full (n=416)** | **416** | Lq25_Lt31_a2.0 | 0.2260 | 0.2167 | −4.1% | 0.0511 (−0.01pp) | **0/4 pass** |
| Cross-dataset full | — | — | — | — | — | — | **❌ 0/4 overlap** |
| VQAv2 (gated) | ✗ | — | — | — | — | — | blocked by full-set fail |

**Full-set verdict (2026-04-30):** The n=100 tentative overlap evaporated at full scale.
Tally full has 1 passing cell (Lq30_Lt28_a0.5, Δ=−9.6%), but that same cell shows Δ=+0.2%
on ChartQA full (i.e., worsens). Best ChartQA reduction is −4.1% (Lq25_Lt31_a2.0), below the
−5% rel threshold. No cell satisfies the cross-dataset constraint on ≥ 2 of 3 datasets.

**Method 2 verdict: ❌ FAILED cross-dataset selection rule.** The per-input probe successfully
improves Tally in isolation but the correction direction conflicts with ChartQA. This is
the same cross-dataset distribution mismatch that killed Methods 0–1 — the probe learns
a query-adaptive direction that overfits to the Tally query distribution and mis-fires on
ChartQA queries. → Escalate to Methods 4c (LEACE), 4a (CogBias), 3 (DPO LoRA).

Pipeline timing: calibrate-qao ×3 datasets ~3 min each; train-probe 11s;
sweep n=100 Tally+ChartQA 34.9 min each; full-set n=346 Tally 18 min;
full-set n=416 ChartQA 17 min.

---

## Method 4c — LEACE Closed-Form Linear Erasure (arXiv:2306.03819)

**Status (2026-04-30): ❌ FAILED — TallyQA 0/20 pass, ChartQA 5/20 pass, cross-dataset overlap = 0**

### Methodology

LEACE (Least-squares Concept Erasure) fits a minimal-norm orthogonal projection P per
decoder layer that removes the subspace predictive of "anchor-present" condition.
At inference: `h ← h − α · (h − h @ P)` where α=1 is full erasure, α∈(0,1) partial.

**Calibration:**
- Class 0 (no anchor): Q_wrong[:N, L] (b-arm reprs, pooled: VQA+Tally+ChartQA, N=1145)
- Class 1 (with anchor): Q_wrong[:N, L] + D_wrong[:N, L] (approx a-arm = b-arm + diff)
- Fit `LeaceEraser.fit(X, Y)` per layer → save P_stack [32, 4096, 4096]
- Calibration: CPU-only, 128.5s

**Sweep grid:** L ∈ {20,25,28,30,31} × α ∈ {0.3,0.5,1.0,2.0} = 20 steered + 1 baseline

**Source paper:** Belrose et al. 2023 (arXiv:2306.03819).

**Implementation:** `scripts/e6_leace.py` (phases: calibrate-leace, smoke-leace, sweep-leace);
`scripts/analyze_e6_methods.py`

**Concept norm by layer (indicator of anchor direction strength):**
L=0: 0.0144 | L=8: 0.0867 | L=16: 0.9606 | L=24: 1.5773 (peak around L24–L31)

### Per-dataset result tracker

| Dataset | n | best cell | df baseline | steered df | Δ% rel | em_a Δpp | n pass / 20 |
|---|---:|---|---:|---:|---:|---:|---|
| TallyQA subset (n=100) | 100 | L30_a2.0 | 0.1200 (12/100) | 0.1042 (10/96) | **−13.2%** | **+5.88pp ❌** | **0/20** |
| ChartQA subset (n=100) | 99 | L30_a2.0 | 0.2121 (21/99) | 0.1313 (13/99) | **−38.1%** | +0.00pp ✅ | **5/20** |
| Cross-dataset (≥2/3) | — | — | — | — | — | — | **❌ 0 overlap** |
| VQAv2 (gated) | ✗ blocked | — | — | — | — | — | not attempted |

**ChartQA passing cells** (5 of 20): L28_a0.5 (−9.5%, +0.0pp), L28_a1.0 (−9.5%, +0.0pp),
L28_a2.0 (−19.0%, +1.0pp), L30_a1.0 (−9.5%, +0.0pp), L30_a2.0 (−38.1%, +0.0pp).

### Cross-dataset verdict — ❌ FAILED

**The root pattern is the inverse of Methods 0–2.** LEACE projection at L28–L30 helps ChartQA
substantially (up to −38.1% df) but fails Tally in two ways:
(a) most cells at L20–L28 barely change or worsen Tally df;
(b) the one cell that reduces Tally df meaningfully (L30_a2.0, −13.2%) damages accuracy (+5.88pp
em, exceeding the ±2pp tolerance).

Methods 0–2 worked on Tally but conflicted with ChartQA; Method 4c works on ChartQA but
conflicts with Tally. The same direction-mismatch structural failure (cos(T,C)≈0.47–0.62 at
key layers) manifests whether the projection is single-direction or LEACE closed-form.

Artifact: `outputs/e6_steering/llava-next-interleaved-7b/sweep_leace_{tally,chartqa}_pooled/_analysis/cell_summary.csv`

---

## Method 4a — CogBias Decode-Step Correction (arXiv:2604.01366)

**Status (2026-04-30): ⏳ IN FLIGHT**

### Methodology

CogBias (arXiv:2604.01366) applies activation steering with **dynamic alpha scheduling**:
correction fires at BOTH prefill last token AND each decode-step token, not just prefill.
This is the key difference from Methods 0–2 (prefill-only correction).

- Prefill: `h[:, -1, :] -= alpha_prefill * v_general[L]`
- Decode:  `h[:, 0, :] -= alpha_decode * v_general[L]`
- v_general = mean(v_wrong) pooled across VQA + TallyQA + ChartQA

**Rationale:** Anchoring bias may manifest during answer generation (decode steps),
not only during context encoding (prefill). CogBias-style two-phase correction captures both.

**Sweep grid:** L ∈ {20,25,28,30,31} × alpha_prefill ∈ {0.5,1.0,2.0} ×
alpha_decode ∈ {0.0,0.5,1.0,2.0} = 60 steered + 1 baseline = 61 cells.
Note: alpha_decode=0 cells are equivalent to Method 0 (prefill-only).

**Implementation:** `scripts/e6_cogbias.py` (phases: smoke-cogbias, sweep-cogbias);
`scripts/analyze_e6_methods.py`

### Per-dataset result tracker

| Dataset | n | best cell | df baseline | steered df | Δ% rel | em_a | pass? |
|---|---:|---|---:|---:|---:|---:|---|
| TallyQA subset (n=100) | — | pending | — | — | — | — | pending |
| ChartQA subset (n=100) | — | pending | — | — | — | — | pending |
| Cross-dataset (≥2/3) | — | — | — | — | — | — | pending |

---

## Method 3 — MIA-DPO LoRA Fine-Tuning (arXiv:2410.17637)

**Status (2026-04-30): ⏳ IN FLIGHT (build-pairs done; training pending)**

### Methodology

Preference pairs built from E5* predictions:
- Prompt: [target_image + anchor_image] + question (a-arm condition)
- Chosen: str(ground_truth) — correct numeric answer
- Rejected: str(anchor_value) — the irrelevant distractor number

Fine-tune llava-next-interleaved-7b with LoRA (rank=256, α=256) using TRL DPOTrainer.
Trains the model to prefer correct counts/values over anchor-biased distractor numbers.

**Training data:**
- TallyQA: 33164 pairs; ChartQA: 509 pairs; VQAv2: 653 pairs → total 34326
- Subsampled to max 5000 for training efficiency

**Implementation:** `scripts/e6_dpo_lora.py` (phases: build-pairs, train-dpo, sweep-adapter);
`scripts/analyze_e6_methods.py`

**Note:** Weakens "train-free" claim to "lightweight LoRA-adapter deployable mitigation".

### Per-dataset result tracker

| Dataset | n | df baseline | steered df | Δ% rel | em_a | pass? |
|---|---:|---:|---:|---:|---:|---|
| TallyQA subset (n=100) | — | pending | — | — | — | pending |
| ChartQA subset (n=100) | — | pending | — | — | — | pending |
| Cross-dataset (≥2/3) | — | — | — | — | — | pending |
