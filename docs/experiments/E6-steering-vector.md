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
