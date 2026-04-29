# E6 — anchor-agnostic steering-vector mitigation: design

**Status:** Pre-implementation design (2026-04-29). Branch:
`e6-steering-vector-mitigation`. Implementation pending — this doc captures
motivation, objective, and the Phase-0/1/2 plan; results will land in a
companion `docs/experiments/E6-steering-vector.md` + evidence doc once
PoC clears Phase 1.

## Goal

Build a **deployable** inference-time mitigation for cross-modal anchoring
that requires **zero anchor labels at inference time**. Concretely, find a
single layer `L*` and scalar `α*` such that subtracting `α* · v_{L*}` from
the residual stream reduces `direction_follow_rate(a)` by ≥ 10 % rel,
while keeping `exact_match(target_only)` and
`exact_match(target_plus_irrelevant_neutral)` invariant within ± 2 pp.

Concrete success criteria (PoC, LLaVA-1.5-7b):

```
df(a)         decreases ≥ 10 % rel  on n=200 stratified  (anchor-arm S1)
em(b)         within ± 2 pp of baseline                  (no-anchor input)
em(d)         within ± 2 pp of baseline                  (no-digit second image)
em(a)         decreases ≤ 2 pp OR rises                  (does not damage anchor arm acc)
mean_distance_to_anchor(a)  within ± 1 unit of baseline  (fluency guard)
```

If those clear, expand to Phase 2 (full n=17,730) and the E4 panel
({llava-1.5-7b, convllava-7b, internvl3-8b}) for direct E4-vs-E6 comparison.

## Motivation — the anchor-agnostic-at-inference axis

E4 (upper-half attention re-weighting) works empirically — df −5.8 to
−14.6 % rel on the mid-stack cluster — but its hook needs the
**anchor-image-token span** as runtime input. That span is reconstructible
in a research pipeline where we labelled which image is the anchor; it is
**not reconstructible at deployment**, where a user hands the model N
images and the model has no ground-truth label for "which one is the
malicious anchor". Same critique applies to two natural sharpenings of E4
(N1: digit-bbox vision-token zero-out; N2: digit-bbox-scoped attention
re-weight) — both still need digit-bbox at inference.

The right axis to factor on is **calibration vs. inference label
requirement**:

| method | calibration label | **inference label** | deployable? |
|---|---|---|---|
| E4 (current) | — | required (anchor span) | ✗ research |
| N1 / N2 (digit-bbox surgery) | — | required (digit bbox) | ✗ research |
| **E6 (steering vector)** | required ((a, m) pairs) | **none** (fixed offset) | **✓** |
| LoRA / DPO fine-tune | required | none | ✓ heavy |
| anchor-detection probe | required | none (probe inferred) | ✓ fragile |
| image-dropout self-consistency | none | none (K× inference) | ✓ slow |

E6 sits in the same deployable bucket as fine-tuning, but **train-free**:
calibration is a single forward-pass sweep on a small labelled set (E5c
S1 wrong-base pairs we already have on disk), then inference applies a
fixed residual-stream offset universally.

This factoring also reframes E4 — it is **not** dropped from the paper;
it remains the §7.4 mechanism story (validates the upper-half locus
causally). E6 is added as **§7.4.5: from research demonstration to
deployable intervention**, motivated by the deployability gap E4
inherently has. N1 and N2 are reframed as **mechanistic analysis tools**
under §7.2 (digit-pixel concentration evidence) rather than mitigations.

## Why this design

Three independently necessary properties:

1. **Single-layer scope, residual stream.** E1d showed single-layer
   *attention* ablation is null on 6/6 panel models (peak L *and* layer 0).
   That null is on the **attention pathway only** — manipulating
   `attention_mask` columns inside one layer leaves the residual stream
   intact and the rest of the stack reconstructs the anchor signal. The
   residual stream itself has not been single-layer-tested in this
   project. ActAdd-class (Turner et al., 2023, "Activation Addition") and
   the contrastive-activation-steering literature (Rimsky et al., 2024,
   "Steering Llama 2 via Contrastive Activation Addition") show
   residual-stream offsets *do* propagate downstream because they
   directly modify what later layers read. So the single-layer null from
   E1d does not refute single-layer residual-stream interventions.

2. **Train-free direction.** With (anchor, masked) sample pairs at the
   same `sample_instance_id` and S1 distance, the residual difference
   `mean(h_L | a) − mean(h_L | m)` isolates the layer-L direction the
   anchor pushes the model along, controlling for sample, target image,
   distance, and second-image-distraction (the masked arm has the same
   image with only the digit pixel inpainted out). No backprop, no
   gradients, no training set beyond the calibration pairs.

3. **Anchor-agnostic at inference.** `v_{L*}` is a fixed tensor. The
   inference hook adds `−α* · v_{L*}` to the residual at layer `L*` for
   every forward pass, regardless of input. If the input contains an
   anchor, the offset cancels part of the anchor-pull direction. If the
   input is anchor-free (`b`, `d`), the offset still applies, but if
   `v_{L*}` is approximately orthogonal to the anchor-free residual
   distribution, the prediction is unaffected. **The orthogonality
   property is testable on the calibration set** and is part of Phase 1
   acceptance.

## Phase 0 — calibration vector extraction

**Source:** existing E5c S1 wrong-base sample_instance_ids
(`pred_b != gt` AND `target_plus_irrelevant_number_S1` AND
`target_plus_irrelevant_number_masked_S1` predictions both present).
Identifiable via `outputs/experiment_e5c_vqa/<model>/<run>/predictions.jsonl`.

**Extraction:** rerun forward passes on the identified pairs **with
residual-stream hooks** at the output of each LLM decoder layer (post-attn
+ post-MLP, before next layer's LayerNorm). Residuals captured at the
**answer-step query position** — the position just before the JSON
`{"result":` template would emit a digit token. (Concretely: tokenise the
prompt + JSON prefix `{"result":`, take residuals at the last input
position of that prefix.)

**Pair construction:**

```
for sid in calibration_set:
    h_anchor[L] = residual at L on (sid, target_plus_irrelevant_number_S1)
    h_masked[L] = residual at L on (sid, target_plus_irrelevant_number_masked_S1)
v[L] = mean_sid( h_anchor[L] − h_masked[L] )    # one tensor per layer
```

`v[L]` lives in the residual-stream embedding dim (4096 for LLaVA-1.5-7b).
Save `v` as a `(n_layers, d_model)` tensor at
`outputs/e6_steering/<model>/calibration/v.pt` plus a JSON sidecar with
`{n_pairs, n_layers, d_model, source_run}` for audit.

**Cost:** PoC (LLaVA-1.5-7b, ~1000 wrong-base sids × 2 conditions × 1
forward pass for residual capture) ≈ 2k forward passes ≈ 20–30 min H200.

**Sanity:**
- `‖v[L]‖₂` per layer — expect smooth curve, peak at one or a few layers
- `cos(v[L_a], v[L_b])` between adjacent layers — should be high (smooth)
- v's residual-projection on a held-out anchor pair should reproduce the
  diff direction (cross-validation check)

## Phase 1 — (L, α) sweep on n=200 stratified

**What.** For LLaVA-1.5-7b, run the existing E1b stratified n=200 set
(top-decile susceptible × 100 + bottom-decile resistant × 100, same set
E1d/E4 used) under **4 conditions** (b / a-S1 / m-S1 / d) × **(L × α)
grid**, with `−α · v[L]` added to the residual at layer L.

**Sweep grid:**

```
L ∈ {2, 6, 10, 14, 16, 20, 24, 28}   # spans early, mid (E1b peak L16), upper
α ∈ {0.5, 1.0, 2.0, 3.0, 5.0}
```

40 (L, α) cells × 4 conditions × 200 samples = 32k generations. Plus
baseline (α=0): 4 × 200 = 800. Total ~33k forward passes ≈ 5–7 h on H200
for LLaVA-1.5-7b. Resumable, same protocol as E4.

**Why include `m` (masked) and `d` (neutral)?** Two reasons specific to
E6 (E4's Phase 1 only ran b/a/d):

- `m` is the calibration negative — applying `−α · v` to a masked-arm
  forward should leave its prediction approximately unchanged (since
  masked is the "anchor signal subtracted" reference). Drift here means
  `v` is capturing more than the anchor-pull direction.
- `d` (neutral, no digit at all) is the cleanest deployability check —
  if `em(d)` drops materially under steering, the mitigation is
  damaging anchor-free inputs and the deployable claim falls apart.

**Metrics (per (L, α, condition) cell, n=200):**

- `direction_follow_rate(a)` (C-form) — primary
- `adopt_rate(a)` — secondary
- `exact_match(b)`, `em(d)`, `em(m)`, `em(a)` — accuracy guards
- `mean_distance_to_anchor(a)` (fluency guard)
- All with bootstrap 95 % CIs (2,000 iter)

**(L*, α*) selection rule:** smallest |α| that satisfies on `a` arm:

```
df(a, L, α)  ≤  0.9 · df(a, baseline)              # ≥ 10 % rel reduction
em(b, L, α)  ≥  em(b, baseline) − 0.02              # no-anchor invariant
em(d, L, α)  ≥  em(d, baseline) − 0.02              # no-digit invariant
em(a, L, α)  ≥  em(a, baseline) − 0.02              # anchor-arm not damaged
mean_distance_to_anchor(a, L, α)  ≤  baseline + 1.0 # fluency guard
```

If multiple (L, α) satisfy, prefer the layer closest to E1b peak L16
(maximum mechanism-narrative coherence); tiebreaker: smaller |α|
(minimum offset magnitude).

**Failure escalation paths:**

- **No (L, α) clears the criteria** → try (a) per-layer-pair `v[L_1] − v[L_2]`
  difference vectors, (b) projection rather than addition (project
  residual orthogonal to `v[L*]`), (c) escalate to multi-layer steering
  (still residual-stream, still single direction per layer, still
  inference-label-free; just K layers instead of 1).
- **`em(b)` or `em(d)` drops > 2 pp** (deployability fails) → `v[L*]` is
  not orthogonal enough to anchor-free residual distribution. Try (a)
  smaller α only, (b) different L, (c) projection-onto-anchor-pull-cone
  instead of fixed offset.
- **`em(a)` drops > 2 pp** (anchor-arm damaged) → α too high; degrade to
  α/2 and re-check; if df reduction also disappears, the steering
  direction is too coarse (likely entangled with non-anchor-specific
  prediction direction).

## Phase 2 — full validation (LLaVA-1.5-7b, then E4 panel)

If Phase 1 clears on LLaVA-1.5-7b, run full VQAv2 number subset
(n=17,730 sample-instances × b/a-S1/m-S1/d × {baseline, steering at (L*,
α*)}) on LLaVA-1.5-7b first. ≈ 142k generations × ~0.5–1 s ≈ 20–40 h on
H200 — same scale as E4 Phase 2 single model.

**E4-vs-E6 head-to-head reporting** on LLaVA-1.5-7b:

| metric | baseline | E4 (s* = −3.0, upper-half attention) | E6 (L*, α*, residual offset) |
|---|---|---|---|
| df(a) | 0.288 | 0.246 (−14.6 %) | TBD |
| em(b) | (invariant in E4) | invariant | **must verify** |
| em(d) | (E4 not reported) | TBD | **must verify** |
| em(a) | 0.334 | 0.342 (+0.8 pp) | TBD |
| inference-time anchor label needed? | n/a | ✓ (anchor span) | **✗** |

The last row is the headline E6 contribution.

If LLaVA-1.5-7b lands cleanly, expand Phase 2 to ConvLLaVA-7b and
InternVL3-8b for the same 3-model E4 panel. Per-model `(L*, α*)` is
expected (different encoders, different L*).

## Code structure

Three files. Two new scripts + one writeup pair.

- **`scripts/e6_steering_vector.py`** — main driver. Sub-commands:
  - `--phase calibrate` — Phase 0: compute `v[L]` from E5c wrong-base S1
    pairs. Reads `outputs/experiment_e5c_vqa/<model>/<run>/predictions.jsonl`,
    selects pairs, runs hooked forward passes, saves `v.pt` + sidecar.
  - `--phase sweep` — Phase 1: 40 (L, α) cells × 4 conditions × n=200
    stratified. Resumable.
  - `--phase full` — Phase 2: chosen `(L*, α*)` × 4 conditions × full
    n=17,730 × {baseline, steered}. Resumable, same protocol as E4.

  Re-uses `_get_llm_layers`, `_resolve_anchor_span` (irrelevant for E6
  but kept for layer enumeration), and `EagerAttentionRunner` family
  from `extract_attention_mass.py` / `causal_anchor_ablation.py`. New
  hook helper `_make_residual_offset_hook(layer_idx, v, alpha)` — adds
  `−α · v` to the residual at the answer-step position on the forward
  pass output of layer `layer_idx`.

- **`scripts/analyze_e6_steering.py`** — Phase 1 Pareto + (L*, α*)
  selection per the rule above; Phase 2 full-validation table; E4-vs-E6
  head-to-head CSV. Pareto plot has L on x-axis, α as line series, df(a)
  on left y-axis, em(b) on right y-axis.

- **`docs/experiments/E6-steering-vector.md`** — written once Phase 1
  clears with the (L*, α*) choice. Updated with Phase 2 numbers when
  full validation lands. **This design doc is superseded by the results
  writeup at that point** but kept in git history.
- **`docs/insights/E6-steering-evidence.md`** — paper-grade evidence doc
  (mirrors `E4-mitigation-evidence.md` style) once Phase 2 lands.

### Output directory

```
outputs/e6_steering/
  <model>/
    calibration/
      v.pt                          # (n_layers, d_model) tensor
      v_meta.json                   # {n_pairs, source_run, ...}
      norms_per_layer.csv           # ‖v[L]‖₂ for sanity
    sweep_n200/
      predictions.jsonl             # 4 cond × 40 (L, α) × 200 = 32k records
    full_n17730/
      predictions.jsonl             # 4 cond × 2 modes × 17,730 records
  _summary/
    sweep_pareto.csv
    sweep_pareto.png
    chosen_layer_alpha.json         # {model: (L*, α*)}
    full_validation.csv
    e4_vs_e6_head_to_head.csv
```

## Open questions for the writeup stage

- **Cross-model `v` portability.** Does `v` extracted on LLaVA-1.5-7b
  transfer to ConvLLaVA-7b (both CLIP-mid-stack)? If yes, that is a
  paper-grade finding about cross-encoder residual-stream geometry.
  Out of scope for the PoC; flagged as F4 future-work candidate.
- **Cross-domain `v` portability.** Does `v` calibrated on VQAv2 wrong-
  base pairs reduce anchor pull on TallyQA and ChartQA? If yes, the
  steering direction is dataset-agnostic. Cheap to test on existing E5e
  data.
- **Per-model L* coherence with E1b peak.** E1b peaks: gemma4-e4b L5,
  llava-1.5 L16, convllava L16, internvl3 L14, qwen2.5-vl L22, fastvlm
  L22. If E6's `L*` aligns with E1b's per-model peak, that strengthens
  the §7 mechanism-mitigation coherence claim. If `L*` is uniformly mid-
  stack regardless of E1b peak, the residual-stream story is decoupled
  from the attention-peak story (interesting on its own).
- **Projection vs. subtraction.** Subtraction (`h ← h − α·v`) is the
  ActAdd default. Projection (`h ← h − (h·v̂)·v̂`) is more principled when
  the goal is "remove the v-component" rather than "translate by α·v".
  Phase 1 evaluates subtraction; if α*-tuning is fragile, projection is
  the natural alternative and is a cheap drop-in change.

## Roadmap update on completion

- §6.5 add new row `E6 — anchor-agnostic steering vector` (status flips
  from `☐ design` → `🟡 Phase 1` → `✅ Phase 2 land`).
- §7 add P1 entry "E6 PoC on LLaVA-1.5-7b"; demote to P3 if Phase 1
  fails. Promote to P0 if Phase 1 succeeds and §7.4.5 paper-section
  becomes the headline mitigation claim.
- §10 changelog entry at each phase landing.

## Reading dependencies

Anyone picking this up should read in order:

1. `references/roadmap.md` §6.5 (mitigation row) and §7 (priority queue)
2. `docs/insights/E1d-causal-evidence.md` — why single-layer attention
   ablation is null and why residual-stream is the unrefuted residual
3. `docs/insights/E4-mitigation-evidence.md` — current mitigation
   baseline E6 must match or exceed at narrower scope
4. `docs/insights/E1-patch-evidence.md` — digit-pixel concentration
   evidence that motivates *why* the anchor signal is localizable; used
   to argue the steering direction is well-defined
5. `scripts/causal_anchor_ablation.py` + `scripts/e4_attention_reweighting.py`
   — hook-installation patterns to fork from
6. Turner et al. 2023 "Activation Addition" (arXiv:2308.10248) and
   Rimsky et al. 2024 "Steering Llama 2 via Contrastive Activation
   Addition" (arXiv:2312.06681) for the steering-vector lineage
