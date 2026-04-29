# E6 — anchor-agnostic steering-vector mitigation: design

**Status:** Pre-implementation design (2026-04-29, revised post-review).
Branch: `e6-steering-vector-mitigation`. Implementation pending — this
doc captures motivation, objective, and the Phase-0/1/2 plan; results
will land in a companion `docs/experiments/E6-steering-vector.md` +
evidence doc once PoC clears Phase 1.

**2026-04-29 revision** — corrects two design-review bugs:
1. **Model panel:** original draft named LLaVA-1.5-7b for PoC and
   referenced "E5c S1 wrong-base sids on disk" as the calibration
   source. E5c was run on `llava-next-interleaved-7b` (the §3.3 main
   panel), not `llava-1.5-7b` (the E4 mechanism panel). To preserve
   the §7.4.5 "research-demo → deployable" head-to-head against E4
   on the same model, this revision keeps the LLaVA-1.5-7b PoC and
   replaces the "use existing E5c sids" plan with a freshly-extracted
   calibration set on LLaVA-1.5-7b (~30–45 min H200, see Phase 0).
   The §3.3 main-panel narrative (`llava-next-interleaved-7b` +
   qwen2.5-vl + gemma3-27b on E5c VQAv2 + TallyQA) is kept as a
   Phase 2 cross-panel deployability check.
2. **`parsed_number` field:** older E5c runs persist `parsed_number =
   None`; numeric prediction must be reconstructed via
   `vlm_anchor.utils.extract_first_number(record['prediction'])`.
   `record['exact_match']` is `0/1` int, not `False/True` bool — filter
   wrong-base with `record['exact_match'] == 0`, not `is False`.

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
calibration is a single forward-pass sweep on a small labelled set
(VQAv2 wrong-base S1 (a, m) pairs), then inference applies a fixed
residual-stream offset universally. The labelled set is constructed
fresh on the PoC model (LLaVA-1.5-7b) since the §3.3 E5c data is on a
different model panel — see Phase 0 for the construction; ~30–45 min
H200 cost.

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

**Calibration set construction** (LLaVA-1.5-7b, no existing E5c run on
this model — extract fresh):

1. Run b condition (`target_only`) on the full VQAv2 number subset
   (n=1000 base sids per `configs/experiment.yaml`); cache
   `extract_first_number(prediction)` and ground truth → identify
   wrong-base sids `{ sid : pred_b != gt }`. Empirical wrong-base
   fraction on the §3.3 main panel is 30–40 % (`exact_match == 0`),
   so expect ~300–400 wrong-base sids on LLaVA-1.5-7b.
2. For each wrong-base sid, run two further forward passes with
   residual-stream hooks installed: condition `a-S1` (anchor) and
   `m-S1` (masked). Total ≈ 1000 (b) + 2 × ~350 (a/m) = ~1700 forward
   passes ≈ **30–45 min on H200**.

**Residual capture position.** Output of LLM decoder layer L at the
**last input token, before any generation** (canonical ActAdd position;
Turner et al. 2023 §3). Concretely: tokenise the full prompt (image
tokens + question + JSON-strict template + JSON prefix `{"result":`),
take the residual at the final position before the model is asked to
generate the digit token. No generation loop needed for calibration —
one forward pass yields all layers' residuals at the captured position.

**Pair construction.** Two `v` variants emitted side-by-side at zero
extra inference cost (advisor 2026-04-29):

```
for sid in calibration_set:
    h_a[L] = residual at L on (sid, target_plus_irrelevant_number_S1)
    h_m[L] = residual at L on (sid, target_plus_irrelevant_number_masked_S1)

v_wrong[L] = mean over wrong-base sids of  ( h_a[L] − h_m[L] )
v_all  [L] = mean over all calibration sids of  ( h_a[L] − h_m[L] )
```

Phase 1 sweeps both as a sub-axis (settles the open "wrong-base only or
all-base?" question empirically).

`v_*[L]` lives in the residual-stream embedding dim (4096 for
LLaVA-1.5-7b). Save as a `(2, n_layers, d_model)` tensor at
`outputs/e6_steering/<model>/calibration/v.pt` plus a JSON sidecar with
`{n_wrong, n_all, n_layers, d_model, model, source_run, base_acc}` for
audit.

**Sanity (Phase 0 deliverables):**

- `‖v_*[L]‖₂` per layer — expect smooth curve, peak at one or a few
  layers; `v_wrong` should have higher norm than `v_all` if the wrong-
  base filter is actually concentrating signal.
- `cos(v_*[L_a], v_*[L_b])` between adjacent layers — should be high
  (smooth across depth).
- `cos(v_wrong[L], v_all[L])` per layer — high cosine ≥ 0.9 means the
  wrong-base filter mostly scales an already-pointing direction;
  low cosine means wrong-base captures a qualitatively different
  direction (also informative).

**Phase 0.5 — wiring smoke test (10 min, before Phase 1).** On 10 held-
out wrong-base anchor-arm forward passes, install the
`_make_residual_offset_hook` (Phase 1's hook) at one layer near E1b
peak (L=16 for LLaVA-1.5) with `α=2.0` and verify:

- The output digit token (or its top-3 logit ranking) actually changes
  between baseline and steered runs.
- `mean_distance_to_anchor` on the held-out 10 doesn't explode
  (e.g., > 100 — fluency catastrophe sign).

If neither happens, Phase 1 is wired wrong (residual position, layer
indexing, hook attachment to vision tower instead of LLM stack);
diagnose before burning the 5–7 h sweep budget.

## Phase 1 — (L, α) sweep on n=200 stratified

**What.** For LLaVA-1.5-7b, run the existing E1b stratified n=200 set
(top-decile susceptible × 100 + bottom-decile resistant × 100, same set
E1d/E4 used) under **4 conditions** (b / a-S1 / m-S1 / d) × **(L × α)
grid**, with `−α · v[L]` added to the residual at layer L.

**Sweep grid:**

```
L      ∈ {2, 8, 14, 16, 18, 22, 28}      # 7 layers, dense around E1b peak L16
α      ∈ {1.0, 2.0, 4.0}                 # 3 magnitudes (geometric, raw scalar on ‖v‖)
v-var  ∈ {v_wrong, v_all}                # both calibration variants from Phase 0
```

7 × 3 × 2 = 42 (L, α, v-var) cells × 4 conditions × 200 samples = 33.6k
generations. Plus baseline (α = 0, no v): 4 × 200 = 800. Total ~34k
forward passes ≈ **5–7 h on H200** for LLaVA-1.5-7b. Resumable, same
protocol as E4.

If Phase 0.5 smoke shows the hook is wired but predictions barely move
at α = 1.0, expand α grid upward (e.g., {2, 4, 8}). If predictions move
too much at α = 1.0 (em catastrophe), grid downward
(e.g., {0.25, 0.5, 1.0}). Phase 0.5 calibrates the α range.

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

**(L*, α*, v-var*) selection rule:** smallest |α| that satisfies on `a` arm:

```
df(a, L, α)  ≤  0.9 · df(a, baseline)              # ≥ 10 % rel reduction
em(b, L, α)  ≥  em(b, baseline) − 0.02              # no-anchor invariant
em(d, L, α)  ≥  em(d, baseline) − 0.02              # no-digit invariant
em(a, L, α)  ≥  em(a, baseline) − 0.02              # anchor-arm not damaged
mean_distance_to_anchor(a, L, α)  ≤  baseline + 1.0 # fluency guard
```

If multiple cells satisfy, tiebreakers in order: (i) layer closest to
E1b peak L16 (maximum mechanism-narrative coherence); (ii) smaller |α|;
(iii) `v_wrong` over `v_all` (cleaner signal-to-noise on the
calibration set).

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

## Phase 2 — full validation (LLaVA-1.5-7b head-to-head, then panels)

If Phase 1 clears on LLaVA-1.5-7b, Phase 2 runs in two stages:

### Phase 2a — E4 head-to-head on LLaVA-1.5-7b

Full VQAv2 number subset (n=17,730 sample-instances × b/a-S1/m-S1/d ×
{baseline, steering at (L*, α*, v-var*)}) on LLaVA-1.5-7b. ≈ 142k
generations × ~0.5–1 s ≈ **20–40 h on H200** — same scale as E4 Phase 2
single model. This is the §7.4.5 paper headline.

**E4-vs-E6 head-to-head reporting** on LLaVA-1.5-7b:

| metric | baseline | E4 (s* = −3.0, upper-half attention) | E6 (L*, α*, v-var*, residual offset) |
|---|---|---|---|
| df(a) | 0.288 | 0.246 (−14.6 %) | TBD |
| em(b) | invariant | invariant (E4 verified) | **must verify** |
| em(d) | TBD | TBD (re-aggregate E4 from raw) | **must verify** |
| em(a) | 0.334 | 0.342 (+0.8 pp) | TBD |
| **inference-time anchor label needed?** | n/a | ✓ (anchor span) | **✗** |

The last row is the headline E6 contribution.

### Phase 2b — panel expansion (gated on Phase 2a)

If 2a lands cleanly, expand to:

- **E4 panel** (ConvLLaVA-7b, InternVL3-8b) — same head-to-head story,
  3-model panel. Per-model `(L*, α*, v-var*)` expected (different
  encoders).
- **§3.3 main panel cross-dataset check** (deferred candidate, not
  blocking) — calibrate `v` on llava-next-interleaved-7b VQAv2 E5c
  (data ready), apply on TallyQA E5c (data also ready). If df reduction
  generalises across datasets without re-calibration, that's a strong
  cross-domain deployability claim. Cheap because both calibration
  source and deployment target use existing E5c data — only the
  residual extraction passes are new (~30 min/dataset).

## Code structure

Three files. Two new scripts + one writeup pair.

- **`scripts/e6_steering_vector.py`** — main driver. Sub-commands:
  - `--phase calibrate` — Phase 0: run b on full VQAv2 number subset to
    identify wrong-base sids; run a-S1 / m-S1 forward passes with
    residual-stream hooks at every LLM decoder layer's last-input-token
    position; emit `v_wrong` and `v_all` to `v.pt` + sidecar. For PoC
    model (LLaVA-1.5-7b) where no E5c run exists, this generates the
    calibration data fresh; for §3.3-panel models with existing E5c
    runs, this can be sped up by reading wrong-base sids from
    `outputs/experiment_e5c_vqa/<model>/<run>/predictions.jsonl` (with
    the `parsed_number → extract_first_number(prediction)` and
    `exact_match == 0` int-not-bool fixes noted in the status block).
  - `--phase smoke` — Phase 0.5: 10-pair wiring test, ~10 min.
  - `--phase sweep` — Phase 1: 42 (L, α, v-var) cells × 4 conditions ×
    n=200 stratified. Resumable.
  - `--phase full` — Phase 2: chosen `(L*, α*, v-var*)` × 4 conditions ×
    full n=17,730 × {baseline, steered}. Resumable, same protocol as E4.

  Re-uses `_get_llm_layers` and `EagerAttentionRunner` family from
  `extract_attention_mass.py` / `causal_anchor_ablation.py`. Two new
  hook helpers:
  - `_make_residual_capture_hook(layer_idx, position)` — Phase 0:
    captures the residual-stream tensor at the last-input-token
    position on the forward pass *output* of layer `layer_idx`.
  - `_make_residual_offset_hook(layer_idx, v, alpha)` — Phase 1/2:
    adds `−α · v` to the residual at the same position on the forward
    pass output of layer `layer_idx`. Note: applies only to the
    last-input-token slice of the seq dim during the prefill forward
    (no per-decode-step application — the digit answer is short and
    one prefill-time offset is sufficient; if Phase 1 finds this
    insufficient, fall back to per-step application as a Phase 1
    escalation).

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
      base_predictions.jsonl        # Phase 0 step 1 (b on full subset)
      v.pt                          # (2, n_layers, d_model) — [v_wrong, v_all]
      v_meta.json                   # {n_wrong, n_all, n_layers, d_model, base_acc, ...}
      norms_per_layer.csv           # ‖v_wrong[L]‖, ‖v_all[L]‖, cos(v_wrong, v_all) per layer
    smoke/
      smoke_results.json            # Phase 0.5 wiring test result
    sweep_n200/
      predictions.jsonl             # 4 cond × 42 cells × 200 = ~34k records
    full_n17730/
      predictions.jsonl             # 4 cond × 2 modes × 17,730 records
  _summary/
    sweep_pareto.csv
    sweep_pareto.png
    chosen_cell.json                # {model: (L*, α*, v_var*)}
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
