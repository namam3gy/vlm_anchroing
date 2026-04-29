# E6 — anchor-agnostic steering-vector mitigation: design

**Status:** Pre-implementation design (2026-04-29, revised post-review).
Branch: `e6-steering-vector-mitigation`. Implementation pending — this
doc captures motivation, objective, and the Phase-0/1/2 plan; results
will land in a companion `docs/experiments/E6-steering-vector.md` +
evidence doc once PoC clears Phase 1.

**2026-04-29 revision history:**

- **First revision** (initial review): caught two bugs — model panel
  mismatch (original named LLaVA-1.5-7b PoC + "E5c sids on disk" but
  E5c was on llava-next-interleaved-7b) and `parsed_number = None`
  field bug.
- **Second revision** (user-driven, current): flipped PoC to
  `llava-next-interleaved-7b` itself rather than keeping LLaVA-1.5-7b
  + fresh extraction. Reasoning: (a) llava-next is the §3.3 main
  panel — paper-headline behavioral results already on this model; (b)
  E5c VQAv2 wrong-base S1 sids ready on disk (399 pairs, verified
  2026-04-29); (c) **free cross-dataset deployability check** —
  E5c TallyQA, E5e ChartQA, E5e MathVista already on this model, so
  Phase 2 can test "calibrate v on VQAv2, deploy on TallyQA / ChartQA /
  MathVista" without re-calibration; (d) cross-dataset robustness is a
  stronger §7.4.5 reviewer-defuse than same-model E4 head-to-head. E4
  comparison becomes a narrative reference (E4's published numbers in
  §7.4) rather than a re-run.

**`parsed_number` bug** (still applies). Older E5c runs persist
`parsed_number = None`; numeric prediction must be reconstructed via
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

Concrete success criteria (PoC, llava-next-interleaved-7b):

```
df(a)         decreases ≥ 10 % rel  on n=200 stratified  (anchor-arm S1)
em(b)         within ± 2 pp of baseline                  (no-anchor input)
em(d)         within ± 2 pp of baseline                  (no-digit second image)
em(a)         decreases ≤ 2 pp OR rises                  (does not damage anchor arm acc)
mean_distance_to_anchor(a)  within ± 1 unit of baseline  (fluency guard)
```

If those clear, expand to Phase 2 — cross-dataset deployability check
(VQAv2-calibrated `v` deployed on TallyQA / ChartQA / MathVista E5c
data, all already on disk for this model) plus optional E4 panel port
for head-to-head reporting.

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
(VQAv2 wrong-base S1 (a, m) pairs from existing E5c data on
`llava-next-interleaved-7b` — 399 pairs verified 2026-04-29), then
inference applies a fixed residual-stream offset universally.

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

**Calibration set construction** (llava-next-interleaved-7b, leveraging
existing E5c data):

1. Identify wrong-base sids from
   `outputs/experiment_e5c_vqa/llava-next-interleaved-7b/20260427-123331/predictions.jsonl`
   — filter `record['condition'] == 'target_only' AND record['exact_match'] == 0`.
   **Empirically verified 2026-04-29: 399 wrong-base S1 pairs on this
   model.** No fresh b-condition pass needed.
2. For each wrong-base sid, run two forward passes with residual-stream
   hooks installed: condition `a-S1` (anchor) and `m-S1` (masked).
   Total ≈ 2 × 399 = ~800 forward passes ≈ **15–20 min on H200**.

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
llava-next-interleaved-7b). Save as a `(2, n_layers, d_model)` tensor at
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
`_make_residual_offset_hook` (Phase 1's hook) at a generic mid-stack
layer (`n_layers // 2`) with `α=2.0` and verify:

- The output digit token (or its top-3 logit ranking) actually changes
  between baseline and steered runs.
- `mean_distance_to_anchor` on the held-out 10 doesn't explode
  (e.g., > 100 — fluency catastrophe sign).

If neither happens, Phase 1 is wired wrong (residual position, layer
indexing, hook attachment to vision tower instead of LLM stack);
diagnose before burning the 5–7 h sweep budget.

## Phase 1 — (L, α) sweep on n=200 stratified

**What.** For llava-next-interleaved-7b, run the existing E1b stratified n=200 set
(top-decile susceptible × 100 + bottom-decile resistant × 100, same set
E1d/E4 used) under **4 conditions** (b / a-S1 / m-S1 / d) × **(L × α)
grid**, with `−α · v[L]` added to the residual at layer L.

**Sweep grid** (`n_layers` = N is read from the model's LLM stack at
runtime; for llava-next-interleaved-7b expect N ≈ 28–32 depending on
the Qwen variant inside):

```
L      ∈ {2, N//4, N//2 - 2, N//2, N//2 + 2, 3N//4, N - 2}   # 7 layers, dense mid-stack
α      ∈ {1.0, 2.0, 4.0}                 # 3 magnitudes (geometric, raw scalar on ‖v‖)
v-var  ∈ {v_wrong, v_all}                # both calibration variants from Phase 0
```

7 × 3 × 2 = 42 (L, α, v-var) cells × 4 conditions × 200 samples = 33.6k
generations. Plus baseline (α = 0, no v): 4 × 200 = 800. Total ~34k
forward passes ≈ **5–7 h on H200** for llava-next-interleaved-7b.
Resumable, same protocol as E4. No prior E1b peak measurement on this
model (mechanism panel is the 6 different models); the L sweep is the
discovery itself.

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

If multiple cells satisfy, tiebreakers in order: (i) smaller |α|
(minimum offset magnitude); (ii) `v_wrong` over `v_all` (cleaner
signal-to-noise on the calibration set); (iii) layer closer to mid-
stack (more interpretable as "anchor signal site").

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

## Phase 2 — cross-dataset deployability + full validation

If Phase 1 clears, Phase 2 runs in two stages.

### Phase 2a — Full VQAv2 validation on llava-next-interleaved-7b

Full VQAv2 number subset (n=17,730 sample-instances × b/a-S1/m-S1/d ×
{baseline, steering at (L*, α*, v-var*)}) on llava-next-interleaved-7b.
≈ 142k generations × ~0.5–1 s ≈ **20–40 h on H200** — same scale as E4
Phase 2 single model. Tightens Phase 1's n=200 confidence intervals on
the same model.

**Reporting (paper §7.4.5 headline):**

| metric | baseline | E6 (L*, α*, v-var*, residual offset) |
|---|---|---|
| df(a) S1 | TBD (from main-panel `experiment` baseline) | TBD |
| em(b) | TBD | **must remain within ±2 pp** |
| em(d) | TBD | **must remain within ±2 pp** |
| em(a) | TBD | TBD |
| **inference-time anchor label needed?** | n/a | **✗** |

E4 narrative reference. §7.4 reports E4 on its own panel
({LLaVA-1.5-7b, ConvLLaVA-7b, InternVL3-8b}); §7.4.5 reports E6 on the
§3.3 main-panel model and notes that E4's published numbers
(df 0.288 → 0.246, −14.6 % rel) are **not directly comparable** because
the model panel is different. The E6-vs-E4 contribution is the
**axis** (anchor-label-at-inference: required → not required), not a
same-model number race.

### Phase 2b — cross-dataset deployability (the §7.4.5 closer)

The `v` calibrated in Phase 0 from VQAv2 wrong-base S1 pairs is applied
**without re-calibration** on:

| dataset | source data | scale |
|---|---|---|
| TallyQA | `outputs/experiment_e5c_tally/llava-next-interleaved-7b/<run>/predictions.jsonl` (E5c TallyQA already on disk) | n=12,000 records |
| ChartQA | `outputs/experiment_e5e_chartqa_full/llava-next-interleaved-7b/<run>/predictions.jsonl` | n=TBD |
| MathVista | `outputs/experiment_e5e_mathvista_full/llava-next-interleaved-7b/<run>/predictions.jsonl` | n=TBD |

Deployability test: rerun the (steered, baseline) ×
{anchor-arm, target_only, neutral-arm} forward passes on each dataset's
sample set with the **VQAv2-calibrated v** (no per-dataset retuning).
If df reduction generalises (≥ 5 % rel on each, em(b) / em(d) within
± 2 pp), the §7.4.5 claim is "calibrate once on VQAv2, deploy on any
numerical-VQA dataset." Cheap because the data exists; only the
forward passes with the steering hook are new.

Estimated cost: ~5–10 h H200 per dataset (smaller scales than
Phase 2a; not all sids need to be hit — match each E5e/E5c run's
sample size).

### Phase 2c (optional) — port to E4 panel for direct head-to-head

Only if §7.4.5 needs the same-model E4 vs E6 comparison after all
(reviewer pushback). PoC-grade port to LLaVA-1.5-7b: identify
wrong-base sids on LLaVA-1.5 by running b condition once (~10 min),
extract residuals on those sids' a/m S1 pairs (~20 min), reuse Phase 1
sweep harness. Total ~1 day/model. Not on the critical path.

## Code structure

Three files. Two new scripts + one writeup pair.

- **`scripts/e6_steering_vector.py`** — main driver. Sub-commands:
  - `--phase calibrate` — Phase 0: read wrong-base sids from
    `outputs/experiment_e5c_vqa/<model>/<run>/predictions.jsonl`
    (`condition == 'target_only' AND exact_match == 0`); run a-S1 /
    m-S1 forward passes with residual-stream hooks at every LLM
    decoder layer's last-input-token position; emit `v_wrong` and
    `v_all` to `v.pt` + sidecar. For models without E5c data
    (e.g., LLaVA-1.5-7b in Phase 2c), the b-condition pass is run
    inline first to identify wrong-base sids. The
    `parsed_number → extract_first_number(prediction)` and
    `exact_match == 0` int-not-bool fixes apply.
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

- **Cross-model `v` portability.** Does `v` extracted on
  llava-next-interleaved-7b transfer to other §3.3 main-panel models
  (qwen2.5-vl, gemma3-27b)? Strong cross-encoder residual-stream
  geometry claim if yes. Out of scope for the PoC; F4 future-work.
- **Cross-domain `v` portability** is now Phase 2b primary, not future
  work — VQAv2-calibrated `v` deployed on TallyQA/ChartQA/MathVista
  without retuning. Result becomes the §7.4.5 closer.
- **Per-model L\* vs encoder mechanism.** llava-next-interleaved-7b is
  not in the E1 mechanism panel (which is gemma4-e4b, qwen2.5-vl-7b,
  llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b). So no E1b
  peak prior. Phase 1 sweep discovers L\* directly. Whether L\* aligns
  with the E1b per-encoder-family pattern (mid-stack cluster ~ L16,
  Gemma early, FastVLM late) is itself a finding.
- **Projection vs. subtraction.** Subtraction (`h ← h − α·v`) is the
  ActAdd default. Projection (`h ← h − (h·v̂)·v̂`) is more principled when
  the goal is "remove the v-component" rather than "translate by α·v".
  Phase 1 evaluates subtraction; if α*-tuning is fragile, projection is
  the natural alternative and is a cheap drop-in change.

## Roadmap update on completion

- §6.5 add new row `E6 — anchor-agnostic steering vector` (status flips
  from `☐ design` → `🟡 Phase 1` → `✅ Phase 2 land`).
- §7 add P1 entry "E6 PoC on llava-next-interleaved-7b"; demote to P3 if Phase 1
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
