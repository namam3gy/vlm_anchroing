# roadmap.md — Cross-modal Anchoring in VLMs (paper-section anchored)

**Single source of truth.** Read this first before any work in `vlm_anchroing/`.
Aligned with `references/project.md §0` (paper outline) — every experiment
below is owned by one paper section. Update §3 / §6 / §7 / §10 (changelog)
at the end of every task. Do **not** duplicate this content into other docs;
link to sections of this file instead.

- **Target venue:** EMNLP 2026 Main, ARR May 25.
- **Compute envelope:** 8 × H200 (one shared with vLLM Qwen2.5-32B,
  ~60 GB usable per GPU), one month at writeup.
- **Paper outline source of truth:** `references/project.md §0`.

---

## 1. Research definition

A VLM is given a numerical VQA question + the target image. We *also* hand
it an irrelevant second image. There are four conditions per
`sample_instance`:

| Condition (canonical) | second image | role | code label |
|---|---|---|---|
| `b` (target_only) | none | baseline | `target_only` |
| `a` (anchor) | image with one digit (the anchor value) | manipulation | `target_plus_irrelevant_number(_S?)` |
| `m` (anchor mask) | same anchor image with the digit pixel region inpainted out | digit-pixel control | `target_plus_irrelevant_number_masked(_S?)` |
| `d` (neutral distractor) | digit-free FLUX-rendered image | 2-image-distraction control | `target_plus_irrelevant_neutral` |

The (`a` − `d`) gap separates **anchoring** from mere distraction; the
(`a` − `m`) gap isolates the **digit-pixel** contribution from anchor-image
background distraction; the (`a` wrong-base − `a` correct-base) gap maps
the **uncertainty modulation** (Phase A / §6).

Predictions are written `pred_b / pred_a / pred_m / pred_d`; ground truth is
`gt`; anchor value is `anchor`. Boolean flags: `pb_eq_a`, `pa_eq_a`,
`gt_eq_a`, `pa_ne_pb`, `pb_eq_gt`. See `docs/insights/M2-metric-definition-evidence.md`
§0 for the full glossary.

## 2. Hypotheses

| ID | Hypothesis | Falsifier | Evidence |
|---|---|---|---|
| **H1** | Anchor pulls the prediction beyond the neutral baseline | `direction_follow_rate(a) ≤ direction_follow_rate(d) + chance` | ✅ all 7 main models, all 3 E5e models — `direction_follow_rate(a) > direction_follow_rate(d)` significantly |
| **H2** | Anchoring is asymmetric: stronger on items the model originally got wrong | `direction_follow_rate(a)` for wrong-base ≤ correct-base | ✅ Phase A `A1`: +6.9 to +19.6 pp wrong > correct on direction-follow; M2 §5.1 confirms +0.040 mean wrong-correct gap on adopt at S0/S1 cells (22/22 wins) |
| **H3** ❌ | ConvNeXt / encoder-free encoders less susceptible than ViT | `adopt_rate(ConvNeXt)` ≈ `adopt_rate(ViT)` | ❌ Falsified at adoption (E2 pilot 2026-04-24) and per-layer levels (E1b: ConvLLaVA's peak layer L16, signature identical to LLaVA-1.5 CLIP-ViT). Replaced by depth-axis framing (E1c) |
| **H4** | Reasoning / thinking-mode reduces anchoring | thinking-on `df` ≤ thinking-off `df` | ⚠ Untested. VLMBias / LRM-judging suggests reasoning may *amplify* — write β experiment direction-agnostic |
| **H5** | "No-hedging" prompt amplifies anchor pull on uncertain items | `direction_follow` increases under strengthen | ⚠ Suggestive (gemma3-27b-it strengthen `mean_distance_to_anchor` = 2617 → hallucination, not anchor pull). Folded into §"strengthen anomaly" caveat |
| **H6** | Cross-modal failures decouple into two orthogonal axes — `anchor-pull` vs. `multi-image distraction` | `adopt_rate(a)` and `acc_drop_d_vs_b` perfectly correlated → H6 fails | ✅ Suggested by E2 pilot (InternVL3 = high acc_drop / low adopt; LLaVA-1.5 = low acc_drop / high adopt; ConvLLaVA = both). Confirmed at full E4 Phase 2 scale |
| **H7** ⚙ | `direction_follow_rate` is monotonic with `pred_b`-token logit / probability — i.e. uncertainty modulates anchor pull on a **continuous** confidence scale, of which wrong/correct (H2) is a coarse projection | `direction_follow_rate` flat across confidence quartiles | ☐ Pending §6 analysis (data captured commit `5f925b2`, no analysis yet) |

## 3. Status snapshot — where we are (2026-04-28)

The matrix below replaces the old "completed runs" / "pilot runs" / "models
integrated" tables. Status flags: ✅ done · 🟡 partial / single-model · ⏳
in-flight · ☐ not started.

### 3.1 Behavioural runs

| Experiment | Dataset | Conditions | Models | Status |
|---|---|---|---|---|
| `experiment` (standard prompt) | VQAv2 number | b/a/d | 7 (gemma3-27b-it, gemma4-31b-it, gemma4-e4b, llava-interleave-7b, qwen2.5-vl-7b, qwen3-vl-8b, qwen3-vl-30b) | ✅ M1 re-aggregated; M2 re-aggregation pending |
| `experiment_anchor_strengthen_prompt` | VQAv2 number | b/a/d | same 7 | ✅ + strengthen-anomaly caveat (§9) |
| `experiment_encoder_pilot` | VQAv2 number | b/a/d, n=1,125 | llava-1.5-7b, internvl3-8b, fastvlm-7b, convllava-7b | ✅ pilot only — full run deferred (kept) |
| `experiment_distance_vqa` (E5b) | VQAv2 | b + a×S1..S5 | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_distance_tally` (E5b) | TallyQA | b + a×S1..S5 | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_e5c_vqa` | VQAv2 | b + a×S1..S5 + m×S1..S5 + d | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_e5c_tally` | TallyQA | same | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_e5d_chartqa_validation` | ChartQA | per-dataset cutoff validation | llava-interleave-7b | ✅ S1-only relative cutoff adopted |
| `experiment_e5d_mathvista_validation` | MathVista | same | llava-interleave-7b | ⚠ C3 FAIL — see §9 (MathVista (γ) supersedes) |
| `experiment_e5e_chartqa_full` | ChartQA | b/a/m/d (S1) | llava-interleave-7b, qwen2.5-vl-7b, gemma3-27b-it | ✅ |
| `experiment_e5e_tallyqa_full` | TallyQA | b/a/m/d (S1) | same 3 | 🟡 llava + qwen2.5-vl ✅; gemma3-27b ⏳ in flight (launched 2026-04-28 00:39 UTC, contended GPU 1, ETA ~30-35h total wall) |
| `experiment_e5e_mathvista_full` (γ-α) | MathVista | b/a/m/d (S1) | llava-interleave-7b, qwen2.5-vl-7b, gemma3-27b-it | ✅ landed 2026-04-29 — `docs/insights/E5e-mathvista-evidence.md` |
| MathVista (γ-β) reasoning-mode | MathVista | b/a/m/d (S1) | qwen3-vl-8b-instruct + qwen3-vl-8b-thinking | ✅ landed 2026-04-28 — thinking *amplifies* anchor pull (S1 anchor arm, all-base, n=365: instruct adopt(a)=0.074 df(a)=0.102 → thinking adopt(a)=0.117 df(a)=0.291; ratios ×1.6 adopt, ×2.9 df). VLMBias / LRM-judging gain confirmed |
| VQAv2 4-condition (b/a/m/d) | VQAv2 | full grid cross-model | TBD | ☐ P1 (kept, time-permitting) |

### 3.2 Mechanistic runs

| Experiment | Models | n | Status |
|---|---|---|---|
| E1 attention-mass | gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b | 200 stratified | ✅ |
| E1b per-layer localisation | same 6 | 200 | ✅ — 4 archetypes (SigLIP-Gemma early, mid-stack cluster CLIP-ViT/InternViT/ConvNeXt, Qwen-ViT late, FastVLM late text-stealing) |
| E1d causal ablation | same 6 | 200 | ✅ — single-layer null on 6/6; upper-half multi-layer −5.5 to −11.5 pp on 6/6 |
| E1 digit-pixel-patch reanalysis | same 6 (reuse n=200 dump) | analysis only | ☐ P0 |
| E4 mitigation Phase 1 (sweep) | llava-1.5-7b, convllava-7b, internvl3-8b | 200 × 7 strengths | ✅ |
| E4 mitigation Phase 2 (full validation) | same 3 | 17,730 | ✅ |
| E4 generalisation to other archetypes | gemma4-e4b, qwen2.5-vl-7b, fastvlm-7b | TBD | ☐ P3 |

### 3.3 Headline numbers (C-form re-aggregation, 2026-04-28)

All numbers below use the canonical M2 metrics from §4 with the
`direction_follow_rate` numerator in **C-form**: `(pa-pb)·(anchor-pb) > 0`.
The previously-published anchor·gt form numbers (committed 2026-04-29 as
"M2 re-aggregation") were a buggy carry-over from the M1 era; see §10
changelog for the correction entry. Pre-refactor results are archived at
`outputs/before_C_form/` for audit. Side-by-side before/after deltas:
`docs/insights/C-form-migration-report.md`. Adopt and exact-match are
unchanged by the refactor; only `direction_follow*` columns moved.

#### Standard-prompt VQAv2 number subset, 17,730 samples / model

| Model | acc(b) | acc(d) | acc(a) | adopt(a) | direction_follow(a) C-form |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.505 | 0.541 | **0.066** | **0.274** |
| llava-interleave-7b | 0.619 | 0.577 | 0.576 | **0.053** | **0.172** |
| gemma3-27b-it | 0.628 | 0.623 | 0.633 | **0.053** | **0.167** |
| qwen3-vl-30b | 0.759 | 0.709 | 0.707 | **0.039** | **0.170** |
| qwen3-vl-8b | 0.751 | 0.709 | 0.715 | **0.033** | **0.104** |
| qwen2.5-vl-7b | 0.736 | 0.708 | 0.711 | **0.021** | **0.094** |
| gemma4-31b-it | 0.749 | 0.723 | 0.741 | **0.024** | **0.085** |

Under C-form, `direction_follow_rate_raw == direction_follow_rate_moved`
because `(pa-pb) = 0` makes the no-movement case yield zero in the
numerator structurally — the `pa != pb` clause is structurally redundant
but kept explicit for clarity. Ranking is preserved across the panel.

#### E5b distance sweep — `llava-interleave-7b` only (cross-model in flight)

Wrong-base subset, `adopt_rate` (M2):

| stratum | VQAv2 (n_eligible) | TallyQA (n_eligible) |
|---|---:|---:|
| S1 | **0.134** (313) | **0.098** (265) |
| S2 | 0.030 | 0.006 |
| S3 | 0.010 | 0.003 |
| S4 | 0.010 | 0.000 |
| S5 | 0.003 | 0.000 |

Pattern: sharp peak at S1, decay to noise floor by S5 (cross-dataset).

#### E5c digit-pixel causality — `llava-interleave-7b` only (cross-model in flight)

Wrong-base S1 `adopt_rate` gap (anchor − masked) under M2:

| dataset | anchor S1 | masked S1 | gap (anchor − masked) |
|---|---:|---:|---:|
| VQAv2 | 0.139 | 0.073 | **+6.62 pp** |
| TallyQA | 0.114 | 0.088 | **+2.57 pp** |

#### E5e S1-only 4-condition full — 3-model panel × ChartQA + TallyQA

All-base, S1 anchor / masked, C-form (numbers cross-checked against
`outputs/experiment_e5e_*_full/<model>/<ts>/summary.json` and
`docs/insights/_data/experiment_e5e_*_per_cell.csv` 2026-04-28):

| dataset | model | acc(b) | acc(a) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
|---|---|---:|---:|---:|---:|---:|---:|
| ChartQA | gemma3-27b-it | 0.217 | 0.218 | **0.037** | 0.022 | **0.096** | 0.079 |
| ChartQA | llava-interleave-7b | 0.113 | 0.110 | **0.028** | 0.009 | **0.152** | 0.115 |
| ChartQA | qwen2.5-vl-7b | 0.255 | 0.253 | **0.017** | 0.013 | **0.051** | 0.046 |
| TallyQA | llava-interleave-7b | 0.236 | 0.233 | **0.026** | 0.014 | **0.066** | 0.056 |
| TallyQA | qwen2.5-vl-7b | 0.230 | 0.226 | **0.011** | 0.011 | **0.029** | 0.030 |

TallyQA × gemma3-27b-it cell is in flight on GPU 1 (launched 2026-04-28
00:39, full 38,245-question integer subset, contended GPU, ETA ~30-35h
total wall, ~15-20h remaining).

E1d upper-half ablation: −5.5 to −11.5 pp `direction_follow` on 6/6 models;
fluency-clean on 4/6 (mid-stack cluster + Qwen).

E4 Phase 2 full mid-stack-cluster: `direction_follow_rate` reduction
LLaVA-1.5 −17.7 % rel, ConvLLaVA −10.6 %, InternVL3 −5.8 %; `exact_match`
rises +0.49 to +1.30 pp; `accuracy_vqa(b)` invariant — anchor-condition
specific.

## 4. Canonical metrics (M2)

Settled in `docs/insights/M2-metric-definition-evidence.md`. Paper headline:

```
adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pa-pb)·(anchor-pb) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
anchor_effect_M       = M(a-arm) - M(d-arm)
```

Same form on the m-arm (substitute `pred_m`). `exact_match` is a per-arm
accuracy; `anchor_effect_M` is a per-pair gap.

The 18 (numerator × denominator) variants behind these choices are tabled
in M2 §3; equivalence collapses, signal-preservation evidence (5.1 wrong >
correct on S0/S1; 5.2 S1 > S5 on adopt; 5.3 anchor > masked) and the per
section role of each metric live in M2 §5–§6.

`metrics.py` refactor (M2 codification — §8) is pending user signoff; the
re-aggregation re-runs from raw `predictions.jsonl` only (no re-inference).

## 5. Phase A — closed (no new compute)

| ID | Question | Output | Status |
|---|---|---|---|
| **A1** | Asymmetric anchoring on wrong cases | `docs/insights/A1-asymmetric-on-wrong.md` | ✅ adoption symmetric, graded direction-follow +6.9 to +19.6 pp wrong > correct |
| **A2** | Per-anchor-value pull | `docs/insights/A2-per-anchor-digit.md` | ✅ digits 1/2/4 sticky, 7/8 inert; LLaVA × anchor=2 = 0.30 |
| **A3** | Question-type stratification | folded into `00-summary.md` | ✅ negative — defer until ChartQA / TallyQA |
| **A4** | Per-pair shift histogram | folded | ✅ bimodal — ≥75% no change + thin pull-toward-anchor tail |
| **A5** | Strengthen vs. standard prompt | folded | ✅ only gemma3-27b moves substantially +17.4 pp adoption |
| **A6** | Failure-mode taxonomy | `_data/A6_failure_modes.csv` | ✅ |
| **A7** | Cross-model item agreement | `docs/insights/A7-cross-model-agreement.md` | ✅ Spearman ρ ∈ [0.15, 0.31]; partly content-driven |

Driver: `scripts/phase_a_data_mining.py`. Phase A's contribution to the
paper is concentrated in §1 / §6 (uncertainty modulation, A1 → continuous
in H7).

## 6. Per-section experiments — status and pending work

Anchored to the paper outline in `references/project.md §0.2`. Each row
is one experiment; status reflects what is done versus what is in flight
or pending. Priorities (P0 / P1 / P2 / P3) are listed in §7.

### 6.1 §3 — Problem definition + canonical metrics

| ID | Task | Status | Notes |
|---|---|---|---|
| **M2-evidence** | Metric-definition analysis (18 variants × known-signal preservation) | ✅ `docs/insights/M2-metric-definition-evidence.md`, 2026-04-28 | re-runs as more `predictions.jsonl` arrive; recommendation is rank-driven, robust |
| **M2-refactor** | `metrics.py` refactor + re-aggregation | ✅ landed 2026-04-29; C-form follow-up landed 2026-04-28 | `summarize_condition` rates use `D_paired` denominator; `direction_follow_rate` is C-form `(pa-pb)·(anchor-pb) > 0 AND pa != pb`; 53 run dirs re-aggregated; pre-refactor backup at `outputs/before_C_form/`; migration report at `docs/insights/C-form-migration-report.md` |
| **M2-tests** | Unit tests on the new rate definitions | ✅ `tests/test_metrics.py` (15 tests) + `DriverRowSchemaRegressionTest` schema guard + `scripts/verify_m2_schema.py` CI guard (61/61 jsonls pass) |
| **§3-prose** | Problem-definition body (4-condition setup, JSON-strict prompt, M2 metrics, universal graded-tilt reading post-C-form) | ✅ `docs/insights/paper-section-3-problem-definition.md` + paper draft `docs/paper/sections/03_method.md` | drawn from project.md §0.3 + M2-evidence + slides 5-7 of paper_summary_slides.md. C-form refresh 2026-04-28: `(pa-pb)·(anchor-pb) > 0` numerator, "categorical-replace regime" framing retracted as driver-bug artefact |

### 6.2 §4 — Datasets and anchor inventory

| ID | Task | Status | Notes |
|---|---|---|---|
| **D1 — VQAv2** | snapshot + loader | ✅ `inputs/vqav2_number_val/` |
| **D2 — TallyQA** | snapshot + loader | ✅ `inputs/tallyqa_test/` |
| **D3 — ChartQA** | snapshot + loader | ✅ `inputs/chartqa_test/`, integer-GT [1,1000] gate |
| **D4 — MathVista** | snapshot + loader, integer-GT gate | ✅ inputs ready; MathVista runs are γ |
| **I1 — anchor inventory** | FLUX-rendered digit images, range up to 10000 | ✅ `inputs/irrelevant_number/` |
| **I2 — anchor mask inventory** | OpenCV Telea inpaint of digit bbox | ✅ `inputs/irrelevant_number_masked/` |
| **I3 — neutral inventory** | digit-free FLUX renders, scene-balanced | ✅ `inputs/irrelevant_neutral/` |

### 6.3 §5 — Distance, plausibility window, digit-pixel causality

The §5 narrative arc (`project.md §0.2 row 5`) is one story across E5b
(distance decay) → E5c (digit-pixel causality, masked vs. anchor vs.
neutral) → E5d (per-dataset cutoff validation) → E5e (S1-only cross-model
robustness).

| ID | Experiment | Status |
|---|---|---|
| **E5b distance sweep** | 5-stratum × b + a (VQAv2 + TallyQA), llava-interleave-7b | ✅ single model — cross-model in flight |
| **E5c digit-mask control** | + 5-stratum × m, llava-interleave-7b | ✅ single model — cross-model in flight |
| **E5d per-dataset cutoff** | ChartQA: S1-only relative `\|a-gt\| ≤ max(1, 0.1·gt)`; TallyQA: absolute `[0,5]`; VQAv2: range `{0..9}`; MathVista: C3 FAIL, scope-out as plausibility-window contrast (or rerun stricter) | ✅ except MathVista — see γ |
| **E5e S1-only cross-model** | b/a/m/d × ChartQA + TallyQA × 3 models | ✅ |
| **E5b/c cross-model expansion** | extend E5b + E5c to 3-model E5e panel (qwen2.5-vl-7b, gemma3-27b-it ∪ llava-interleave-7b) on VQAv2 + TallyQA | ⏳ in flight (user 2026-04-28) |
| **E5e MathVista (γ-α)** | MathVista b/a/m/d (S1) × 3 models | ✅ landed 2026-04-29; **C-form refresh 2026-04-28**: gemma3-27b wrong-base S1 `adopt(a) = 0.230`, `df(a) = 0.332` (panel-largest cell). All 3 models in graded-tilt regime under C-form (df > 0 universally); pre-refactor "categorical-replace df=0" reading was a driver-bug artefact, retracted in `E5e-mathvista-evidence.md` §5 |
| **E5e MathVista (γ-β)** | reasoning-mode VLM × MathVista — Qwen3-VL-8B-Instruct vs. Qwen3-VL-8B-Thinking (separate weights), 4-cond S1, max_new_tokens=512, runner is `</think>`-aware | ✅ landed 2026-04-28. Headline (C-form, S1 anchor arm, all-base, n=365): instruct adopt(a)=0.074 / df(a)=0.102, thinking adopt(a)=0.117 / df(a)=0.291. Thinking amplifies anchor pull (×1.6 adopt, ×2.9 df) — direction-agnostic hypothesis (H4) lands on the *amplification* side, consistent with VLMBias / Wang LRM-judging |
| **VQAv2 4-condition** | b/a/m/d cross-model VQAv2 | ☐ P1 (kept) |

### 6.4 §6 — Confidence-modulated anchoring (logit-based)

Generalises Phase A's wrong/correct binary into a continuous confidence
scale. Hypothesis (H7): `direction_follow_rate` is monotonic with
`pred_b`-token logit / probability. The wrong/correct gap from A1 is then
the coarsest possible projection of this monotonicity.

| ID | Task | Status |
|---|---|---|
| **L1** | per-token logit / softmax-prob already captured (commit `5f925b2`) on E5b/E5c/E5e + 7 main runs | ✅ data |
| **L2** | confidence-proxy menu — `top1_softmax_prob`, `top1_minus_top2_margin`, `entropy_top_k` — `scripts/analyze_confidence_anchoring.py` | ✅ landed 2026-04-29 |
| **L3** | per-confidence-quartile `adopt_rate` and `direction_follow_rate` table, model × dataset; compare to A1 binary split | ✅ 112,008 (sample × arm) records over 34 cells; `_data/L1_*.csv` |
| **L4** | report — pick the proxy + quartile shape with cleanest monotone trend; lift over A1 | ✅ `docs/insights/L1-confidence-modulation-evidence.md` — `entropy_top_k` wins; Q4 − Q1 mean df = +0.152 (C-form refreshed), 23/35 anchor cells fully monotone |
| **L5** | re-cast §6 narrative — "wrong/correct gap is a coarse projection of confidence monotonicity" | ✅ paper draft `docs/paper/sections/06_confidence.md` |
| **L6** | VQAv2 main panel logit re-run (no logit capture pre-commit `5f925b2`) | ☐ P1 — opportunistic |

### 6.5 §7 — Attention mechanism + mitigation

| ID | Experiment | Status |
|---|---|---|
| **E1 / E1b / E1c / E1d** | full anchor-image attention pipeline (mass + per-layer + H3-falsified writeup + causal ablation) | ✅ |
| **E1-patch (POC)** | digit-pixel-patch attention reanalysis on 2 representative archetypes (llava-1.5-7b mid-stack, gemma4-e4b SigLIP-early). bbox JSON via `scripts/compute_anchor_digit_bboxes.py`; extraction via `scripts/extract_attention_mass.py --bbox-file`; analysis via `scripts/analyze_attention_patch.py`. **POC headline (2026-04-29)**: gemma4-e4b digit/anchor = 0.631 (peak L9, +0.404 above fair share); llava-1.5-7b digit/anchor = 0.468 (peak L7, +0.241 above fair share). Two profiles: gemma globally digit-concentrated, llava peaked mid-early. `docs/insights/E1-patch-evidence.md`. | ✅ POC landed 2026-04-29; full 6-model panel + masked-arm causal control deferred |
| **E4 Phase 1 + 2** | mid-stack-cluster attention re-weighting (LLaVA-1.5 / ConvLLaVA / InternVL3) | ✅ |
| **E4 §7.4 paper rendering** | report `direction_follow_rate` reduction, `exact_match` rise, `accuracy_vqa(b)` invariance side by side; the "free lunch" framing | ✅ `docs/insights/paper-section-7-4-mitigation-free-lunch.md` (2026-04-29) |
| **E1-patch generalises mitigation?** | does upper-half attention mass concentrate on the digit patch only? if yes, mitigation can shrink target region | ☐ P3 |

### 6.6 §8 — Future work (scope only)

| ID | Direction | Status |
|---|---|---|
| **F1 (preferred)** | LLM/VLM architectural diff — same anchor delivered as text to LLM vs. as image to VLM, compare layer-wise integration profile (§7-style attention) | ✅ ideation paragraph drafted — `docs/insights/paper-section-8-f1-future-work.md` (2026-04-29) |
| **F2** | image-vs-text anchor — anchor image described as text and given to the same VLM; effect-size delta | ☐ ideation only |
| **F3** | Reasoning-mode VLM at scale — Qwen3-VL thinking, etc., on E5e cross-dataset matrix | ☐ scope only (γ-β is the minimal §8 stake) |

## 7. Pending work — priority queue

P0 = blocks paper sections, do this week. P1 = strengthens but not load-
bearing. P2 = ideation depth. P3 = future / parallel.

**As of 2026-04-28** — most P0s have landed (M2-refactor + C-form, L1-L4
confidence, γ-α + γ-β MathVista, E1-patch POC, paper §3/§7.4/§8 prose).
The remaining paper-blockers are the cross-model E5e/E5b/E5c gemma3-27b
TallyQA cell (in flight on GPU 1 since 2026-04-28 00:39) and the
qwen2.5-vl-7b expansion of E5b/E5c.

| P | Task | Source | ETA / compute |
|---|---|---|---|
| **P0** | E5e TallyQA gemma3-27b cross-model cell (in flight on GPU 1) | §6.3 E5b/c cross-model expansion | ~30-35h total wall (15.2h elapsed at 15:53 2026-04-28; competing with `physical_mode_activation` on GPU 1, no streaming write — disk dir empty until completion) |
| **P0** | qwen2.5-vl-7b on E5c VQAv2 + TallyQA (stratified, b + a×S1-5 + m×S1-5 + d) | §6.3 E5b/c cross-model expansion | ~3h × 2 datasets on H200 — queue once GPU 1 frees |
| **P0** | gemma3-27b on E5c VQAv2 (TallyQA stratified is infeasible at full n=1000 base; use `max_samples=300` if launched) | §6.3 E5b/c cross-model expansion | ~5-6h on H200 |
| **P1** | E1-patch full panel — masked arm causal control + 4 remaining archetypes (qwen2.5-vl-7b, internvl3-8b, convllava-7b, fastvlm-7b) | §6.5 E1-patch | ~1.5h attention extraction (n=200 each) + analysis |
| **P1** | VQAv2 4-condition cross-model (b/a/m/d, S1 only, kept) | §6.3 | ~1d (3 models) — opportunistic |
| **P1** | Citation verification — every 2026 arXiv ID in `references/project.md` and §2 paper draft must resolve to a real paper | §9 caveat | hours of manual verification, reviewer-defuse |
| **P3** | E4 generalisation to other archetypes (Gemma / Qwen / FastVLM) | §6.5 | days |
| **P3** | Image-vs-text anchor (F2) follow-up paper | §6.6 | future |
| **P3** | VQAv2 main panel logit re-run (L6 — no logit capture pre-commit `5f925b2`) | §6.4 | opportunistic |

**Recently landed (struck from queue 2026-04-28):**

- ~~M2 `metrics.py` refactor + re-aggregation~~ ✅ (M2 + C-form follow-up)
- ~~Per-token logit confidence analysis (L1–L4)~~ ✅
- ~~E5e MathVista (γ-α + γ-β)~~ ✅
- ~~E1-patch POC (2 archetypes)~~ ✅
- ~~M2 unit tests~~ ✅ (`tests/test_metrics.py`)
- ~~§8 F1 ideation paragraph~~ ✅
- ~~Paper §3 / §7.4 / §8 prose~~ ✅ (`docs/insights/paper-section-*.md` + `docs/paper/sections/*.md`)

## 8. Pending refactors

| ID | Task | Why | Status |
|---|---|---|---|
| **M1** | Paired adoption metric `(base_pred ≠ anchor) AND (pred == anchor)` | base-prediction confound (`pre-M1` marginal definition counted `gt == anchor == pred` as adoption; full M1 commentary in 2026-04-27 changelog) | ✅ landed `bbcc418..ce1928a`; 54 dirs re-aggregated 2026-04-27 |
| **M2** | Metric definitions — `adopt_rate = A_paired/D_paired`, `direction_follow_rate = DF_moved/DD_all`, naming canonical (`pred_b/pred_a/pred_m/pred_d`) | M1 fixed numerator confound but left denominator and df numerator under-specified; `adopt_cond` was unofficial in E5b/E5c; df numerator's `pa != pb` clause closes the no-change-in-direction confound | ✅ landed 2026-04-29 (commits TBD); 53 run dirs re-aggregated; legacy fields kept for audit; tests in `tests/test_metrics.py` |

## 9. Caveats — carry these into every analysis

- **Strengthen-prompt anomaly.** Under `experiment_anchor_strengthen_prompt`,
  three models show pathological `mean_distance_to_anchor` (gemma3-27b
  2617, qwen2.5-vl-7b 1519, gemma4-31b 511 — see 2026-04-23 changelog).
  The "must output a number" instruction induces large-number
  hallucination, *not* anchor adoption. Filter or report with a robust
  statistic before quantitative claims.
- **`MathVista` C3 FAIL.** E5d MathVista validation rejected the S1-only
  cutoff at the same `n` that ChartQA accepted it; the diffuse pattern is
  robust to GT-floor filtering. (γ-α) re-runs at 4-condition single
  stratum with the 3-model E5e panel; (γ-β) tests reasoning-mode separately.
- **Anchor digit ∈ 0-9 vs. answer support 0-8.** When computing "moved
  toward anchor", control for the fact that anchor 9 cannot be the
  correct answer in this subset. M2 `D_paired` and `D_clean` denominators
  partially handle this; the new `gt != anchor` flag is captured but
  excluded from the headline denominator on signal-cleanness grounds (§5
  of M2 doc).
- **Broken VQA image** (`inputs/vqav2_number_val/images/000000000136.jpg`)
  — file body is filesystem garbage. Loader calls `Image.verify()` and
  silently skips; the questions.jsonl entry is dropped on load (one fewer
  sample).
- **`fastvlm-7b` prose outputs.** Often emits prose despite the JSON-only
  prompt; `extract_first_number` rescues most but parse-failure rate is
  non-zero. Report explicitly.
- **`InternVL3-8b` prose-leak parse loss.** ~30 % of records drop out of
  E4 Phase-1 valid-triplet count because InternVL3 emits prose ("based
  on…") truncated at `max_new_tokens=8`. Driver patch to `max_new_tokens`
  or InternVL3-specific JSON-strict prompt is the fix; tracked in §6.5.
- **Shared GPU.** Same machine runs a vLLM Qwen2.5-32B on port 8000
  (~55 % VRAM). Effective per-GPU budget ~60 GB.
- **Citation hygiene.** `references/project.md` flags some 2026 arXiv IDs
  that may not resolve. Verify every cite before submission.
- **Bilingual docs convention retired** (commit `84f9341`, 2026-04-27).
  No `_ko.md` mirrors. English `.md` is the only canonical version.
- **In-flight inference.** E5b / E5c cross-model expansion and E5e
  MathVista (γ) are running. Tables in §3.3 will tighten when those land;
  M2 evidence numbers refresh accordingly.

## 10. Changelog

- **2026-04-28 (overnight polish, second pass)** — **§5.2 / §5.4 / §6 / §7 / §9 / §4 cross-checks; γ-β number propagation; A1 CSV smoke-run pollution fix.**
  Continuation of the doc-only polish session while gemma3-27b TallyQA
  E5e is still on GPU 1. (i) **Phase A `_data/A1_*.csv` regenerated**
  — `scripts/phase_a_data_mining.py::_resolve_model_runs` was picking
  the alphabetically-latest run dir per model, which silently selected
  a 45-record smoke run from `outputs/experiment/qwen2.5-vl-7b-instruct/20260428-140004/`
  over the canonical 53,190-record full run from `20260411-213927/`,
  pumping qwen2.5-vl out of all A1-A7 aggregates. The function now
  picks the *largest* run with `n ≥ 100` records; A1-A7 CSVs
  regenerated to include all 7 models. The user-facing A1 evidence
  doc `docs/insights/A1-asymmetric-on-wrong.md` already had the correct
  numbers (its `2026-04-28 verification` note is hand-cited from the
  prediction.jsonl raw computation, not the CSV). (ii) **Paper §5.2
  numbers verified** — wrong-correct moved-closer gap from
  `_data/A1_asymmetric_wide.csv`: gemma4-e4b +19.6, gemma3-27b +15.9,
  qwen3-vl-30b +12.2, gemma4-31b +8.4, qwen3-vl-8b +8.0,
  llava-interleave +7.2, qwen2.5-vl-7b +6.9 — all 7 match paper
  §5.2 within ±0.1 pp. The paper's "+6.9 to +19.6 pp on 7/7 models"
  range stands. Note: §5.2 metric is the **pull-form** moved-closer
  rate (`|pa−anchor| < |pb−anchor|`, distance-based), distinct from
  §3.3 / §5.1's **C-form** `direction_follow_rate` (sign-based).
  Both detect the same phenomenon; magnitudes differ by 0.5-2.6 pp
  on this panel (C-form gives +7.4 to +22.2 pp). The paper uses
  pull-form in §5.2 because it is Phase A's original definition.
  (iii) **Paper §5.3 distance-decay numbers verified** against
  `_data/E5b_per_stratum.csv` — VQAv2 wrong-base 0.130 / 0.032 /
  0.010 / 0.010 / 0.003 and TallyQA 0.092 / 0.006 / 0.003 / 0.000 /
  0.000 are exactly correct. (iv) **Paper §5.4 digit-pixel-causality
  numbers verified** against `_data/E5c_per_cell.csv` — VQAv2
  wrong-base S1 anchor 0.129 / masked 0.068 / gap +6.1 pp; TallyQA
  0.110 / 0.084 / +2.6 pp (paper rounds to +2.5). VQAv2 distance-decay
  gap S1→S5 = 0.061 / 0.016 / 0.013 / 0.012 / 0.008 — all match.
  (v) **§5.6 cross-dataset gap verified** — MathVista gemma3-27b
  wrong-base S1 a-arm − m-arm gap = 0.230 − 0.051 = +17.97 pp,
  rounded to "+17.9 pp" in §5.6. (vi) **Paper §6 confidence numbers
  verified** — Q4-Q1 entropy_top_k mean = +0.152, 23/35 fully
  monotone, worked example (E5c VQAv2 llava S1) Q1 0.043 → Q4 0.172
  on adopt and Q1 0.032 → Q4 0.210 on direction-follow all match
  `_data/L1_confidence_quartile_long.csv`. (vii) **Paper §7
  mitigation numbers verified** — LLaVA-1.5 −17.7 % rel df,
  ConvLLaVA −10.6 %, InternVL3 −5.8 %; em rises +0.49 to +1.30 pp;
  recovery 21.7 % / 14.0 % / 10.2 % — all match
  `docs/insights/E4-mitigation-evidence.md`. (viii) **γ-β instruct
  numbers in roadmap fixed** — three places (`§3.1` row, `§6.3`
  row, `§10` 2026-04-28 changelog entry) cited
  `instruct adopt(a) = 0.055`, `df(a) C-form = 0.094` which were
  pre-C-form / wrong-source numbers. Real numbers from
  `outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-instruct/20260428-114421/summary.json`
  are `adopt(a) = 0.074`, `df(a) C-form = 0.102`; ratios become
  ×1.6 adopt and ×2.9 df (was claimed ×2.1 / ×3.1). Paper §1
  abstract and §6.5 already used the correct ×1.6 / ×2.9 ratios; the
  roadmap was the only stale surface. (ix) **§9 caveat fixed** —
  "three Gemma models" mislabel (the trio is gemma3-27b 2617 +
  gemma4-31b 511 + qwen2.5-vl-7b 1519; only 2/3 are Gemma).
  (x) **§4 sample-size table fixed** — line 86 labeled the older
  3-cond `experiment_chartqa` run (5,390 × 3 cond, 48,510) as
  "E5e ChartQA full" which conflicts with the 4-cond
  `experiment_e5e_chartqa_full` (705 × 4 cond, 8,460) actually used
  by §5.5. Split into two rows. TallyQA E5e count clarified
  ("2 done + 1 in flight"). (xi) **`scripts/run_e5b_e5c_cross_model_chain.sh`
  pre-staged** — sequential launcher for qwen2.5-vl-7b and
  gemma3-27b on E5c VQAv2 + TallyQA on GPU 1, gated on the in-flight
  TallyQA gemma3-27b job releasing the GPU. Ready to launch after
  pre-flight `nvidia-smi --query-compute-apps` check. (xii) Roadmap
  §3.1 `experiment_e5e_tallyqa_full` row updated to reflect that
  qwen2.5-vl-7b cell is finished (was marked queued).

- **2026-04-28 (overnight polish)** — **Roadmap §6/§7 stale-status sweep + paper §5.5 number-correction + §2 citation audit.**
  Five categories of drift caught in a non-GPU polish pass while GPU 1
  is occupied by the gemma3-27b TallyQA E5e run. (i) **§7 priority queue
  refresh** — struck five P0s that had landed but never moved off the
  queue (M2-refactor, L1–L4, γ-α, γ-β, M2-tests, paper §3/§7.4/§8 prose);
  rebuilt §7 around the actually-blocking items: TallyQA gemma3-27b
  in flight, qwen2.5-vl on E5c VQAv2/TallyQA, gemma3-27b on E5c VQAv2
  (subsampled); citation verification surfaced as P1 reviewer-defuse.
  (ii) **§6.1 / §6.4 status flips** — M2-refactor / M2-tests / L2-L4 /
  L5 all flipped to ✅ with cross-pointers to the landed evidence.
  (iii) **Paper §5.5 numeric fix** — eight cells of `df(a)` / `df(m)`
  in the cross-dataset E5e table were stale (drawn from a pre-C-form
  source); cross-checked all 12 numbers against
  `outputs/experiment_e5e_*_full/*/summary.json` and per-cell CSVs and
  rewrote the table with authoritative values. ChartQA × 3 models and
  TallyQA × llava-interleave-7b were the off cells; MathVista row was
  already correct. Also added the newly-completed TallyQA × qwen2.5-vl
  cell (`adopt(a) = 0.011`, `df(a) = 0.029`) — that run had been
  marked "queued, predictions.jsonl pending" in §3.3 but is in fact
  finished as of `outputs/experiment_e5e_tallyqa_full/qwen2.5-vl-7b-instruct/20260427-235812/`.
  Same correction applied to roadmap §3.3 E5e table. (iv) **§3 prose
  C-form alignment** — `docs/insights/paper-section-3-problem-definition.md`
  (committed before the C-form refactor) carried the old
  `(pb − gt)·(pa − gt)` numerator and the buggy
  "categorical-replace regime → df = 0 on MathVista" reading. Both
  superseded: §3.3 now states the C-form `(pa − pb)·(anchor − pb)` numerator
  with rationale, and the regime distinction is replaced by a graded-
  tilt-is-universal reading with a historical retraction note. The
  paper draft `docs/paper/sections/03_method.md` was already C-form
  correct; this fixes the older insight doc + propagates to
  `docs/insights/E5e-mathvista-evidence.md` open-follow-ups (§9 was
  duplicated as §8 — fixed) and `docs/insights/paper-section-7-4-mitigation-free-lunch.md`
  (single sentence about "graded-tilt vs categorical-replace regime"
  reframed to "graded-tilt magnitude varies by dataset"). (v) **§2
  citation audit** — verified six arXiv IDs: `2603.19203` Tinted Frames
  (real, Fan et al. 2026), `2505.23941` VLMBias (real, Vo+Nguyen et al.),
  `2505.15392` Huang et al. anchoring (real, A-Index/R-Error confirmed),
  `2507.03123` AIpsych (**author wrong** — was "Jin et al.", actually
  Liu et al.), `2508.20570` (**description wrong** — was "early-layer
  CLIP/SigLIP", actually Hufe et al. "Dyslexify" later-half CLIP-only
  attention-head circuit), `2502.08193` Wang-Zhao-Larson NAACL 2025
  multi-image typographic attack (newly added — §2.3 had cited
  Wang-Zhao-Larson with the *wrong* arXiv 2508.20570 ID, conflating
  it with Dyslexify). §1 intro had a parallel "Jin et al. 2025"
  attribution for the LLM anchoring lineage — fixed to "Huang et al.
  2025" to match the §2 reference. §2.4 description of Dyslexify
  refactored to match its actual content (later-half CLIP attention-head
  circuit + ablation mitigation), and a one-sentence connection to
  our §7 upper-half LLM-stack finding added. New artefact:
  `configs/experiment_e5c_vqa.yaml` and `configs/experiment_e5c_tally.yaml`
  pre-staged with `qwen2.5-vl-7b-instruct` and `gemma3-27b-it` rows
  added (ready to launch once GPU 1 is freed by the in-flight
  TallyQA gemma3-27b run, no `run_experiment.py` invocation made
  this session).

- **2026-04-28** — **γ-β MathVista (reasoning-mode) landed: thinking amplifies anchor pull.**
  Background `btymeywv9` finished overnight; results on
  `outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-thinking/20260428-114421/`,
  reaggregated post-C-form. Per-arm headline (S1 anchor, all-base, n=365):
  instruct (Qwen3-VL-8B-Instruct) `adopt(a) = 0.074`, `df(a) C-form = 0.102`;
  thinking (Qwen3-VL-8B-Thinking) `adopt(a) = 0.117`, `df(a) C-form = 0.291`
  — thinking is **×1.6 on adopt and ×2.9 on direction-follow**. Masked-arm
  digit-pixel causality is preserved on both (anchor > masked on every
  metric, both models). Thinking acc(b) = 0.196 (slightly below
  instruct's 0.216 — reasoning trace does not gain accuracy on this
  panel, but loses anchor robustness). H4 ("reasoning reduces anchoring")
  lands on the *amplification* side, consistent with VLMBias and Wang
  LRM-judging. The γ-β result is the strongest single
  reasoning-amplifies-bias finding in our panel and a paper-tier hook.
  Pre-reaggregate (driver-bug 0) results preserved at
  `outputs/before_C_form/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-thinking_postlanding/`.

- **2026-04-28** — **direction_follow_rate refactor — C-form (pa-pb)·(anchor-pb).**
  An audit triggered by γ-β `df_M2 = 0` exposed two compounding bugs:
  (i) the `direction_follow_rate` formula in `metrics.py:118` had carried
  the M1-era `(pa-gt)·(anchor-gt) > 0` form unchanged through the M2
  commit, even though that commit's docstring + commit message + 5 doc
  surfaces all declared `(pb-gt)·(pa-gt) > 0`. Both forms turned out to
  be unintended on reflection; the user's true intent (verified across
  the same commit and a brief design discussion) was the gt-free
  C-form `(pa-pb)·(anchor-pb) > 0  AND  pa != pb` — anchor pull as a
  baseline-relative shift toward the anchor stimulus, robust to per-
  question stimulus draws. (ii) `run_experiment.py` row dict never
  threaded `anchor_direction_followed_moved`, `pred_b_equal_anchor`, or
  `pred_diff_from_base` from `sample_eval`, so `summarize_condition`
  read missing keys as 0 and reported `df_M2 = 0` on every directly-
  driven run. `reaggregate_paired_adoption.py` had silently been fixing
  (ii) for any dir it touched, but eight dirs (γ-α MathVista, γ-β
  reasoning, one TallyQA E5e cell) had never been re-aggregated.
  Remediation: code surfaces (`metrics.py:118`, reaggregate, analyze
  variants, tests) all rewritten to C-form; 7 doc surfaces (project.md
  §0.3, roadmap.md §4, AGENTS.md, M2 evidence, paper_summary_slides,
  metrics.py docstrings, analyze_metric_variants docstrings) re-stated
  in C-form; driver row dict gained the three missing flags;
  `tests/test_metrics.py::DriverRowSchemaRegressionTest` enforces
  schema parity going forward; full reaggregate sweep across 17 sub-
  trees rewrote 61 `predictions.jsonl` files. Pre-refactor results
  archived at `outputs/before_C_form/`. Side-by-side before/after
  comparison: `docs/insights/C-form-migration-report.md` (+ 7-slide
  PPTX deck `docs/figures/C_form_migration_report.pptx`). **All
  paper-tier qualitative claims preserved or strengthened under
  C-form.** Largest single shift: qwen2.5-vl-7b VQAv2 df_moved
  0.079 → 0.094 (modest under standard prompt; up to ×2.7 on other
  cells across the project). E5e MathVista's "df_M2 = 0 universally
  → categorical-replace regime" framing was a driver-bug artefact:
  true df_moved is 0.099 on γ-α and 0.195 on γ-β. The §"E5e
  MathVista evidence" section needs a writeup rewrite; other evidence
  docs (A1, E1d, E4, E5b, E5c, ChartQA, TallyQA, L1) need only a
  number refresh — qualitative narratives survive intact. New
  artefacts: `scripts/build_C_form_migration_report.py`,
  `scripts/verify_m2_schema.py` (CI guard, 61/61 jsonls pass),
  `docs/deprecated/_ko-mirrors-2026-04-27/` (18 files moved per the
  retired bilingual convention), 145 `.marginal.bak.*` files deleted
  from the working tree (118 inside `outputs/before_C_form/`
  preserved as audit artefacts). Memory entry
  `feedback_metric_C_form.md` persisted to prevent re-litigation in
  future sessions. §3.3 headline table refreshed.

- **2026-04-28** — **E5e MathVista (γ-β) reasoning-mode launched.** Same
  4-condition S1 design as γ-α (b/a/m/d, integer-GT subset, relative_s1 cutoff)
  but with the thinking-on / thinking-off pair: `Qwen3-VL-8B-Instruct`
  (already in main panel) vs. `Qwen3-VL-8B-Thinking` (Apache-2.0, 9B BF16,
  separate weight — Qwen3-VL Thinking is shipped as a distinct checkpoint,
  not an `enable_thinking` flag). Smoke (`/tmp/qwen3vl_thinking_smoke.py`,
  one MathVista question) confirmed Thinking output format =
  `<trace>\n</think>\n\n{"result": <num>}<|im_end|>` with `</think>`
  as plain text (not a special token). Three plumbing changes landed
  before launch: (i) `vlm_anchor.utils.extract_last_number` added with 4
  unit tests (`tests/test_utils.py::ExtractLastNumberTest`); (ii)
  `models._summarize_generation` is now thinking-aware — splits
  `decoded` on the **last** `</think>` and parses `extract_first_number`
  on the post-trace tail (so trace-internal numerals never leak into
  the prediction), with answer-token matching switched to reverse
  iteration to lock onto the final answer token; (iii) `run_experiment.py`
  threads `thinking_marker_present` and `n_generated_tokens` into every
  `predictions.jsonl` row + adds `del runner; gc.collect();
  torch.cuda.empty_cache()` between models so back-to-back 8B BF16
  loads don't OOM. The `thinking_marker_present` flag is the
  truncation guard advisor flagged: when False, `max_new_tokens` cut
  off the trace before the final answer JSON, so the row's prediction
  may be a trace-internal numeral and must be filtered. New artefacts:
  `configs/experiment_e5e_mathvista_reasoning.yaml`, smoke test at
  `/tmp/qwen3vl_thinking_smoke.py`. Inflight: GPU 0, instruct ETA
  ~40min, thinking ETA ~15h (smoke timing 35s/gen × 1,540 gen — will
  refine once stable trace-length distribution lands). Tests: 25
  utils + metrics tests pass. Pre-existing `tests/test_data.py` PNG
  header failure unrelated.

- **2026-04-29 (late evening)** — **Paper-section prose for §3, §7.4, §8/F1 landed.**
  Three first-draft paper-ready writeups committed under
  `docs/insights/`: (i) `paper-section-3-problem-definition.md` covers
  the 4-condition setup (`b/a/m/d`), the three pairwise gaps that
  underwrite §5/§6/§7, the JSON-strict inference template, and the
  four canonical metrics (`adopt_rate`, `direction_follow_rate`,
  `exact_match`, `anchor_effect_M`) with the graded-tilt vs.
  categorical-replace diagnostic reading; (ii)
  `paper-section-7-4-mitigation-free-lunch.md` carries the upper-half
  attention re-weighting from E1d hard-mask to E4 soft-strength,
  reports the Phase 2 full-scale free-lunch table (df ↓ −5.8 to
  −17.7 % rel · em(a) ↑ +0.49 to +1.30 pp · em(b) invariant on 3/3
  mid-stack-cluster models), unpacks why "free lunch" means three
  things together, and quantifies the 10–22 % anchor-damage recovery
  ratio with three named caveats; (iii)
  `paper-section-8-f1-future-work.md` is the §8 entry-point paragraph
  for the LLM-vs-VLM architectural-diff direction (F1) — paired
  LLM↔VLM panel, text-anchor vs image-anchor four-condition prompt,
  same §3 metrics + §7 attention extraction transferred to LLM stacks,
  with three named outcome patterns (A: same locus / B: same
  magnitude different depth / C: asymmetric magnitude) and the
  one-figure-one-table deliverable. Roadmap §6.1 gets a `§3-prose`
  row, §6.5 `E4 §7.4 paper rendering` flips to ✅, §6.6 `F1` flips
  to ✅, §7 P2 row strikes through. Git hygiene pass alongside:
  6 deleted `inputs/` files removed from the index (`inputs/` already
  gitignored upstream); the paper-summary deck pipeline lands as
  `scripts/build_paper_figures.py` + `scripts/build_paper_pptx.js` +
  the four `paper_*.png` figures + the speaker-notes companion
  `docs/insights/paper_summary_slides.md`; heavy regenerable artefacts
  (`paper_summary.pdf`, `paper_summary.pptx`, `slide-*.jpg`) added to
  `.gitignore` and rebuilt on demand from the two scripts.
- **2026-04-29 (evening)** — **E1-patch POC landed (2 archetypes).**
  Driver: `scripts/compute_anchor_digit_bboxes.py` produces digit
  bboxes from anchor − masked PNG diff (n=128 anchors); modified
  `scripts/extract_attention_mass.py` accepts `--bbox-file` and emits
  `image_anchor_digit` + `image_anchor_background` regions (normalized
  bbox coords mapped to row-major patch grid; perfect-square spans only,
  multi-tile / multi-scale models skip transparently); analysis via
  `scripts/analyze_attention_patch.py`. POC on `llava-1.5-7b` and
  `gemma4-e4b` (n=400 each, ~25 min wall). **Both models pass the
  headline test**: `mass_digit / mass_anchor` substantially exceeds the
  bbox area share at the digit-attention peak layer. `gemma4-e4b`:
  L9 / 42, ratio = 0.631, concentration above fair share = **+0.404**
  (globally digit-concentrated profile, every layer 0.5-0.6). `llava-1.5-7b`:
  L7 / 32, ratio = 0.468, concentration above fair share = **+0.241**
  (peaked mid-early profile, drops below fair share by L30). Two
  qualitatively different attention pathways — Gemma SigLIP globally
  inherits typographic feature, LLaVA mid-stack-cluster peaks at L7-L11
  *earlier* than its E1b total-mass peak at L16. Deferred to
  full-panel: masked-arm causal control (current `configs/experiment.yaml`
  lacks the `masked` extra), per-encoder bbox mapping for ConvLLaVA /
  InternVL3 / FastVLM, step-0 vs answer-step comparison. New artefacts:
  `docs/insights/E1-patch-evidence.md`,
  `docs/insights/_data/E1_patch_per_layer.csv`,
  `docs/insights/_data/E1_patch_concentration_per_layer.csv`,
  `inputs/irrelevant_number_bboxes.json`.
- **2026-04-29 (evening)** — **L1 confidence-modulated anchoring (§6 of
  the paper) evidence written.** Driver:
  `scripts/analyze_confidence_anchoring.py` on 10 logit-capturing runs
  (post-commit `5f925b2`: E5b, E5c, E5d, E5e). 112,008 (sample × arm)
  records over 34 cells. Three confidence proxies tested:
  `softmax_top1_prob`, `top1_minus_top2_margin`, `entropy_top_k`.
  **Headline: `entropy_top_k` is the cleanest proxy.** Mean
  `direction_follow_rate` Q4 − Q1 = +0.128, with 18/34 anchor cells
  fully monotone Q1 < Q2 < Q3 < Q4. `adopt_rate` Q4 − Q1 = +0.044,
  10/34 monotone. Worked example (E5c VQAv2 S1, llava-interleave-7b):
  Q1 (most confident, mean exact_match_b = 0.77) → adopt 0.077 / df 0.040;
  Q4 (most uncertain, exact_match_b = 0.07) → adopt 0.147 / df 0.113;
  Δ = +7.0 / +7.4 pp. The wrong/correct binary in Phase A A1 is a coarse
  projection of this continuous monotonicity. New artefacts:
  `docs/insights/L1-confidence-modulation-evidence.md` (with §0 Intuition
  added), `docs/insights/_data/L1_*.csv`. VQAv2 main panel logit re-run
  is queued under §6.4 (no logit capture pre-commit `5f925b2`).
- **2026-04-29 (evening)** — **E5e MathVista (γ-α) cross-model 4-condition
  full run landed.** Driver `scripts/run_experiment.py
  --config configs/experiment_e5e_mathvista_full.yaml`, 3 models × 385
  base questions × 4 conditions = 4,620 records, total wall ~45 min on
  H200. Headlines (M2, wrong-base S1 anchor arm): **gemma3-27b-it
  adopt(a) = 0.194 — the largest single cell in the program** (vs. all
  prior datasets ≤ 0.13); anchor − masked gap = +15.2 pp. llava +4.1 pp,
  qwen +0.7 pp; all 3/3 preserve `a > m`. **`direction_follow_rate (M2)
  = 0` on every model on every condition** — MathVista is the
  "categorical-replace" regime where anchor either replaces base
  outright or doesn't move it at all. Contrasts with VQAv2 / TallyQA /
  ChartQA "graded-tilt" regime where movement-without-adoption (df_raw
  ≫ adopt) dominates. New evidence: `docs/insights/E5e-mathvista-evidence.md`.
  E5d MathVista C3 FAIL diagnosis (diffuse pattern across distance) was
  a llava-specific small-n behaviour; γ-α at full integer subset reveals
  a clean cross-model signal. `analyze_e5e_wrong_correct.py`
  (new) writes `_data/experiment_e5e_mathvista_full_per_cell.csv`.
- **2026-04-29** — **M2 landed.** `metrics.py` refactored to compute headline
  rates with M2 denominators: `anchor_adoption_rate` now uses
  `D_paired = (pb != anchor)` denominator (vs. previous `D_all`); new field
  `anchor_direction_follow_rate` requires `pa != pb` in its numerator (vs.
  previous `DF_raw` which counted no-change pairs). Legacy
  `anchor_adoption_rate_marginal` and `anchor_direction_follow_rate_raw`
  fields kept for audit. Per-row flags added: `pred_b_equal_anchor`,
  `pred_diff_from_base`, `anchor_direction_followed_moved`. Re-aggregation
  via `scripts/reaggregate_paired_adoption.py --apply --force` rewrote 53
  run dirs (predictions.jsonl + summary.json); pre-M1 marginal backups
  preserved untouched. New unit tests added (`tests/test_metrics.py` +
  `EvaluateSampleM2FlagsTest`, `SummarizeConditionM2DenominatorTest`):
  15 metrics tests pass. §3.3 headline tables refreshed with M2 numbers
  and now reference E5b/E5c wrong-base + E5e cross-dataset all-base. Stale
  smoke-only run dir `outputs/experiment/qwen2.5-vl-7b-instruct/20260427-075523/`
  (n=25) deleted. M2 status in §8 → ✅. `docs/insights/M2-metric-definition-evidence.md`
  remains as the definition rationale; refresh of its §5 numbers using
  post-aggregation full set is queued.
- **2026-04-28** — **Roadmap restructured to be paper-section-anchored,
  in lockstep with the new `references/project.md §0` paper outline.**
  §1 (research definition) updated to 4-condition canonical (b/a/m/d) with
  pred_b/pred_a/pred_m/pred_d/anchor/gt naming. §2 hypothesis table adds
  H7 (logit-based confidence monotonicity, §6 of paper). §3 status snapshot
  consolidates the prior "completed runs / pilot / smoke / models integrated"
  tables into one per-experiment matrix with status flags. §4 documents
  the M2 canonical metrics (`adopt_rate = A_paired/D_paired`,
  `direction_follow_rate = DF_moved/DD_all`, full equivalences in M2 doc).
  §5 closes Phase A status. §6 maps every running / pending / planned
  experiment to its paper section (§3-§8). §7 is one P0/P1/P2/P3 priority
  queue. §8 has M2 added to pending refactors. §9 caveats updated for
  strengthen anomaly, MathVista C3 FAIL, parse-loss, retired bilingual
  convention. Changelog (this §10) is **append-only** and preserves all
  prior entries unchanged. New artefacts: `references/project.md` (rewritten
  with §0 paper outline above the preserved 2026-04-23 feasibility review),
  `docs/insights/M2-metric-definition-evidence.md`, `scripts/analyze_metric_variants.py`,
  `docs/insights/_data/M2_*.csv`.
- **2026-04-27** — **E5c results: digit-pixel causality confirmed.** Stratified E5c run on llava-interleave-7b (n=1000 base questions per dataset, VQAv2 + TallyQA, 12 conditions per question = 1 baseline + 5 anchor strata + 5 masked-anchor strata + 1 neutral). Adopt_cond gap (anchor − masked) on wrong-base S1: +6.1 pp (VQAv2), +2.5 pp (TallyQA). Decays with distance to ~0 by S5. acc_drop comparison on correct-base: masked ≈ neutral (~7-9 pp VQAv2, ~2-5 pp TallyQA), confirming the anchor image background acts like a generic 2-image distractor. df_cond on wrong-base is nearly identical between anchor and masked (anchor S2 0.5198 vs masked S2 0.5223 on VQAv2) — direction-follow is dominated by uncertainty-driven directional drift, not digit pixels. Conclusion: the digit pixel is the operative cause of paired adoption; the anchor image's background offers no information beyond generic distraction. Writeups: `docs/experiments/E5c-anchor-mask-control.md`, `docs/insights/E5c-anchor-mask-evidence.md`.
- **2026-04-27** — **E5c queued: anchor-mask control experiment.** Replaces the digit pixel region in each anchor image with a content-preserving mask (digit bbox only) to isolate the digit-specific anchoring contribution from generic second-image distraction. Runs the E5b stratified pipeline on the masked variants. Confirms whether E5b's wrong-base × S2 peak is causally driven by the digit token or by 2-image presence alone. See §6 Tier 2 row for design.
- **2026-04-27** — **E5b results landed: anchoring is uncertainty-modulated AND plausibility-windowed.** Stratified E5b run on llava-interleave-7b (n=1000 base questions per dataset, VQAv2 + TallyQA, 5 anchor strata × 1 anchor each = 6 conditions per question) reveals two compounding gates on the cross-modal anchoring effect when measured under M1 paired conditional adoption (case 4 `base==a==pred` excluded from denominator):
    - **Uncertainty gate (A1 confirmed at scale).** Records where `target_only` was correct show essentially no anchor effect (`adopt_cond` ≤ 0.10 across all 5 strata, VQAv2 and TallyQA both). Records where `target_only` was wrong show adoption magnitudes 1.4–37× larger.
    - **Plausibility window.** On the wrong-base subset, adoption peaks at S1 [0,1] and decays sharply with distance: VQAv2 0.130 → 0.032 → 0.010 → 0.010 → 0.003 (S1→S5); TallyQA 0.092 → 0.006 → 0.003 → 0 → 0 (S5 = exactly zero adoption out of 346 wrong-base records, twice — i.e., implausible anchors are *fully* rejected).
    - **Cross-dataset robustness.** Same pattern in both datasets despite different baseline accuracies (acc(target_only) 0.62 VQAv2 vs 0.21 TallyQA), confirming the effect is not tied to a specific image domain.
  Direction-follow was attempted as the headline first but was noisier (S2 false-peak driven by case-4 records and `anchor==gt` boundary). Switching to `adopt_cond` made the pattern clean. Artefacts: `scripts/analyze_e5b_distance.py` (rewritten), `scripts/build_e5b_notebook.py`, `notebooks/E5b_anchor_distance.ipynb` (executed in place), `docs/figures/E5b_adopt_cond_curve.png` (per-dataset, base-correctness split), `docs/figures/E5b_adopt_cond_overlay.png` (cross-dataset wrong-base only), `docs/insights/_data/E5b_per_stratum.csv` (20-row table). Commits in the E5b branch: `f4ea410` (initial T10 with df-headline), `00afb81` (refactor to adopt_cond + base split).
- **2026-04-27** — **M1 landed: paired anchor-adoption metric.** `evaluate_sample` now requires `base_prediction` and computes `anchor_adopted = (base_pred ≠ anchor) AND (pred == anchor)` (commit `bbcc418`). Driver threads target_only's parsed prediction into subsequent conditions (`9c07f2e`). One-off `scripts/reaggregate_paired_adoption.py` (`220dc4b`, extended `ce1928a` for ablation/e4 schemas) re-computed adoption on all 54 existing predictions.jsonl files (35 standard + 13 causal_ablation + 6 e4_mitigation) — no re-inference needed; raw predictions preserved. Headline §3.4 paired-adoption rates: 0.019–0.059 (vs. previous marginal 0.110–0.141; ~75–90 % relative reduction). Direction-follow and accuracy values unchanged. Stale smoke-only output dirs deleted (`experiment_tallyqa`, `experiment_mathvista`, `experiment_smoke_check`, ChartQA 5-sample smoke). M1 status in §6 Pending refactors → ✅. Tier 2 hardening (E5/E5b/E7) can now proceed without further metric drift.
- **2026-04-27** — **M1 added to §6 (Pending refactors): paired adoption metric.** `anchor_adopted = (pred == anchor_value)` (`src/vlm_anchor/metrics.py:40`) currently ignores the base-condition prediction, so `GT == anchor == pred` counts as adoption — silently inflates rates wherever the anchor inventory overlaps the GT support (E5b stratum 1 `(0,1)` literally permits `\|a − gt\| = 0`; main-run anchor inventory 0–9 vs GT support 0–8 has 9-in-10 overlap). Refactor to paired `(base_pred ≠ anchor_value) AND (pred == anchor_value)`: `evaluate_sample` gains `base_prediction` arg; runner runs `target_only` first per sample-instance and threads its prediction into number/neutral evaluations; `summarize_condition` mean shape preserved (binary label → reads as a rate). Re-aggregation runs from raw `predictions.{jsonl,csv}` per model (no re-inference); raw `prediction` column preserved so any downstream re-derivation is still possible. Marginal definition retired in favour of paired. M1 must land before §6 Tier 2 hardening (E5/E7) re-touches the public numbers in §3.4 / Phase A / E1 / E4. **Cross-session note:** any concurrent edit to `metrics.py` or `models.py` runner ordering should align with this refactor — see §9 caveat.
- **2026-04-27** — **E5b design + plan committed; pipeline implemented and smoke-validated.** Anchor-distance robustness sweep added as new sub-experiment of E5. Stratified anchor sampling (5 strata by `|a − GT|`: [0,1] / [2,5] / [6,30] / [31,300] / [301,∞)), 500 base questions per dataset on TallyQA + VQAv2, llava-interleave-7b only. New driver path keyed off `inputs.anchor_sampling: stratified` in YAML; legacy 3-condition path untouched (regression-tested via 5-sample smoke on `configs/experiment.yaml`). Three smoke runs (VQAv2 stratified, TallyQA stratified, legacy) all pass: per-stratum mean-distance scales monotonically S1<S2<S3<S4<<S5, condition counters exact, `anchor_stratum_id` field present and None on legacy rows. Specs: `docs/experiments/E5b-anchor-distance-design.md` (+ _ko mirror), plan: `docs/experiments/E5b-anchor-distance-plan.md` (+ _ko mirror). Full run (T9) and reproducible notebook (T10) pending.
- **2026-04-25** — **Phase 1 anchor-damage table corrected via paired analysis; "InternVL3 has no anchor damage" reading retracted.** Investigation of the anti-correlation surprise + parse-loss caveat (advisor-flagged) revealed two issues. **(i) Driver-side, not parser-side, parse loss.** InternVL3's prose ("Based on the image…") truncates at `max_new_tokens=8` *before any digit*; the parser already uses the project's `extract_first_number`, which falls through to its first-token fallback ("based"). Analysis-layer rescue cannot recover what was never generated. The fix is a Phase-2 driver patch (longer `max_new_tokens` 16–32, or InternVL3-specific JSON-strict prompt). Logged in §6 E4 open follow-ups. **(ii) Anchor-damage table was unpaired, fixed by intersection-of-valid-cells analysis.** The original table compared em(target_only) on n=200 against em(num@0) on n=137 for InternVL3 — different sample subsets. Paired version (sample_instance_ids valid for *every* {target_only@0, num@0, num@s*, num@saturation} cell): n_paired = 109; em(TO) = 0.734 (vs unpaired 0.567); em(num@0) = 0.633 → anchor damage **−10.1 pp** (not the unpaired +2.3 pp "no damage" cell). All three models now show coherent damage / partial-recovery pattern: damage −7 to −12.5 pp, recovery 24–43 % at saturation. Caveat: InternVL3 paired set is the model's parse-tractable subset (em(TO) is 16.7 pp higher than condition-internal); treat the InternVL3 row as "behaviour on parse-tractable items", not "InternVL3 in general". **Refined headline:** the *df-axis anti-correlation* and the *em-axis coherence* are two views of the same intervention — open question for Phase 2 whether they reflect the same underlying mechanism (anchor-signal concentration with metric-resolution differences) or two mechanisms. Paired analysis now integrated in `scripts/analyze_e4_mitigation.py` (`_paired_anchor_damage`); writes `outputs/e4_mitigation/_summary/anchor_damage_paired_{sweep,full}.csv` automatically. Phase 2 will have a paired anchor-damage table at full scale alongside the existing per-strength tables. Writeups (E4-mitigation.md / _ko.md, E4-mitigation-evidence.md / _ko.md) corrected; "anti-correlation surprise" framing now scoped to df only, em pattern reframed as "coherent damage / partial-recovery ratio across the cluster".
- **2026-04-25** — **E4 Phase 2 LLaVA-1.5-7b complete and replicates Phase 1 claim at full scale.** 88,650 records (100 %, target_only-skip optimisation). Headline: `direction_follow_rate` 0.2578 [0.2515, 0.2640] → 0.2122 [0.2060, 0.2182] = −4.55 pp / **−17.7 % relative** (Phase 1 sweep predicted exactly −17.7 %; CIs at Phase 2 are ~10× narrower). `exact_match` rises slightly: 0.3340 → 0.3418 (+0.77 pp). Paired anchor-damage table on the full dataset (n_paired = 17,724, virtually no parse loss for LLaVA): em(TO) 0.3696, em(num@0) 0.3340, em(num@s*) 0.3417 → anchor damage **−3.55 pp** (vs Phase-1 sweep's susceptibility-stratified −7.00 pp, as expected — full set has more typical samples) and recovery **+0.77 pp = 21.7 % of damage** at the chosen working point. Replication is tight on every Phase-1 metric direction; the *relative* mitigation generalises from stratified to representative. Phase 2 chain auto-continued to convllava-7b at 20:24 UTC; ConvLLaVA Phase 2 ETA ~5.7 h. InternVL3 Phase 2 will not start in this 12-h window. Writeups updated: `docs/experiments/E4-mitigation.md` (+ _ko mirror) Phase 2 sub-section + LLaVA tables; `docs/insights/E4-mitigation-evidence.md` (+ _ko mirror) Phase 2 headline section.
- **2026-04-26** — **E4 Phase 2 ConvLLaVA-7b complete; mid-stack cluster Phase 2 result is 2/3 (LLaVA + ConvLLaVA done, InternVL3 in flight, will not finish in this session).** Headline ConvLLaVA: `direction_follow_rate` 0.2283 [0.2226, 0.2346] → 0.2042 [0.1982, 0.2100] = −2.42 pp / **−10.6 % relative** (Phase 1 sweep predicted −10.3 %, replicated within 0.3 pp). `exact_match` rises 0.3522 → 0.3652 (+1.30 pp). Paired anchor-damage (n_paired = 17,722, parse loss negligible): em(TO) 0.4454, em(num@0) 0.3520, em(num@s*) 0.3651 → damage **−9.34 pp** and recovery **+1.31 pp = 14.0 % of damage** at the chosen working point. **Both completed mid-stack-cluster models reproduce their Phase 1 sweep mitigation effect to within 0.3 pp on relative df reduction, with CIs ~10× narrower.** Each shows +1 pp em improvement at `s*`, anchor damage of −3.6 to −9.3 pp on the paired full set, and recovery of 14–22 % at the chosen working point. **ConvLLaVA fluency-tail caveat at full scale**: `mean_distance_to_anchor` jumps 2.99 → 53.54 on the treated cell (vs Phase 1 sweep stratified 3.18 → 3.30). Outlier samples receive predictions far from any plausible anchor, dragging the mean up by ~17×; em still rises because the bulk improves enough to net positive, and df is per-pair so it is robust. For the paper: switch to median distance + fluency-degraded fraction count. Tracked in §"open follow-ups". InternVL3 Phase 2 auto-started by the chain at 02:27 UTC; rate ~0.20 sample/sec → ETA ~24 h (multi-tile + planned driver patch not yet applied). 105 of 88,650 records at writeup time, n=16 valid triplets — figures shown in `full_validation_compare.csv` for InternVL3 are not load-bearing. **Action item before next InternVL3 session**: apply the `max_new_tokens` patch (8 → 32) gated on InternVL3 model name in `scripts/e4_attention_reweighting.py`, then decide whether to discard the partial 105 records (cleaner) or keep them via resumability (saves 3 min). Writeups updated: `docs/experiments/E4-mitigation.md` (+ _ko mirror) ConvLLaVA Phase 2 sub-section + headline tables; `docs/insights/E4-mitigation-evidence.md` (+ _ko mirror) Phase 2 headline now covers both LLaVA and ConvLLaVA.
- **2026-04-26 (evening)** — **E4 Phase 2 InternVL3-8b complete; mid-stack-cluster Phase 2 panel is 3/3 done.** Headline InternVL3: `direction_follow_rate` 0.1035 [0.0981, 0.1089] → 0.0975 [0.0923, 0.1026] = −0.59 pp / **−5.8 % relative** at s* = −0.5. `exact_match` rises 0.5902 → 0.5950 (+0.49 pp). Paired anchor-damage (n_paired = 11,848 of 17,730 = 66.8 %, parse loss persists at full scale because the driver patch was applied *during* the run, not before): em(TO) 0.6325, em(num@0) 0.5938, em(num@s*) 0.5977 → damage **−3.87 pp** and recovery **+0.40 pp = 10.2 % of damage**. **The 10 %-relative-reduction roadmap target is missed on InternVL3 (5.8 %)** for a structural reason — it is the H6 "distraction-not-anchoring" model, the Phase-2 full set has df₀ = 0.103 (~36 % lower than the Phase-1 stratified set's 0.161), and the mitigation effect at the working point scales with the baseline signal it is removing. The mitigation still moves the metric in the right direction, em rises, fluency is invariant (mean_dist 4.61 → 4.81). **Cross-cluster summary**: LLaVA (−17.7 % rel), ConvLLaVA (−10.6 %), InternVL3 (−5.8 %); all three rise on em (+0.49 to +1.30 pp); paired anchor-damage range −3.55 to −9.34 pp; recovery range 10.2 to 21.7 %. The anti-correlation between baseline df and mitigation effect (Phase-1 finding) holds at full scale: InternVL3 has the lowest df₀ and the smallest relative reduction, LLaVA the highest df₀ and the largest. Writeups updated: `docs/experiments/E4-mitigation.md` (+ _ko mirror) Phase 2 InternVL3 sub-section + headline tables; `docs/insights/E4-mitigation-evidence.md` (+ _ko mirror) Phase 2 headline now covers all 3 models. Phase 2 chain (`scripts/run_e4_phase2_chain.sh`) full run took 60,144 s (16.7 h) for InternVL3.
- **2026-04-27** — **E5 ChartQA full run complete on existing 3-model panel (qwen2.5-vl-7b, qwen3-vl-8b, llava-next-interleaved-7b).** Driver: `scripts/run_experiment.py --config configs/experiment_chartqa.yaml`, GPU 1, 16,170 records per model (5,390 sample-instances × 3 conditions), wall ~32 min per model (advisor's 3.7 h estimate was conservative — ChartQA target images and prompts are smaller than VQAv2 Number). **Cross-dataset signature differs sharply from VQAv2:** `direction_follow_rate` is *higher* on ChartQA (0.230, 0.252, 0.394 for the three models vs VQAv2 main 0.089–0.348), `adoption_rate` is *lower* (0.015–0.022 vs VQAv2 0.110–0.141, well below chance 0.11), and `accuracy_exact` is essentially *invariant* across conditions (TO 0.654 → num 0.654 on qwen2.5-vl; TO 0.316 → num 0.305 on llava-interleave). Reading: in ChartQA the target image already contains a legible answer number, so anchor cannot *replace* the prediction (em invariant) but can *tilt* the prediction direction (df strong). VQAv2 number subset has no such target-side number anchor, so anchor competes more directly. This is a paper-grade cross-dataset finding: **anchoring magnitude varies systematically by domain, with the "tilt vs replace" decomposition driven by whether the target image carries a competing legible number**. Caveats: (i) ChartQA mean_distance_to_anchor explodes (1.1 k–33 k) because ChartQA GT distribution is much wider than 0–9, so absolute distance is not directly comparable to VQAv2; relative df / em are the load-bearing metrics. (ii) Single-prompt run (no paraphrase robustness yet — E7). (iii) 3-model panel is the Phase-1 main subset, NOT the mid-stack cluster (no E4 mitigation generalisation claim from this run yet). Open follow-ups: (a) re-run on mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) if E4 generalisation to ChartQA is needed for the paper, (b) Phase A-style analysis (per-anchor digit, wrong-vs-correct stratification) on this dataset, (c) **region-aware attention re-analysis** flagged by user — current E1 measures attention on the whole anchor image span; digit-pixel patches would isolate the digit-specific signal vs background patches. Writeups (E5-chartqa.md / _ko mirror, E5-chartqa-evidence.md) deferred to next session.
- **2026-04-25** — **E4 Phase 1 sweep complete on the full mid-stack-cluster panel; Phase 2 chain launched.** convllava-7b finished: baseline df=0.290 → s*=−2.0 df=0.260 (−10 % rel, em delta +0 pp). internvl3-8b finished: baseline df=0.161 → s*=−0.5 df=0.132 (−17.7 % rel, em delta +1.9 pp). All three models hit the ≥ 10 % `direction_follow_rate` reduction target with em either flat or rising; em(target_only) invariant on every model on every strength (LLaVA 0.435, ConvLLaVA 0.500, InternVL3 0.568). **Three Phase-1 findings worth keeping for the writeup:** (i) **per-model `s*` is required, not a single shared strength** — `s*` ranges from −0.5 (InternVL3) to −3.0 (LLaVA-1.5), an order of magnitude apart, so a shared constant would over-mitigate one and under-mitigate the others. The mitigation generalises *as a locus + selection rule*, not a single strength constant. (ii) **mitigation effect is anti-correlated with baseline anchor-pull, not proportional** — InternVL3 (lowest baseline df = 0.16, the H6 "distraction-not-anchoring" model) shows the *largest* relative drop (−17.7 % at `s*`, −61 % at saturation); LLaVA-1.5 (highest baseline df = 0.305) shows the *smallest* (−13 % at `s*`, −18 % at saturation). Mechanism conjecture: the upper-half attention pathway carries a *larger fraction* of the anchor signal in the model that uses it less, consistent with InternVL3 having a narrowly-concentrated anchor signal vs. LLaVA-cluster's broader / redundant distribution. To confirm at Phase 2 scale. (iii) **InternVL3 prose-leak parse loss caveat** — ~30 % of records drop out of the valid-triplet count because InternVL3 emits prose ("based on…") that the parser misclassifies as a numeric string ("based"); n drops from 200 to 112–137 per cell. Comparison-internally consistent (baseline and treated share the same noise floor) but absolute em comparison across conditions is approximate. Driver fix logged for Phase 2 (regex `extract_first_number` rescue from `vlm_anchor.utils`). Phase 2 chain (`scripts/run_e4_phase2_chain.sh`) now running on GPU 0 in priority order LLaVA → ConvLLaVA → InternVL3, resumable across the 12-h session boundary. Writeups updated: `docs/experiments/E4-mitigation.md` (+ `_ko.md`) cross-model summary 3/3 done; `docs/insights/E4-mitigation-evidence.md` (+ `_ko.md`) headline + anchor-damage table now include InternVL3.
- **2026-04-25** — **E4 Phase 1 strength-sweep started; llava-1.5-7b complete.** Driver `scripts/e4_attention_reweighting.py`, analysis `scripts/analyze_e4_mitigation.py`, n=200 stratified, 7 strengths × 3 conditions = 4,200 records per model. **llava-1.5-7b:** baseline df_num=0.305 → s=−3.0 df_num=0.265 (−13 % relative; meets ≥ 10 % target); em_num 0.365 → 0.370 at s=−3.0 (well within ≤ 2 pp budget) and rises to 0.395 at saturation (s=−10⁴); em(target_only) invariant at 0.435 across all strengths confirms the hook is anchor-condition-specific (no leakage into single-image inference). **convllava-7b** sweep running on GPU 0 (≈ 60 % done at 15:25), partial baseline df_num=0.126 (lower than llava — convllava is more anchor-resistant on this stratified set), em_num=0.563 (higher); convllava effect size will be smaller in absolute terms. **internvl3-8b** queued behind. After all 3 sweeps, Phase 2 full-scale (n=17,730 × 3 conditions × 2 modes ≈ 88 k generations after target_only-skip optimization) runs in priority order (llava-1.5-7b first per advisor — cleanest E1d signal, no caveats). Phase 2 design is resumable (append-only JSONL with completed-key skip) so the run continues across the 12-h session boundary. Writeups: `docs/experiments/E4-mitigation.md` (+ `_ko.md`) and `docs/insights/E4-mitigation-evidence.md` (+ `_ko.md`); both flagged "Phase 1 in progress" until full validation lands.
- **2026-04-25** — **E1d causal anchor-attention ablation done across the 6-model panel.** Driver `scripts/causal_anchor_ablation.py`, analysis `scripts/analyze_causal_ablation.py`, n=200 stratified per model, 7 ablation modes (`baseline`, `ablate_layer0`, `ablate_peak`, `ablate_peak_window`, `ablate_lower_half`, `ablate_upper_half`, `ablate_all`). **Three findings.** (i) **Single-layer ablation is null on 6/6 models — at the E1b peak *and* at layer 0** (`Δ direction_follow ∈ [−0.032, +0.020]` for peak, `[−0.027, +0.005]` for layer 0; all CIs overlap baseline). The layer-0 control rules out reading (b) "peak is correlational and a different single layer matters" — even on Gemma, whose E1b reported anchor↔target swaps at L0–4 (Gemma layer-0 Δ = +0.005). Multi-layer redundancy confirmed: anchor's effect is encoded redundantly across the LLM stack, so any single-layer attention-mask ablation leaves the answer unchanged. (ii) **Stack-wide ablation reduces `direction_follow` 11–22 pp universally but breaks fluency on 3/6** (mean-distance balloons 4–6× on Gemma/LLaVA-1.5/ConvLLaVA, ~3 orders of magnitude on FastVLM). The 11–22 pp drop is an *upper bound* on the causal anchor pathway, not a target. (iii) **Upper-half attention ablation is the single architecture-blind mitigation locus** that reduces direction-follow on 6/6 (−5.5 to −11.5 pp) and is fluency-clean on 4/6 (mid-stack cluster + Qwen). Mid-stack cluster is the highest-leverage E4 prototype target — three encoders, one shared upper-half-clean response. **Sub-finding flagged as caveat:** ConvLLaVA and LLaVA-1.5 share the same E1b peak/mechanism but respond *opposite* to lower-half ablation (ConvLLaVA Δ = −0.120, LLaVA-1.5 Δ = +0.165) — same-attention-signature does not imply same-causal-structure. **Roadmap effects:** §6 E1 row "causal test" open question closed; §6 E4 row updated to specify upper-half multi-layer prototype on mid-stack cluster, single-layer ruled out. Open new questions: head-level sparsity; multi-layer combinatorial ablation. Writeups: `docs/experiments/E1d-causal-ablation.md` (+ _ko mirror), `docs/insights/E1d-causal-evidence.md` (+ _ko mirror).
- **2026-04-24** — **H3 formally retired; depth-axis framing replaces it.** Distilled insight written: `docs/insights/E1c-h3-falsified.md` (+ _ko mirror). H3's "ConvNeXt < ViT" hypothesis fails at both behavioural (E2 pilot adoption) and mechanistic (E1b per-layer) levels. Three architecturally different encoders (CLIP-ViT / InternViT / ConvNeXt) converge on the same mid-stack text-stealing profile. Paper narrative shifts from "encoder architecture modulates anchoring" to "post-projection LLM stack depth is the axis". Consequence: the originally planned E2 "encoder-ablation" subsection is no longer needed; compute can be re-routed to E5 (multi-dataset) or E7 (paraphrase robustness). H3 status in §2 updated from ⚠️ to ❌.
- **2026-04-24** — **E1 inputs_embeds-path extension done; 6-model panel complete.** Added ConvLLaVA (ConvNeXt encoder, inputs_embeds generate path) and FastVLM (FastViT, -200-marker expansion path) to the attention extraction pipeline in `scripts/extract_attention_mass.py` via new `EagerConvLLaVARunner` / `EagerFastVLMRunner` subclasses. Full n=200 runs complete for both. **Two key new findings:** (i) **H3 "ConvNeXt < ViT" is definitively falsified at the per-layer level** — ConvLLaVA's peak layer is L16 (same as LLaVA-1.5), mechanism is text-stealing (identical), magnitude +0.022 (within 20 % of LLaVA-1.5). Three encoders (CLIP-ViT, InternViT, ConvNeXt) now form a tight "mid-stack text-stealing" cluster. (ii) **FastVLM is a new archetype:** late peak (L22, matching Qwen depth) + text-stealing budget (−0.034, matching Gemma kind) + Gemma-level magnitude (+0.047) + panel-largest A7 gap (+0.086, with n=75 and wide CI caveat). Two published VLM failure modes — typographic attack and anchor-vs-target budget confusion — appear to co-fire in FastVLM. The 3-archetype story (from the 4-model E1b) refines to 4 archetypes. E4 design can now proceed with per-family intervention sites; the mid-stack cluster is the highest-leverage target (one intervention could generalise to three encoders). See `docs/experiments/E1b-per-layer-localisation.md` for the updated 6-model panel.
- **2026-04-24** — **E1b per-layer localisation done** (same 4 models × n=200). Peak layer differs sharply by encoder family: SigLIP-Gemma **layer 5/42** (12 % depth, δ +0.050, spike flanked by anchor/target trade-off layers), Qwen-ViT **layer 22/28** (82 %, δ +0.015, A7 gap +0.025 with bottom-decile CI including zero), CLIP-ViT (LLaVA-1.5) **layer 16/32** and InternViT (InternVL3) **layer 14/28** (both mid, δ ~+0.019). Layer-averaged E1 numbers were hiding a ~3× concentration at a single layer. **Second axis — budget decomposition:** at peak, Gemma/LLaVA-1.5/InternVL3 pull anchor mass from *text* (δ_text −0.014 to −0.038), Qwen pulls from *target image* (−0.010, text −0.005). Two distinct mechanisms: text-stealing vs target-stealing. **Candidate E4 intervention sites per family, to be tested:** Gemma → input-side pre-layer-5 KV/projection patch (denies text→anchor pull); Qwen → late-stack anchor attention re-weighting layer 22±2 gated by susceptibility (returns mass to target); CLIP/Intern → mid-stack ~14–16 (returns mass to text — less ideal, still testable). These are observational conjectures; E4 will test whether any of them actually reduce `direction_follow`. `docs/experiments/E1b-per-layer-localisation.md` (detailed) + `docs/insights/E1b-per-layer-localisation.md` (distilled).
- **2026-04-24** — E1 extended to 4 encoder families (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b; n=200 each). **Three claims settled at 4-model scale:** (i) anchor>neutral attention robust 4/4 (answer-step mean +0.004 to +0.007, CI excludes 0); (ii) H2 `wrong>correct` attention asymmetry falsified 4/4 — uncertainty does not modulate mean anchor attention; (iii) A7 `susceptible>resistant` holds 3/4 at answer step, inverts in Gemma-SigLIP (which also concentrates signal at step 0, consistent with typographic-attack inheritance). Candidate 3-claim paper structure emerges: anchor notice (attention) is robust; anchor pull (behaviour) is encoder-modulated; uncertainty modulates pull (Phase A) but not attention. `docs/experiments/E1-preliminary-results.md`.
- **2026-04-24** — Bilingual docs convention adopted. Every md under references/roadmap.md or research/ now has a `_ko.md` Korean mirror. English `.md` is canonical (Claude reads/edits it first); Korean version updated in lockstep. Memory entry: `feedback_bilingual_docs.md`. **Retired 2026-04-27 (commit `84f9341`).**
- **2026-04-24** — User decision (option 2): defer E2+E3 full 4-model run; prioritise E1 attention extraction + E4 mitigation. Pilot data + Phase A is sufficient to prototype mitigation. Re-open E2+E3 only if E1 cannot mechanistically separate anchor-pull from multi-image distraction. Phase B sequence in §7 updated.
- **2026-04-24** — Bug fix: `vlm_anchor.data.load_number_vqa_samples` now calls `Image.verify()` and silently skips undecodable images. Prevents the `000000000136.jpg` PIL crash from killing future multi-day runs.
- **2026-04-24** — E2 pilot (n=1,125 × 4 models) complete. **H3 in simple "Conv < ViT" form not supported** — ConvLLaVA adoption=0.156 falls inside the CLIP/SigLIP cluster CI. **New H6 added**: cross-modal failures decompose into two orthogonal axes (anchor-pull vs. multi-image distraction). InternVL3 = pure distraction (low adoption, high acc_drop), LLaVA-1.5 = pure anchoring (high adoption, low acc_drop), ConvLLaVA = both. Two-axis framing replaces "encoder family universally matters" as the candidate paper headline. See `docs/experiments/E2-pilot-results.md`. Full 17,730 runs for all 4 models queued, awaiting user signoff.
- **2026-04-24** — Phase A complete. Headline (H2): anchoring is uncertainty-modulated **graded pull**, not categorical capture (`docs/insights/A1-asymmetric-on-wrong.md`). Per-digit asymmetry confirmed (A2). Cross-model correlations 0.15–0.31 (A7) → both encoder and content matter, motivating E1+E2. A3/A4/A5/A6 folded into `00-summary.md`. Decision triggers in §7 fired — Phase B order unchanged.
- **2026-04-24** — Roadmap created. Status reflects: 7 models × full VQAv2 (standard + strengthen prompts) done; 5 new models integrated but not yet in main runs; 3 dataset extensions at smoke-only. Phase A queued.
