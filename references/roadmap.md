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
| `experiment` (standard prompt) | VQAv2 number | b/a/d | 7 (gemma3-27b-it, gemma4-31b-it, gemma4-e4b, llava-interleave-7b, qwen2.5-vl-7b, qwen3-vl-8b, qwen3-vl-30b) | ✅ M2 + C-form re-aggregated 2026-04-28; numbers in §3.3 |
| `experiment_anchor_strengthen_prompt` | VQAv2 number | b/a/d | same 7 | ✅ + strengthen-anomaly caveat (§9) |
| `experiment_encoder_pilot` | VQAv2 number | b/a/d, n=1,125 | llava-1.5-7b, internvl3-8b, fastvlm-7b, convllava-7b | ✅ pilot only — full run deferred (kept) |
| `experiment_distance_vqa` (E5b) | VQAv2 | b + a×S1..S5 | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_distance_tally` (E5b) | TallyQA | b + a×S1..S5 | llava-interleave-7b | 🟡 single-model — cross-model expansion in flight |
| `experiment_e5c_vqa` | VQAv2 | b + a×S1..S5 + m×S1..S5 + d | llava-interleave-7b, qwen2.5-vl-7b-instruct, gemma3-27b-it | ✅ 3/3 — gemma3-27b cell landed 2026-04-29 (n=12,000) |
| `experiment_e5c_tally` | TallyQA | same | llava-interleave-7b, qwen2.5-vl-7b-instruct, gemma3-27b-it | ✅ 3/3 — gemma3-27b cell landed 2026-04-29 at `max_samples=300` (n=3,600; full n=1000 base infeasible per §9) |
| `experiment_e5d_chartqa_validation` | ChartQA | per-dataset cutoff validation | llava-interleave-7b | ✅ S1-only relative cutoff adopted |
| `experiment_e5d_mathvista_validation` | MathVista | same | llava-interleave-7b | ⚠ C3 FAIL — see §9 (MathVista (γ) supersedes) |
| `experiment_e5e_chartqa_full` | ChartQA | b/a/m/d (S1) | llava-interleave-7b, qwen2.5-vl-7b, gemma3-27b-it | ✅ |
| `experiment_e5e_tallyqa_full` | TallyQA | b/a/m/d (S1) | same 3 | ✅ — gemma3-27b cell landed 2026-04-29 (inference 2026-04-28 23:28, C-form re-aggregation 2026-04-29) |
| `experiment_e5e_mathvista_full` (γ-α) | MathVista | b/a/m/d (S1) | llava-interleave-7b, qwen2.5-vl-7b, gemma3-27b-it | ✅ landed 2026-04-29 — `docs/insights/E5e-mathvista-evidence.md` |
| MathVista (γ-β) reasoning-mode | MathVista | b/a/m/d (S1) | qwen3-vl-8b-instruct + qwen3-vl-8b-thinking | ✅ landed 2026-04-28 — thinking *amplifies* anchor pull (S1 anchor arm, all-base, n=365: instruct adopt(a)=0.074 df(a)=0.102 → thinking adopt(a)=0.117 df(a)=0.291; ratios ×1.6 adopt, ×2.9 df). VLMBias / LRM-judging gain confirmed |
| VQAv2 4-condition (b/a/m/d) | VQAv2 | full grid cross-model | TBD | ☐ P1 (kept, time-permitting) |

### 3.2 Mechanistic runs

| Experiment | Models | n | Status |
|---|---|---|---|
| E1 attention-mass | gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b | 200 stratified | ✅ |
| E1b per-layer localisation | same 6 | 200 | ✅ — 4 archetypes (SigLIP-Gemma early, mid-stack cluster CLIP-ViT/InternViT/ConvNeXt, Qwen-ViT late, FastVLM late text-stealing) |
| E1d causal ablation | same 6 | 200 | ✅ — single-layer null on 6/6; upper-half multi-layer **−4.0 to −10.5 pp** (C-form) on 6/6 |
| E1-patch (digit-pixel attention) | gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b (4 perfect-square archetypes, n=400 each) | analysis + 2026-04-29 extension extraction | ✅ 4/6 — peak digit/anchor 0.468–0.631 (+24 to +40 pp above fair share) on every panel model. internvl3-8b (multi-tile) + qwen2.5-vl-7b (17×23 non-square) deferred (see §6.5 "E1-patch non-square archetypes") |
| E4 mitigation Phase 1 (sweep) | llava-1.5-7b, convllava-7b, internvl3-8b | 200 × 7 strengths | ✅ |
| E4 mitigation Phase 2 (full validation) | same 3 | 17,730 | ✅ |
| E4 generalisation to other archetypes | gemma4-e4b, qwen2.5-vl-7b, fastvlm-7b | TBD | ☐ P3 |

### 3.3 Headline numbers (C-form re-aggregation, 2026-04-28)

Full tables (standard-prompt 7-model panel, E5b distance sweep,
E5c digit-pixel causality 2/3 models, E5e ChartQA+TallyQA 3-model,
E1d / E4 mechanism summary) live in
[`docs/insights/headline-numbers.md`](../docs/insights/headline-numbers.md).
Quick orientation:

- **VQAv2 main panel:** `adopt(a)` 0.021–0.066, `df(a) C-form`
  0.085–0.274 across 7 models — graded pull, not categorical.
- **E5b distance:** sharp S1 peak (0.134 / 0.098 wrong-base
  VQAv2 / TallyQA), floor by S5 — plausibility-windowed.
- **E5c digit-pixel (3-model panel, S1, wrong-base, paired
  conditional adopt):** llava-interleave-7b VQAv2/TallyQA `a−m` =
  +6.1 pp / +2.6 pp; qwen2.5-vl-7b at floor on both arms (+0.4 / −0.5 pp);
  gemma3-27b-it VQAv2 `a−m` = **+5.69 pp adopt / +5.99 pp df**
  (anchor adopt 0.138 vs masked 0.082; df 0.280 vs 0.221 — second-largest
  cell after llava). gemma3-27b-it TallyQA at `max_samples=300`
  shows `a−m` = **+2.05 pp adopt / −0.33 pp df**; the df-side −0.33 pp
  is a 1-sample noise-floor artefact (df-eligible n≈95/arm,
  95 % CI half-width ~±8 pp at p≈0.19), not a sign reversal —
  consistent with main-panel ranking that gemma3 sits between
  llava and qwen on TallyQA susceptibility.
- **E5e ChartQA + TallyQA:** 3-model panel, all `a > m`.
- **E4 mitigation:** LLaVA-1.5 / ConvLLaVA / InternVL3 mid-stack-cluster
  Phase 2 C-form `df` reduction −14.6 % / −9.6 % / −5.8 % rel; em ↑;
  acc(b) invariant.

Numbers re-aggregate from raw `predictions.jsonl` via
`scripts/reaggregate_paired_adoption.py` if drift is suspected.

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
| **E5b distance sweep** | 5-stratum × b + a (VQAv2 + TallyQA), llava-interleave-7b | ✅ 3/3 — qwen2.5-vl-7b landed 2026-04-29 (E5c is strict superset); gemma3-27b-it landed 2026-04-29 (TallyQA at `max_samples=300`) |
| **E5c digit-mask control** | + 5-stratum × m, llava-interleave-7b | ✅ 3/3 — qwen2.5-vl-7b landed 2026-04-29; gemma3-27b-it landed 2026-04-29 (TallyQA at `max_samples=300`) |
| **E5d per-dataset cutoff** | ChartQA: S1-only relative `\|a-gt\| ≤ max(1, 0.1·gt)`; TallyQA: absolute `[0,5]`; VQAv2: range `{0..9}`; MathVista: C3 FAIL, scope-out as plausibility-window contrast (or rerun stricter) | ✅ except MathVista — see γ |
| **E5e S1-only cross-model** | b/a/m/d × ChartQA + TallyQA × 3 models | ✅ |
| **E5b/c cross-model expansion** | extend E5b + E5c to 3-model E5e panel (qwen2.5-vl-7b, gemma3-27b-it ∪ llava-interleave-7b) on VQAv2 + TallyQA | ✅ 3/3 — qwen2.5-vl-7b landed 2026-04-29 (a−m at floor on both datasets); gemma3-27b-it landed 2026-04-29 (VQAv2 n=12,000 full; TallyQA at `max_samples=300`, n=3,600). Headline: VQAv2 `a−m` = +5.7 pp adopt / +6.0 pp df (second-largest, behind llava); TallyQA `a−m` = +2.1 pp adopt / df-tie. Direction-consistent with main-panel ranking |
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
| **E1-patch (POC + perfect-square panel extension)** | digit-pixel-patch attention reanalysis. POC headline (2026-04-29): gemma4-e4b digit/anchor = 0.631 (peak L9, +0.404 above fair share); llava-1.5-7b digit/anchor = 0.468 (peak L7, +0.241 above fair share). Extension 2026-04-29: ConvLLaVA-7b (CLIP-ConvNeXt, 24×24 = 576-token square span) and FastVLM-7b (FastViT, 16×16 = 256-token square span) re-use the existing perfect-square `_compute_anchor_bbox_mass` path with no new code. Pipeline: bbox JSON via `scripts/compute_anchor_digit_bboxes.py`; extraction via `scripts/extract_attention_mass.py --bbox-file`; analysis via `scripts/analyze_attention_patch.py`. `docs/insights/E1-patch-evidence.md`. | ✅ 4-model panel landed 2026-04-29 (gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b); InternVL3-8b + Qwen2.5-VL-7b deferred — see "E1-patch non-square archetypes" row |
| **E1-patch non-square archetypes** | InternVL3-8b (3328-token multi-tile span — needs per-tile bbox routing) and Qwen2.5-VL-7b (391-token = 17×23 non-square span — needs `grid_thw` plumbing through the processor). Each requires its own per-encoder bbox-to-token mapping in `_compute_anchor_bbox_mass` because the current `int(math.isqrt(n)) ** 2 == n` gate returns None on these spans by design. Estimated 1–2 days each — 4–7 days panel-wide. | ☐ P3 (deferred — 2026-04-29 audit corrected the original §7 P1 1.5h budget after discovering the actual implementation cost) |
| **E1-patch masked-arm causal control** | re-run extraction on the 4-model panel using a 4-cond config (b/a/m/d) instead of the existing 3-cond `configs/experiment.yaml`. Pairs `image_anchor_digit` on the anchor arm against the masked arm's anchor-region attention as a digit-pixel causal control. Adds ~1 hour GPU per model + 4-cond config wiring. | ☐ P3 (deferred 2026-04-29 — independent of the non-square work above) |
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

**As of 2026-04-29 (post-gemma3-27b E5c)** — all paper-blocking P0s
have landed (M2-refactor + C-form, L1-L4 confidence, γ-α + γ-β
MathVista, E1-patch POC, paper §3/§7.4/§8 prose, E5e TallyQA
gemma3-27b cell, qwen2.5-vl-7b E5c VQAv2 + TallyQA, **gemma3-27b-it
E5c VQAv2 + TallyQA**). No P0 outstanding — remaining queue is
P1/P3 strengthening / future work.
| **P3** | E1-patch full panel non-square archetypes — InternVL3-8b (multi-tile bbox routing) and Qwen2.5-VL-7b (`grid_thw` plumbing). 2026-04-29 audit re-budgeted the original §7 1.5h estimate to 4–7 days panel-wide after finding the bbox-to-token mapping is per-encoder (POC's `int(math.isqrt(n)) ** 2 == n` gate returns None on multi-tile / rectangular grids). ConvLLaVA-7b + FastVLM-7b were perfect-square and landed in the 4-model panel 2026-04-29. | §6.5 E1-patch | ~1–2 days/model implementation + ~12 min/model H200 extraction |
| **P3** | E1-patch masked-arm causal control — re-run extraction on 4-model panel under 4-cond config | §6.5 E1-patch | 4-cond config wiring + ~1h GPU/model |
| **P1** | VQAv2 4-condition cross-model (b/a/m/d, S1 only, kept) | §6.3 | ~1d (3 models) — opportunistic |
| **P1** | Citation verification — every 2026 arXiv ID in `references/project.md` and §2 paper draft must resolve to a real paper | §9 caveat | hours of manual verification, reviewer-defuse |
| **P3** | E4 generalisation to other archetypes (Gemma / Qwen / FastVLM) | §6.5 | days |
| **P3** | Image-vs-text anchor (F2) follow-up paper | §6.6 | future |
| **P3** | VQAv2 main panel logit re-run (L6 — no logit capture pre-commit `5f925b2`) | §6.4 | opportunistic |

**Recently landed (struck from queue 2026-04-29):**

- ~~gemma3-27b-it on E5c VQAv2 + TallyQA~~ ✅ (2026-04-29 — VQAv2 full n=12,000 ran in ~95 min on GPU 1; TallyQA at `max_samples=300` ran in ~28 min. Both re-aggregated to C-form via `reaggregate_paired_adoption.py --apply`; `docs/insights/_data/E5c_per_cell.csv` rebuilt by `analyze_e5c_distance.py --models llava-next-interleaved-7b qwen2.5-vl-7b-instruct gemma3-27b-it`. VQAv2 wrong-base S1 `a−m` = +5.69 pp adopt / +5.99 pp df — second-largest panel cell, behind llava. TallyQA wrong-base S1 `a−m` = +2.05 pp adopt / df-tie. Closes the only remaining paper-blocking P0)
- ~~E5e TallyQA gemma3-27b cell~~ ✅ (inference 2026-04-28 23:28; C-form re-aggregation 2026-04-29 — `predictions.jsonl` rewritten with `_moved` flag, `summary.json` refreshed, `docs/insights/_data/experiment_e5e_tallyqa_full_per_cell.csv` rebuilt with the new model row, §3.3 panel updated)
- ~~C-form re-aggregation of gemma3-27b TallyQA cell + §3.3 panel insertion~~ ✅ (2026-04-29; folded into above)
- ~~qwen2.5-vl-7b on E5c VQAv2 + TallyQA~~ ✅ (2026-04-29 — both runs landed n=12,000 each; `analyze_e5c_distance.py` extended to multi-model with `--models` flag; `docs/insights/_data/E5c_per_cell.csv` rebuilt with model column; per-model figures saved as `E5c_<kind>_<model>.png`. a−m gap at floor on qwen2.5-vl, consistent with §3.3 main-panel anchor-resistance ranking)

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
- **In-flight inference.** None — all paper-blocking inference cells
  have landed as of 2026-04-29 (gemma3-27b-it E5c VQAv2 + TallyQA,
  γ-α / γ-β MathVista, qwen2.5-vl-7b E5c VQAv2 + TallyQA, E5e
  TallyQA gemma3-27b cell).

## 10. Changelog

- **2026-04-29 (gemma3-27b-it E5c cross-model expansion landed —
  closes the last paper-blocking P0).** Two runs of `experiment_e5c_*`
  on gemma3-27b-it:

  - **VQAv2** at full config (n=1000 base × 12 conditions = 12,000
    records) — launched 2026-04-29 17:10 on GPU 1, finished ~18:45
    (~95 min H200 wall; well under the 5–6h budget).
  - **TallyQA** at `max_samples=300` (per §9 infeasibility note for
    full n=1000 stratified) — launched 2026-04-29 18:54 on GPU 1,
    finished ~19:23 (~28 min H200 wall).

  Both runs re-aggregated to C-form via
  `scripts/reaggregate_paired_adoption.py --apply --root
  outputs/experiment_e5c_{vqa,tally}/gemma3-27b-it` (15,600 records
  total, `_moved` flag persisted). `scripts/analyze_e5c_distance.py
  --models llava-next-interleaved-7b qwen2.5-vl-7b-instruct
  gemma3-27b-it` rebuilt `docs/insights/_data/E5c_per_cell.csv` (3-model)
  and emitted per-model figures `docs/figures/E5c_*_gemma3-27b-it.png`.

  **Headline (S1, paired conditional, post-C-form):**

  | Dataset | base | model | adopt(a) | adopt(m) | df(a) | df(m) | a−m adopt | a−m df |
  |---|---|---|---:|---:|---:|---:|---:|---:|
  | VQAv2 | all | gemma3-27b-it | 0.099 | 0.060 | 0.142 | 0.108 | +3.97 pp | +3.45 pp |
  | VQAv2 | wrong | gemma3-27b-it | 0.138 | 0.082 | 0.280 | 0.221 | **+5.69 pp** | **+5.99 pp** |
  | TallyQA | all | gemma3-27b-it | 0.051 | 0.028 | 0.086 | 0.069 | +2.34 pp | +1.68 pp |
  | TallyQA | wrong | gemma3-27b-it | 0.074 | 0.053 | 0.190 | 0.193 | +2.05 pp | −0.33 pp |

  **Reading.** Gemma3-27b-it sits between llava-interleave-7b
  (largest a−m gap) and qwen2.5-vl-7b (floor) on the §3.3 main panel;
  the E5c expansion confirms this rank: VQAv2 gemma3 a−m
  (+5.69 pp adopt / +5.99 pp df) is **second-largest behind llava**
  (+6.1 pp / +5.3 pp), and **clearly above qwen** (+0.4 pp / ~0).
  TallyQA gemma3 a−m on adopt (+2.05 pp) sits between llava (+2.6 pp)
  and qwen (−0.5 pp), with df at the noise floor on both arms —
  consistent with TallyQA being the harder dataset for any model
  to be pulled by an irrelevant numeric anchor (lower base accuracy
  → less plausible-window-resolved confidence to perturb in the
  first place). Direction-consistent across the panel: "largest
  pull → largest gap" holds for adopt, df separates llava-only on
  TallyQA.

  **Status changes:** §3.1 `experiment_e5c_vqa` / `experiment_e5c_tally`
  rows flipped from 🟡 2/3 to ✅ 3/3; §3.3 E5c headline now lists
  3-model panel; §6.3 E5b distance / E5c digit-mask / E5b/c
  cross-model expansion all flipped to ✅ 3/3; §7 P0 entry struck
  ("no P0 outstanding" header rewritten); §9 in-flight inference
  caveat retired.

- **2026-04-29 (doc hygiene).** Paper-deck deliverables relocated from
  `docs/figures/` to `docs/ppt/` (commit `8eda31c`); `*.pptx` removed
  from git tracking as regenerable artefacts (`53d3bbd`); cached
  `docs/deprecated/_ko-mirrors-2026-04-27/` (18 files) removed
  (`6fe7be2`); stale Status banners on 5 experiment + insight writeups
  backfilled to point at 2026-04-29 commits (`d6efecd`, `dc38d6c`,
  `abf10b3`). No findings change.

- **2026-04-29 (E1-patch perfect-square panel extension — 2 → 4 models).**
  POC (gemma4-e4b + llava-1.5-7b) extended to convllava-7b + fastvlm-7b
  on the same n=400 stratified attention dump infrastructure.
  Both new models have perfect-square anchor spans (convllava 576 =
  24², fastvlm 256 = 16²), so no new bbox-to-token mapping code was
  needed — the existing `_compute_anchor_bbox_mass` perfect-square
  path applies. Wall: convllava 5 min, fastvlm 18 min on H200.

  **4-model headline (peak `digit/anchor`, n=400 each):**

  | Model | Peak L | digit/anchor | concentration above fair |
  |---|---|---:|---:|
  | gemma4-e4b | L9 / 42 | 0.631 | +0.404 |
  | convllava-7b | L7 / 32 | 0.552 | +0.325 |
  | fastvlm-7b | L4 / 28 | 0.531 | +0.304 |
  | llava-1.5-7b | L7 / 32 | 0.468 | +0.241 |

  4/4 panel models exceed fair share (~0.227) by +24 to +40 pp. Three
  qualitative profiles: (A) globally digit-concentrated (gemma); (B)
  peaked mid-early then decay (llava-1.5 + convllava — same shape on
  two architecturally distinct CLIP-ViT vs CLIP-ConvNeXt encoders);
  (C) sharp early peak with sustained mid-stack plateau (fastvlm-7b).

  **Roadmap audit + scope correction.** Original §6.5 / §7 P1 row had
  budgeted "~1.5h attention extraction" for the full 6-model E1-patch
  panel. 2026-04-29 inspection revealed the 1.5h figure was for
  extraction only and assumed all archetypes were drop-in
  perfect-square; in reality InternVL3 (3328-token multi-tile) and
  Qwen2.5-VL (391-token = 17×23 non-square) each need per-encoder
  bbox-to-token mapping logic (1–2 days each, 4–7 days panel-wide
  including masked-arm causal control). The 1.5h budget was
  retrospectively rebasedto P3 in §7 with two new explicit deferred
  rows: "E1-patch non-square archetypes" and "E1-patch masked-arm
  causal control".

  **Surfaces updated:** `docs/insights/E1-patch-evidence.md` (rewritten
  for 4-model panel, TL;DR + §1 + §2 + §3 + §4 + §6 + §7 + §8 all
  reflect new numbers and three-profile-shape framing);
  `docs/paper/sections/07_mechanism_mitigation.md` (new §7.2.1
  "Digit-pixel concentration within the anchor (E1-patch)" inserted
  after §7.2); roadmap §3.2 (E1-patch row) + §6.5 (POC + non-square +
  masked-arm rows) + §7 P3 (deferred non-square + masked control).

- **2026-04-29 (qwen2.5-vl-7b E5c cross-model expansion landed).**
  Two runs of `experiment_e5c_*` (VQAv2 + TallyQA, b + a×S1-5 + m×S1-5
  + d, n=1000 base / dataset, n_total=12,000 / dataset) on
  qwen2.5-vl-7b-instruct, launched 2026-04-29 03:12 on GPU 0/1 in
  parallel and finished within ~50 min each. Re-aggregated to C-form
  via `scripts/reaggregate_paired_adoption.py --apply --root
  outputs/experiment_e5c_*/qwen2.5-vl-7b-instruct` (24,000 records
  total, `_moved` flag now persisted). `scripts/analyze_e5c_distance.py`
  extended to take `--models` (variadic) and emit a `model` column on
  `docs/insights/_data/E5c_per_cell.csv`; per-non-default-model figures
  saved as `docs/figures/E5c_<kind>_<model>.png`.

  **Headline (S1, wrong-base, paired conditional adoption):**

  | Dataset | llava-interleave-7b | qwen2.5-vl-7b | a−m llava | a−m qwen |
  |---|---:|---:|---:|---:|
  | VQAv2 (anchor / masked) | 0.129 / 0.068 | 0.070 / 0.066 | +6.1 pp | +0.4 pp |
  | TallyQA (anchor / masked) | 0.110 / 0.084 | 0.033 / 0.037 | +2.6 pp | −0.5 pp |

  Direction-follow `df_cond` shows the same pattern (llava VQAv2
  S1: anchor 0.208 / masked 0.155, qwen S1: 0.148 / 0.163 ~ tie).
  Distance decay holds qualitatively on qwen2.5-vl (S1 → S5 anchor
  adopt: VQAv2 0.070 → 0.003, TallyQA 0.033 → 0.000), so the
  plausibility-window claim from §5.3 generalises as a S1-peak /
  S3+ floor pattern.

  Reading: the digit-pixel-causality claim from §5.4 holds where
  the anchor pull is large enough to detect (llava-interleave); on
  qwen2.5-vl the entire E5c effect is at the noise floor on both
  arms, consistent with §3.3 main-panel ranking placing qwen2.5-vl
  at the lowest `df(a) = 0.094`. The cross-model expansion is
  direction-consistent ("largest pull → largest gap") but the
  absolute magnitude of the gap is model-dependent. gemma3-27b-it
  cell pending will arbitrate whether mid-panel models track
  llava-style or qwen-style behaviour.

  Status changes: §3.1 `experiment_e5c_vqa` / `experiment_e5c_tally`
  rows flipped from 🟡 single-model to 🟡 2/3 (gemma3-27b pending);
  §3.3 E5c headline panel rewritten to show both models side-by-side;
  §6.3 `E5b/c cross-model expansion` row updated; §7 P0 entry for
  qwen2.5-vl-7b struck and replaced with the focused gemma3-27b-it
  E5c VQAv2 P0 (~5–6h H200).

- **2026-04-29 (E5e TallyQA gemma3-27b cell landed end-to-end).**
  Inference finished 2026-04-28 23:28 UTC (n=38,245, full integer
  subset, b/a/m/d S1), ahead of the projected 30–35h wall budget.
  C-form re-aggregation ran 2026-04-29 via
  `scripts/reaggregate_paired_adoption.py --apply` (152,980 records
  rewritten in place; `predictions.marginal.bak.jsonl` /
  `predictions.marginal.bak.csv` / `summary.marginal.bak.json` retained
  for audit). `docs/insights/_data/experiment_e5e_tallyqa_full_per_cell.csv`
  was rebuilt via `scripts/analyze_e5e_wrong_correct.py --exp-dir
  experiment_e5e_tallyqa_full` (3 run dirs, 12 → 18 rows). §3.3 panel
  TallyQA section now carries the gemma3-27b row at the top (largest
  TallyQA `df(a)` cell): `acc(b) = 0.237 / acc(a) = 0.236 / adopt(a) =
  0.027 / adopt(m) = 0.016 / df(a) C-form = 0.073 / df(m) C-form =
  0.060`. Wrong-base S1 (per-cell): `adopt(a) = 0.059`, `df(a) C-form
  = 0.152`. §3.1 row 11 fully ✅, §7 P0 cleared.

  In parallel, qwen2.5-vl-7b cross-model E5c expansion (b + a×S1-5 +
  m×S1-5 + d on VQAv2 + TallyQA) launched 2026-04-29 on GPU 0/1
  with `--models qwen2.5-vl-7b-instruct`; logs at
  `outputs/_logs/e5c_*_qwen25vl_*.log`.

- **2026-04-28 (B안 — full C-form propagation to E1d / E4 / §7).**
  Closing the metric-consistency gap between §3.3 / §5 (C-form) and
  §7 (Phase A pull-form) by switching `analyze_causal_ablation.py` and
  `analyze_e4_mitigation.py` to read the canonical M2
  `anchor_direction_followed_moved` flag instead of computing a
  pull-form `(|num_pred − anchor| < |base_pred − anchor|)` directly.
  The `_moved` flag is the post-C-form refactor M2 numerator
  `(pa-pb)·(anchor-pb) > 0 AND pa != pb`, written into every
  `predictions.jsonl` by `reaggregate_paired_adoption.py`.
  Adoption rate also switched from pre-M1 marginal to M2 paired
  (denominator: `pred_b != anchor`).
  
  **Refactor scope:** (i) `_load_model` in causal_ablation grabs the
  M2 flags into the in-memory frame; (ii) `_compute_metrics` and
  `_bootstrap_ci` in both scripts use the M2 predicates per draw;
  (iii) `_build_triplets` in e4 carries M2 columns through the merge;
  (iv) the InternVL3 prose-leak `_to_int(rescue_text=decoded)` rescue
  is retained on `em_num` (gt-comparison), but dropped on the M2
  anchoring rates — matching the §3.3 / §5 main-panel handling
  (no rescue), making §7 consistent with the canonical metric.

  **Number changes (Phase 2, full-scale 88,650 records / model):**

  | model | df₀ pull → C-form | df at s* pull → C-form | rel reduction pull → C-form |
  |---|---|---|---|
  | LLaVA-1.5-7b | 0.258 → **0.288** | 0.212 → **0.246** | −17.7 % → **−14.6 %** |
  | ConvLLaVA-7b | 0.228 → **0.258** | 0.204 → **0.233** | −10.6 % → **−9.6 %** |
  | InternVL3-8b | 0.103 → **0.126** | 0.098 → **0.119** | −5.8 % → **−5.8 %** |

  **E1d ablation deltas (n=200 stratified, 6 models):**

  | mode | pull-form range | C-form range |
  |---|---|---|
  | `ablate_peak` | \|Δ\| ≤ 3.2 pp | \|Δ\| ≤ 2.0 pp |
  | `ablate_layer0` | Δ ∈ [−2.7, +0.5] pp | same range |
  | `ablate_upper_half` | −5.5 to −11.5 pp | **−4.0 to −10.5 pp** |
  | `ablate_all` | −10 to −22 pp | **−9.6 to −24.5 pp** |

  **em-based metrics unchanged** (em is gt-comparison; pull-form vs
  C-form divergence does not affect it). Paired anchor-damage /
  recovery numbers in §7.4 (21.7 % / 14.0 % / 10.2 %) are preserved.

  **Phase 1 vs Phase 2 s*** caveat — under C-form, the Phase 1 sweep
  would prefer s*=−5.0 for LLaVA / ConvLLaVA (instead of −3.0 / −2.0)
  to clear the same ≥10 % rel df reduction threshold. Phase 2 was
  launched at the original pull-form-optimal s*; we have not re-run
  Phase 2 at C-form-optimal s*. §7.4 paragraph + E4 evidence doc
  explicitly note this. `outputs/e4_mitigation/_summary/chosen_strength.json`
  is pinned to the historical Phase 2 s* values (with a `_note` key
  explaining the situation) so future re-runs of `--phase full` do
  not silently switch.

  **Surfaces updated:** `scripts/analyze_causal_ablation.py`,
  `scripts/analyze_e4_mitigation.py`, `outputs/e4_mitigation/_summary/{full_validation,full_validation_compare,sweep_pareto,anchor_damage_paired_full,anchor_damage_paired_sweep,chosen_strength}.{csv,json,png}`,
  `outputs/causal_ablation/_summary/{per_model_per_mode,by_stratum,fig_direction_follow,fig_adoption}.{csv,png}`,
  `docs/insights/E4-mitigation-evidence.md` (Phase 2 + Phase 1 tables
  + InternVL3 stratified-vs-full paragraph + anti-correlation paragraph),
  `docs/insights/E1d-causal-evidence.md` (mode results table +
  upper-half range + lower-half ConvLLaVA/LLaVA delta values),
  `docs/insights/paper-section-7-4-mitigation-free-lunch.md` (Phase 2
  table + anti-correlation prose), `docs/paper/sections/07_mechanism_mitigation.md`
  §7.3 ablation table + §7.4 Phase 2 table + s*-Phase 1-vs-Phase 2
  C-form caveat paragraph + anti-correlation prose,
  `docs/paper/sections/01_intro.md` (×2 abstract / §1.5 references
  to "5.8-17.7 %" → "5.8-14.6 %"), `references/roadmap.md` §3.2 +
  §3.3 headline numbers.

  **Test suite still green** (74/74) — the two analysis scripts have
  no unit-test coverage on the metric formula directly; the rates are
  re-validated end-to-end by reading the canonical M2 flag, which is
  itself tested in `tests/test_metrics.py` (15 metrics tests).

- **Older entries (≤ 2026-04-28 first wave).** All earlier changelog
  entries — overnight polish audits, γ-β / direction_follow_rate / E5e
  γ-β / paper-section prose / E1-patch POC / L1 confidence / γ-α / M2
  landed / roadmap restructured / E5b-E5c-M1 / E4 Phase 1+2 / E1d /
  Phase A + E1 / H3 retired — moved to
  [`docs/CHANGELOG.md`](../docs/CHANGELOG.md) on 2026-04-29 to keep
  the roadmap lightweight.
