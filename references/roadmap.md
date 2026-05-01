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

## 3. Status snapshot — where we are (2026-05-02 — Phase 1 paper architecture restructure)

### 3.0 Architecture restructure summary (2026-05-02)

Decision finalised 2026-05-02: paper-wide model + dataset consistency push.
See `references/project.md §0.4.1–§0.4.3` for full tiering. Headlines:

- **Main model**: `llava-interleave-7b` (every section primary).
- **Sub panel**: + `qwen2.5-vl-7b-instruct` + `gemma3-27b-it` (3-model behavioural).
- **Mechanism panel**: 5-model perfect-square (gemma4-e4b + llava-1.5-7b +
  convllava-7b + fastvlm-7b + **llava-interleave-7b**) — InternVL3 + Qwen2.5-VL
  → appendix only (non-perfect-square correctness risk).
- **Datasets**: 5-dataset main matrix at **n=5000 target** (TallyQA + ChartQA +
  MathVista + **PlotQA + InfographicVQA**); VQAv2 dropped from main, kept in
  appendix. PlotQA snapshot refetched at n=5000 stratified (5 gt bins × 1000),
  TallyQA capped at 5000 (38k full-set archived to `outputs/_legacy_tallyqa_n38k/`).
- **§7.4.5 mitigation**: recalibrate Subspace on **PlotQA + InfoVQA pooled**
  full gt range; sweep cap raised 500 → 5000 wrong-base (statistical-power
  revision based on wrong-base df being 3-5× stronger than correct-base).
- **§6 confidence (Phase 1.5 fix)**: multi-token answer support via
  `recompute_answer_span_confidence.py`. Handles Qwen2 / Llava-Interleave
  per-digit tokenization that broke single-token `answer_token_logit`
  capture on 75-89 % of PlotQA / InfoVQA / ChartQA predictions.
- **§3 acc(b) reporting**: dual strict + dataset-official supplementary
  (PlotQA-relaxed-5% / InfoVQA-ANLS≥0.5) for cross-dataset comparability
  + literature alignment.

### 3.1 Behavioural runs (historical context — paper consistency push will overlay)

The matrix below records what is on disk. Rows tagged `(historical)` were
run pre-restructure; Phase 1 may overlay them with new runs at the
canonical setup. Status flags: ✅ done · 🟡 partial / single-model · ⏳
in-flight · ☐ not started.

### 3.1 Behavioural runs (historical + Phase 1 plan)

**Phase 1 target rows** (new, plan):

| Experiment | Dataset | Conditions | Models | Status |
|---|---|---|---|---|
| `experiment_e7_plotqa_full` | PlotQA test V1 | b/a/m/d (S1) | Main + Sub-A + Sub-B | ⏳ Phase 1 — pilot landed 2026-05-01 (llava n=200; baseline df 0.327, a−m gap +14.6pp); **target n=2500 stratified × 3 models** |
| `experiment_e7_infographicvqa_full` | InfographicVQA val | b/a/m/d (S1) | Main + Sub-A + Sub-B | ⏳ Phase 1 — pilot landed 2026-05-01 (llava n=200; baseline df 0.190, a−m gap +5.1pp); **target n=1147 (full numeric) × 3 models** |
| E5b/c PlotQA + InfoVQA + ChartQA + MathVista 3-model | 4 datasets | b + 5×a-strat + 5×m-strat + d | same 3 | ☐ Phase 2 (digit-pixel causality breadth) |
| E6 Subspace recalibration on PlotQA + InfoVQA pooled, 5-dataset full-range eval | 5 datasets | b/a/m/d (S1) + cell sweep | Main only | ☐ Phase 1 — see §6.5 E6 |

**Historical rows (pre-2026-05-02 restructure)**:

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

**Main matrix datasets (5)** — finalised 2026-05-02:

| ID | Task | Status | Notes |
|---|---|---|---|
| **D2 — TallyQA** | snapshot + loader | ✅ `inputs/tallyqa_test/` (sub-dataset) |
| **D3 — ChartQA** | snapshot + loader | ✅ `inputs/chartqa_test/`, integer-GT [1,1000] gate (sub-dataset) |
| **D4 — MathVista** | snapshot + loader, integer-GT gate | ✅ `inputs/mathvista_testmini/` (sub-dataset) |
| **D5 — PlotQA** | snapshot + loader (V1, gt ≤ 10k, template ∈ {data_retrieval, min_max, arithmetic}) | ✅ `inputs/plotqa_test/`, fetcher `scripts/fetch_plotqa_test.py` (96K numeric Q-A available; 2,500 stratified for full panel) — **Main dataset** |
| **D6 — InfographicVQA** | snapshot + loader (val, positive int, gt ≤ 10k) | ✅ `inputs/infographicvqa_val/`, fetcher `scripts/fetch_infographicvqa_val.py` (1,147 numeric, gt-bin (20,100] dominant per percent-question heavy) — **Main dataset** |
| **I1 — anchor inventory** | FLUX-rendered digit images, range up to 10000 | ✅ `inputs/irrelevant_number/` |
| **I2 — anchor mask inventory** | OpenCV Telea inpaint of digit bbox | ✅ `inputs/irrelevant_number_masked/` |
| **I3 — neutral inventory** | digit-free FLUX renders, scene-balanced | ✅ `inputs/irrelevant_neutral/` |

**Dropped (2026-05-02 decision)**: `D1 — VQAv2` from main matrix. Multiple-GT
(10-annotator vote) + open-vocabulary text answers + full-eval impractical
+ legacy benchmark for 2024–2026 numeric VQA. TallyQA covers the natural
image counting axis VQAv2 contributed. Existing `inputs/vqav2_number_val/`
preserved for appendix breadth panel only.

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

**Restructured 2026-05-02**: §7.1–7.3 *analysis* anchored on **E1-patch
digit-bbox attention** (not full-anchor attention) on **5-model
perfect-square panel**. §7.4 E4 attention re-weighting extends to add
Main. §7.4.5 E6 Subspace recalibrates on PlotQA+InfoVQA pooled at full
gt range (no [0,8] restriction).

#### §7.1–7.3 Analysis (digit-bbox-centric)

| ID | Experiment | Status |
|---|---|---|
| **E1 / E1b / E1d (full-anchor, historical)** | full anchor-image attention pipeline (mass + per-layer + causal ablation), 6-model encoder-archetype panel | ✅ landed pre-restructure — kept as supplementary in §7.1 background; main §7.1–7.3 now anchored on E1-patch |
| **E1-patch perfect-square panel — 5-model (Phase 3 target)** | digit-pixel-patch attention. Existing 4-model panel (gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b) → **add Main `llava-interleave-7b` (SigLIP encoder, 27×27 = 729 perfect-square)**. Same `_compute_anchor_bbox_mass` perfect-square path, no new code. Existing headline: digit/anchor ratio +24–40 pp above fair share at peak layer on every panel model. | ⏳ Phase 3 — Main extraction ~30 min on H200; 5-model panel headline target |
| **E1-patch causal ablation (digit-bbox region zero-mask) — Phase 3** | causal ablation restricted to digit-bbox region (not full anchor span). Tests: does zeroing only the digit patch reproduce E1d's df reduction? Patch-level surgical ablation (anchor label needed at extraction time, not inference time). Replaces the existing full-anchor E1d as §7.3 main analysis. | ☐ Phase 3 — implementation depends on E1-patch attention path + bbox JSON pipeline |
| **E1-patch masked-arm causal control** | re-run extraction on 5-model panel under 4-cond config (b/a/m/d). Pairs anchor-arm digit-bbox attention against masked-arm anchor-region attention as digit-pixel causal control. | ☐ Phase 3 — 4-cond config wiring + ~1h GPU/model |
| **InternVL3-8b + Qwen2.5-VL-7b non-perfect-square (appendix only)** | InternVL3 multi-tile + Qwen2.5-VL 17×23 non-square. Per-encoder bbox-to-token routing required; correctness hard to guarantee uniformly. **Decision 2026-05-02: not in main §7 panel** — appendix-only mention with caveat. | ☐ appendix |

#### §7.4 E4 attention re-weighting mitigation

| ID | Experiment | Status |
|---|---|---|
| **E4 Phase 1 + 2 (existing 3-model mid-stack cluster)** | mid-stack-cluster attention re-weighting (LLaVA-1.5 / ConvLLaVA / InternVL3) | ✅ landed pre-restructure |
| **E4 + Main `llava-interleave-7b` (Phase 3)** | E4 sweep + full validation for Main model. Risk: Main may not fall in mid-stack cluster archetype (SigLIP ≠ CLIP-ViT/InternViT/ConvNeXt) → E4 may null-effect or backfire. Result drives §7.4 framing: if archetype-conditional, document as such; if cleanly transferred, headline strengthened. | ☐ Phase 3 — depends on E1-patch Main archetype assignment |
| **E4 §7.4 paper rendering** | report `direction_follow_rate` reduction, `exact_match` rise, `accuracy_vqa(b)` invariance side by side; the "free lunch" framing | ✅ `docs/insights/paper-section-7-4-mitigation-free-lunch.md` (2026-04-29) — needs Phase 3 update with Main extension |

#### §7.4.5 E6 Subspace deployable mitigation

| ID | Experiment | Status |
|---|---|---|
| **E6 (historical) — Tally-only N=5000 calibration, gt ∈ [0,8] eval** | Subspace L31_K04_α=1.0 clears 4-dataset selection rule. df −46% to −56% on TallyQA/ChartQA/VQAv2/MathVista; em +0.9 to +3.3 pp. Caveat: gt ∈ [0,8] restriction made result look like partial solution. | ✅ landed 2026-05-01 — historical baseline |
| **E6 Phase 1 — recalibrate on PlotQA + InfoVQA pooled, evaluate full gt range, 5-dataset** | New calibration source: pooled wrong-base sids from `experiment_e7_plotqa_full` + `experiment_e7_infographicvqa_full` (Main model baselines). Evaluate at full gt range (no [0,8] restriction) on 5-dataset matrix. **Risk**: chart+info-calibrated subspace may not transfer to small-gt natural-image counting (TallyQA). Plan B = pooled multi-source (chart + info + count) recalibration. | ☐ Phase 1 P0 — depends on Phase 1 §3 baseline runs |
| **E6 Pilot validation (2026-05-01, llava n=200)** | Existing Tally-calibrated subspace tested on PlotQA + InfoVQA pilots: PlotQA gt∈[1,8] Δdf −60%, em +3.85pp; InfoVQA gt∈[1,8] Δdf −24%, em +1.09pp. Validates cross-dataset transferability of subspace projection method on the new datasets at small-gt subset. Phase 1 will determine whether **PlotQA+InfoVQA-calibrated** subspace at full gt range achieves the same. | ✅ pilot done |

### 6.6 §8 — Future work (scope only)

| ID | Direction | Status |
|---|---|---|
| **F1 (preferred)** | LLM/VLM architectural diff — same anchor delivered as text to LLM vs. as image to VLM, compare layer-wise integration profile (§7-style attention) | ✅ ideation paragraph drafted — `docs/insights/paper-section-8-f1-future-work.md` (2026-04-29) |
| **F2** | image-vs-text anchor — anchor image described as text and given to the same VLM; effect-size delta | ☐ ideation only |
| **F3** | Reasoning-mode VLM at scale — Qwen3-VL thinking, etc., on E5e cross-dataset matrix | ☐ scope only (γ-β is the minimal §8 stake) |

## 7. Pending work — Phase-structured priority queue (2026-05-02 restructure)

Phase 1 = paper consistency push (Main matrix at canonical setup). Phase 2 =
breadth-strengthening (digit-pixel causality 5-strat × 5 datasets). Phase 3 =
mechanism-Main alignment (E1-patch + E4 + Main). Within phase, P0 blocks the
phase target, P1 is opportunistic.

**As of 2026-05-02** — paper architecture restructure committed. All
P0s for the OLD 4-dataset paper structure have landed (E6 4-dataset
headline at gt∈[0,8], paper §3/§7.4/§7.4.5/§8 prose, M2/C-form
refactor, L1-L4 confidence). The work below operationalises the new
5-dataset main matrix + Main-model-led architecture.

### Phase 1 — paper consistency push (target: ARR-clean main matrix)

| Pri | Task | Where | Estimate |
|---|---|---|---|
| **P0** | `experiment_e7_plotqa_full` 3-model run (Main + Sub-A + Sub-B) at n=2500 stratified — config exists at `configs/experiment_e7_plotqa_full.yaml` | §6.3 §5 / §3 | ~9h wall (3 GPU parallel) |
| **P0** | `experiment_e7_infographicvqa_full` 3-model run at n=1147 (full numeric) — config exists at `configs/experiment_e7_infographicvqa_full.yaml` | §6.3 §5 / §3 | ~4h wall |
| **P0** | E6 Subspace recalibration: pool wrong-base sids from PlotQA + InfoVQA Main baselines, recompute subspace at L=31, evaluate at full gt range across 5 datasets | §6.5 E6 | ~3h calibration + ~10h sweep wall |
| **P0** | Section §3.3 + §5 + §6 reaggregation against new 5-dataset main matrix (drop VQAv2 from headline tables, fold appendix mention) | §6.1 / §6.3 / §6.4 | analysis only, ~1d |
| **P1** | §3 main panel (3-model × 5-dataset) consolidated table — compose §3.3 from existing 3-model E5e cells (TallyQA + ChartQA + MathVista) + Phase 1 new (PlotQA + InfoVQA) | §6.1 | reaggregation only |
| **P1** | gemma3-27b-it on E5e TallyQA at n≥1000 (currently n=300 max_samples) for consistency with main panel | §6.3 | ~3h compute |

### Phase 2 — breadth strengthening (Phase 1 results in hand)

| Pri | Task | Where | Estimate |
|---|---|---|---|
| **P1** | E5b/c (5-stratum + digit-mask) on PlotQA + InfoVQA × 3-model panel | §6.3 §5 | ~10h wall |
| **P1** | E5b/c on ChartQA + MathVista × 3-model panel (currently absent) | §6.3 §5 | ~10h wall |
| **P1** | §6 confidence-modulated reaggregation across new 5-dataset matrix | §6.4 | reaggregation only |

### Phase 3 — mechanism-Main alignment (E1-patch / E4 / Main)

| Pri | Task | Where | Estimate |
|---|---|---|---|
| **P1** | E1-patch attention extraction on Main `llava-interleave-7b` (perfect-square SigLIP, 27×27) — joins 4-model panel → 5-model | §6.5 §7.1–7.3 | ~30 min H200 + bbox JSON regen |
| **P1** | E1-patch causal ablation refactor: digit-bbox region zero-mask only (replaces full-anchor E1d as §7.3 main); 4-cond config required | §6.5 §7.1–7.3 | 4-cond wiring + ~1h GPU/model on 5-model panel |
| **P1** | E1-patch masked-arm causal control on 5-model panel (4-cond extraction) | §6.5 §7 | ~5h GPU |
| **P1** | E4 attention re-weighting on Main `llava-interleave-7b` (Phase 1 sweep + Phase 2 full) | §6.5 §7.4 | ~6–8h depending on archetype assignment |

### Phase 4 — paper polish (write phase)

| Pri | Task | Where | Estimate |
|---|---|---|---|
| **P1** | §7.4.5 paper prose update (Tally-cal headline → PlotQA+InfoVQA-cal headline at full gt range) | `docs/paper/sections/07_*.md` | text only |
| **P1** | §3 / §5 / §6 paper prose update for 5-dataset matrix | `docs/paper/sections/0[3-6]_*.md` | text only |
| **P1** | Citation verification — every 2026 arXiv ID in `references/project.md` and §2 paper draft must resolve to a real paper | §9 caveat | hours of manual verification, reviewer-defuse |
| **P3** | Image-vs-text anchor (F2) follow-up paper | §6.6 | future |
| **P3** | InternVL3 + Qwen2.5-VL E1-patch non-square (appendix only) | §6.5 §7 | 1–2 days/model implementation if pursued |

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
- **In-flight inference (2026-05-02 restructure).** Phase 1 P0 runs not
  yet started — `experiment_e7_plotqa_full` 3-model n=2500 + `experiment_e7_infographicvqa_full` 3-model n=1147 +
  E6 PlotQA+InfoVQA-pooled recalibration. See `references/plan_phase1.md`
  for the runbook.
- **gt ∈ [0,8] restriction retired (2026-05-02).** Old §7.4.5 headline used
  this restriction because Tally-only calibration's gt distribution capped
  there. Phase 1 retargets at full gt range with PlotQA+InfoVQA pooled
  calibration. If full-range fails, plan B = pooled multi-source
  (count + chart + info) calibration.
- **Heavy-tailed prediction distance (2026-05-02).** PlotQA model emits raw
  chart values (e.g., "10000000" for gt=22) on a small minority of samples.
  `mean_distance_to_anchor` was inflated by single outliers (PlotQA S3 mean
  59,707 vs median 75). Switched to `median_distance_to_anchor` in live
  metrics; legacy `mean_distance_to_anchor` field preserved in older
  `predictions.jsonl` only.

## 10. Changelog

- **2026-05-02 02:30 (Phase 1 P0 in flight on GPU 1).** Branch
  `phase1/p0-baseline-recalibration`, ~14 commits.

  **Per-dataset n cap raised to 5000** (user revision from initial 2500
  PlotQA target):
  - PlotQA snapshot refetched at `--max-samples 5000 --stratified` (5 gt
    bins × 1000); config `max_samples` 2500 → 5000.
  - TallyQA config now `max_samples=5000 + samples_per_answer=700`
    (effective n=5000, balanced across gt 0-8). Existing 38k full-set
    runs archived to `outputs/_legacy_tallyqa_n38k/`.
  - InfoVQA / ChartQA / MathVista data-bound below 5000, no change.

  **§7.4.5 sweep cap raised 500 → 5000 wrong-base** based on evidence
  that wrong-base direction-follow rate is **3-5× stronger than
  correct-base** on existing chartqa/mathvista/tallyqa per_cell data.
  Larger wrong-base sweep cell preserves the signal that mitigation
  targets while lifting statistical power.

  **Multi-token confidence proxy fix (Phase 1.5)**: discovered Qwen2 /
  Llava-Interleave tokenizers split each digit into a separate token,
  so 75-89 % of multi-digit predictions on PlotQA / InfoVQA / ChartQA
  had `answer_token_logit = None` (single-token text matching failed).
  Wrote `scripts/recompute_answer_span_confidence.py` to identify the
  answer span via forward-walk digit accumulation (handles bare digits,
  JSON wrappers, decimals with `.` interior, word-numbers) and emit 14
  span-aggregate proxy fields (logit / prob / log-prob / entropy
  variants) plus `plotqa_relaxed_correct` (5%) + `infovqa_anls` (≥0.5)
  supplementary correctness columns. Smoke-tested on chartqa+mathvista:
  100% span coverage. `analyze_confidence_anchoring.py` refactored to
  multi-proxy registry — emits `L1_proxy_comparison.csv` ranking
  proxies by mean df(Q4-Q1) gap so paper §6 primary proxy can be
  chosen post-hoc from data.

  **Strict + relaxed/ANLS dual reporting**: §3 main panel headline
  table (`build_e5e_e7_5dataset_summary.py`) now emits both strict
  exact_match (canonical, 5-dataset comparable, anchor-metric aligned)
  AND PlotQA-relaxed-5% / InfoVQA-ANLS≥0.5 (each dataset's official
  metric) for acc(b) base accuracy. ChartQA llava strict 33.9% /
  relaxed 44.8% — 11pp gap matches Methani 2020 numeric-VQA conventions.

  **Pollution fixes**: `discover_inputs` (analyze_confidence_anchoring)
  and `discover_runs` (analyze_e5e_wrong_correct) now pick the largest
  predictions.jsonl per (exp, model) — matches the canonical
  `phase_a_data_mining._resolve_model_runs` rule and prevents pilot
  (n=200) runs from polluting full-run aggregates.

  **Watcher**: `scripts/_phase1_watcher.sh` — 10-min poll detects
  COMPLETE / ZOMBIE / STALLED transitions (zombie = process gone w/o
  completion marker; stall = log untouched > 15 min). Hourly heartbeat
  during normal alive operation. `scripts/_phase1_post_baseline.sh`
  orchestrates §1.2 → §1.3 → §1.4 once §1.1 finishes (idempotent).

  **Status**: §1.1 baseline running on GPU 1 since 01:45. Llava plotqa
  85% at 02:20. ETA ~12h-14:00 for full Phase 1 P0 (3 datasets × 3
  models baseline + calibration + sweep + reaggregation).

- **2026-05-02 (Phase 1 kickoff — paper architecture restructure).**
  Coordinated paper-wide model + dataset consistency push.

  **Decisions finalised** (`references/project.md §0.4` is canonical):
  - **Main model** = `llava-interleave-7b` (every section primary).
    Rationale: E6 mitigation backbone, multi-image native (LLaVA-Interleave
    designed for interleaved multi-image), modern 2024 release with stable
    HF integration, 7B sweet spot, established 2025-2026 strong VLM baseline.
  - **Sub panel** = + `qwen2.5-vl-7b-instruct` + `gemma3-27b-it`.
    qwen2.5 (not qwen3) for disjointness from §5 γ-β qwen3 reasoning
    ablation + 2025-2026 standard baseline status. gemma3-27b-it (not 4B)
    because 4B base accuracy is too weak for clean wrong/correct split
    measurement on numeric VQA. gemma4 (not 3) avoided due to less HF
    integration maturity at 2026-05.
  - **Mechanism panel** restricted to **5-model perfect-square**:
    gemma4-e4b + llava-1.5-7b + convllava-7b + fastvlm-7b + Main.
    InternVL3 + Qwen2.5-VL → appendix only (non-perfect-square implementation
    correctness risk per-encoder).
  - **5-dataset main matrix**: TallyQA + ChartQA + MathVista + **PlotQA** +
    **InfographicVQA**. **VQAv2 dropped** from main matrix (multi-GT, legacy
    benchmark, full-eval impractical). Existing 7-model VQAv2 panel preserved
    in appendix for behavioural breadth supplementary.
  - **§7.4.5 mitigation**: recalibrate Subspace on **PlotQA + InfoVQA pooled**
    at full gt range, drop the [0,8] restriction. Goal: replace "deployable
    across 4 datasets at gt ∈ [0,8]" with "deployable across 5 datasets at
    full gt range".

  **Pilot evidence (2026-05-01 / 2026-05-02 on llava-interleave-7b, n=200)**:
  - PlotQA: baseline df 0.327, a−m gap +14.6pp (strong digit-pixel signal).
  - InfoVQA: baseline df 0.190, a−m gap +5.1pp.
  - Existing Tally-calibrated Subspace L31_K04_α=1.0 transferred at gt∈[1,8]
    on PlotQA Δdf −60% / em +3.85pp; InfoVQA Δdf −24% / em +1.09pp.
    (Demonstrates method transferability; Phase 1 will determine whether
    PlotQA+InfoVQA-calibrated subspace works at full gt range.)

  **Code/scaffolding landed 2026-05-01–02**:
  - `scripts/fetch_plotqa_test.py` — V1 numeric, template ∈ {data_retrieval,
    min_max, arithmetic}, gt ≤ 10000 filter
  - `scripts/fetch_infographicvqa_val.py` — positive int, gt ≤ 10000 filter
  - `configs/experiment_e7_plotqa_pilot.yaml` + `experiment_e7_plotqa_full.yaml`
  - `configs/experiment_e7_infographicvqa_pilot.yaml` + `experiment_e7_infographicvqa_full.yaml`
  - `inputs/plotqa_test/` + `inputs/infographicvqa_val/` snapshots
  - `metrics.py` + `visualization.py`: `mean_distance_to_anchor` →
    `median_distance_to_anchor` (heavy-tailed prediction outliers on
    wide-gt datasets — single PlotQA outlier inflated S3 mean to 59,707
    while median was 75)
  - Detailed Phase 1 runbook at `references/plan_phase1.md`

- **2026-05-01 (E6 FINAL: Subspace L31_K04_α=1.0 cross-dataset paper headline confirmed).**
  Tally-only N=5000 calibration rerun (master_v2 pipeline) ran S0→S3 over ~17h
  on GPU 1 + post-hoc gt-bin breakdown. Final 4-dataset matrix on gt ∈ [0,8]
  intersection (TallyQA / ChartQA / VQAv2 / MathVista):

  | Cell | Tally | ChartQA | VQAv2 | MathVista |
  |---|---|---|---|---|
  | L31_K04_a0.5 | df −41% em −0.5 | df −8% em +2.1 | df −58% em +2.0 | df −52% em +1.0 |
  | **L31_K04_a1.0** ⭐ | df −55% em +1.6 | df −46% em +3.3 | df −52% em +0.9 | df −56% em +2.0 |
  | L31_K04_a2.0 | df −64% em +1.3 | df −33% em +2.3 | df −60% em −0.8 | df −63% em +3.3 |
  | L24_K04_a1.0 | df −39% em −0.9 | df −23% em +3.0 | df −46% em +2.2 | df −43% em −1.3 |

  All 4 cells pass on every dataset; em changes uniformly within ±3.3 pp
  (deal-breaker −6 pp cleared by wide margin). Selected **L31_K04_α=1.0**
  as paper headline cell — most uniform cross-dataset reduction (−46 % to
  −56 %), all em changes positive (+0.9 to +3.3 pp).

  Methods 0a/0b/1-pooled/2/3/4a/4c all rejected:
  - Single-direction (0a/0b): cross-dataset fails (cos(v_tally,v_chartqa) ≈ 0.5)
  - LEACE 4c: ChartQA backfires across all gt bins (rank-1 limit, same as 0a/0b)
  - QAO 2 / CogBias 4a: ChartQA below threshold at full set
  - DPO 3 mix_synthetic: anchor adoption → 0 universally (mitigation transfers)
    BUT em −5.85pp on VQAv2 borderline; gt-distribution training bias
    confounds ChartQA full-range measurement (see gt-bin breakdown commit
    2eb665e for details).

  Caveat documented: gt ∈ [0,8] subset is the natural intersection of
  TallyQA training distribution (96% gt ≤ 4, 100% gt ≤ 8) and the four
  evaluation distributions. Outside [0,8] (ChartQA full / MathVista
  large-gt), measurement is confounded by gt-distribution shift —
  calibration data limitation, not method failure. To extend coverage,
  calibrate on ChartQA + MathVista training data with wider gt range
  (out-of-scope this paper).

  Paper §7.4.5 prose written at `docs/paper/sections/07_mechanism_mitigation.md`
  (132 lines). §7.5 free-lunch extended to cover both E4 and E6. §7.6 summary
  updated. §1 abstract + §1.4 mitigation + §1.6 contributions all updated to
  reflect E6 deployable cross-dataset result.

  Branch: `e6-tally-only-rerun`. ~17h GPU 1 (S0 calibration + S1 LEACE + S2
  Subspace + S3 DPO mix_synthetic + 1h supplementary VQAv2/MathVista).
  Tracker doc: `docs/experiments/E6-tally-only-rerun-tracker.md` (full
  intermediate-results log).

- **2026-04-30 (E6 Method 4c LEACE verdict revised ⚠ → ✅ TENTATIVE; Subspace n=500 Tally landed; ChartQA cancelled).**
  Re-analysis under one-sided em rule (`em_pp ≥ baseline − 2pp`; allow em gains as
  intended mitigation effect, only filter em drops): **L30_a2.0 passes both Tally and
  ChartQA at n=100** — Tally df −13.2 % rel with em **+5.88 pp** (improvement), ChartQA df
  **−38.1 % rel** with em invariant. Cross-dataset overlap = 1 cell. The original
  two-sided rule had rejected this cell because em rose on Tally, treating em gains as
  suspicious. Method 4c becomes the only multi-method-search candidate to survive the
  cross-dataset selection rule (under one-sided em). Caveat: n=100 baselines were
  selection-biased on prior CogBias case; LEACE n=100 likely has the same issue → full-set
  validation required to confirm. `analyze_e6_methods.py` extended with `--em-rule` flag
  (two_sided / one_sided); existing LEACE + CogBias sweeps re-analyzed in place.
  
  Subspace Method 1 n=500 Tally completed (162,000 records, 365 min wall, GPU 1):
  baseline df=0.1328 / em=0.1423; under two-sided rule 7/80 cells pass with best
  L31_K04_a2.0 (df −45.2 %, em −1.19 pp); under one-sided rule 32/80 pass. Selection bias
  partially supported (n=100 11/81 → n=500 7/80 two-sided) but best-cell strength
  preserved. ChartQA n=500 cancelled to free GPU 1 for Tally-only LEACE recalibration —
  Method 1 deprioritised vs. LEACE-one-sided result. Documented in
  `docs/experiments/E6-steering-vector.md` (Method 4c section + Method 1 n=500 subsection).

- **2026-04-30 (E6 Method 4a CogBias ❌ FAILED — full-set validation + Subspace n=500 rerun launched).**
  CogBias full-set validation (Tally n=500, ChartQA n=416): L31_ap0.5_ad0.5 achieves Tally −6.2%
  (60/498 vs 64/498, Z=0.38 SE — below significance) but ChartQA only −4.3% (88/407 vs 92/407,
  below −5% threshold). Cross-dataset overlap = 0. n=100 Tally baseline was inflated by selection
  bias (14.0% apparent vs 12.85% true population baseline). CogBias ❌ FAILED. Also discovered
  that all n=100 method screens had elevated baselines due to selection bias — launched Subspace
  (Method 1) n=500 full 81-cell grid rerun (Tally ~6h, ChartQA ~5h, GPU 1, /tmp/e6_subspace_n500.log).

- **2026-04-30 (E6 Methods 4a CogBias inconclusive + Method 3 DPO ❌ FAILED — all methods complete).**
  CogBias sweep completed both datasets. TallyQA 8/60 pass (all L31, ap∈{0.5,1.0}, df −7.1%, em −1.0pp).
  ChartQA 14/60 pass (best L30_ap0.5_ad0.5: df −16.7%, em −1.0pp). Cross-dataset: 1 cell nominally passes
  (L31_ap0.5_ad0.5: T −7.1%, C −8.3%), but both effects are < 0.5 SE at n=100 (1–2 sample differences =
  sampling noise). alpha_decode has no measurable effect at this scale.
  DPO (Method 3): sweep adapter fixed (template KeyError + multimodal content format). Tally: df −42%,
  em +15pp (massively better for counting but 97% Tally training data). ChartQA: df −3.9%, em −3.7pp,
  parse failure rate spikes (73/300 valid predictions). → DPO ❌ FAILED cross-dataset.
  All 6 methods (0a/0b/1/2/4c/3) now confirmed ❌. CogBias (4a) requires full-set validation
  (n=346/416, ~10 min) to determine if the 1 nominally-passing cell survives at higher n.
  Root cause remains: anchor representation differs between Tally and ChartQA (cos ≈ 0.47–0.62).

- **2026-04-30 (E6 Method 4c LEACE ❌ FAILED; Methods 4a+3 still in flight).**
  LEACE sweep completed on both datasets. TallyQA n=100: 0/20 cells pass
  (best L30_a2.0: df −13.2% but em +5.88pp, exceeds ±2pp tolerance). ChartQA n=100:
  5/20 cells pass (best L30_a2.0: df −38.1%, em +0.00pp). Cross-dataset overlap = 0
  → Method 4c ❌ FAILED. Inverse failure mode to Methods 0–2: LEACE works on
  ChartQA but conflicts with Tally. Same root cause: direction-mismatch between
  datasets (cos(T,C) ≈ 0.47–0.62 at key layers). Method 4a (CogBias) sweep running
  on GPU 1 (cell ~24/61, ETA ~40 min for Tally + ~70 min for ChartQA).
  Method 3 (DPO) training at ~50% (step ~310/625, ETA ~45 min).
  E6 doc + roadmap §7 updated to reflect LEACE ❌.

- **2026-04-30 (E6 Methods 4c+4a+3 in-flight; Methods 1+2 ❌ FAILED).**
  Method 2 (AFTER QAO) full-set validation completed: Tally n=346 → 1/4 cells
  pass (Lq30_Lt28_a0.5, df Δ=−9.6%, em Δ=+0.29pp); ChartQA n=416 → 0/4 cells
  pass (best Δ=−4.1%, below −5% threshold). No cross-dataset overlap → Method 2
  ❌ FAILED. Root cause: PCA-Ridge probe overfits Tally query distribution;
  correction direction conflicts with ChartQA. User-approved escalation to
  Methods 3+4c+4a regardless of Method 2 outcome. Now running in parallel:
  Methods 4c (LEACE, `scripts/e6_leace.py`), 4a (CogBias decode-correction,
  `scripts/e6_cogbias.py`), 3 (MIA-DPO LoRA, `scripts/e6_dpo_lora.py`).
  New scripts: `scripts/analyze_e6_methods.py` (unified M2/C-form analysis for 4c+4a).
  LEACE calibration complete (CPU, 128.5s; P_stack [32, 4096, 4096]).
  DPO pairs built: 34326 across TallyQA+ChartQA+VQAv2.
  Sweep pipeline at `/tmp/e6_m4c_m4a_pipeline.sh`.

- **2026-04-30 (E6 Method 1 ❌ FAILED; Method 2 QAO in flight).**
  Method 1 (multi-direction subspace projection) completed: TallyQA ✅ 11/81
  cells pass, ChartQA ✅ 3/81 cells pass, but cross-dataset overlap = 0 — no
  cell satisfies the selection rule on both simultaneously. Pre-M2 diagnostics
  run on the data: (1) only 5/81 cells reduce df on BOTH datasets; the near-miss
  L31_K08_a4.0 is blocked by Tally em −3.94pp with no feasible alpha rescuing
  it; (2) cos(T,C) = 0.47–0.62 at key layers (NOT 0.98 — correction of a prior
  assumption); (3) d'(between-mean) = 1.7–3.6 confirms datasets are linearly
  separable in residual space → supports per-input adaptation as the right
  remedy. Method 2 (AFTER QAO, arXiv:2601.01957) implemented and pipeline
  launched on GPU 1: `scripts/e6_query_adaptive_offset.py`
  (calibrate-qao/train-probe/smoke-qao/sweep-qao) + `scripts/analyze_e6_qao.py`.
  Probe: PCA-100 + Ridge (λ=1e3) on pooled VQA+Tally+ChartQA calibration
  (N≈1200 wrong-base pairs). Sweep grid: 4×3×4 = 48 steered cells, Tally n=100
  then ChartQA n=100. Results pending; will append to
  `docs/experiments/E6-steering-vector.md`.

- **2026-04-30 (E6 Method 1 implementation — multi-direction subspace
  projection in flight).** Implemented three new phases in
  `scripts/e6_steering_vector.py`: `calibrate-subspace` (collects
  per-pair D matrices (n_pairs, n_layers, d_model) for SVD instead of
  just mean v.pt), `smoke-subspace` (10-pair projection hook wiring
  check), `sweep-subspace` (61-cell L×K×α sweep with h←h−αV_K^TV_Kh
  projection). Added `scripts/e6_compute_subspace.py` (pooled SVD per
  layer, saves top-K right singular vectors) and
  `scripts/analyze_e6_subspace.py` (M2 metrics per cell, 5 % rel-df
  selection threshold). Key fix: wrong-base sids prioritised in
  calibration loop so D_wrong gets max coverage before hitting
  max_calibrate_pairs cap. Pipeline running on GPU 0 (CUDA_VISIBLE_DEVICES=0):
  calibrate TallyQA → ChartQA → VQAv2 → pooled SVD → smoke → sweep
  TallyQA n=100 → sweep ChartQA n=100. Branch:
  `e6-method1-subspace-projection`. Results pending; will append to
  `docs/experiments/E6-steering-vector.md`.

- **2026-04-29 (E6 mitigation-search frame — single-direction
  failed, pivoting to multi-method search across multiple sessions).**
  After Phase 1 PoC landed VQAv2-only df −14.2 % rel, cross-dataset
  testing on TallyQA + ChartQA showed **the VQAv2-calibrated v
  backfires** (+5.5 % / +1.3-4.5 %). User-driven reverse-direction
  calibration (TallyQA-cal, ChartQA-cal) confirmed the failure is
  structural — `cos(v_VQA, v_tally) = 0.98` at L=30 yet behaviour
  differs by target dataset; TallyQA-cal v even **backfires on its
  own dataset at α=1** (+4.3 %). Single-direction subtraction is
  hitting the n≈350 / d=4096 SNR floor.

  Deep paper-search-mcp survey (14 axes, 22 methods on arXiv +
  Semantic Scholar) produced a 3-method pivot plan in the canonical
  plan file at
  `~/.claude/plans/task-notification-task-id-bugsfzyep-tas-lively-dongarra.md`:

  - Method 1 (PRIMARY): multi-direction subspace projection
    (CIPHER / VCE / RepE family) — top-K SVD basis, drop-in for the
    existing offset hook
  - Method 2 (FALLBACK 1): query-adaptive offset (AFTER QAO) — small
    probe estimates per-input correction
  - Method 3 (FALLBACK 2): MIA-DPO LoRA fine-tune — multi-image
    preference optimisation

  Plus 9 extras held in reserve (CogBias, Spherical Steering, LEACE,
  DSO, Dual Steering, PAI/AIR/Modality-Bias, VCD-family, CIPHER-exact,
  CAST). User policy update: **every new method tested first on
  TallyQA + ChartQA SUBSETS, only graduate to VQAv2 after cross-
  dataset proves out** (reverses prior VQAv2-first default;
  cross-dataset failure is the binding problem).

  Em invariance ✓ on every cross-test → deployable safety claim
  intact even though mitigation claim doesn't transfer for
  single-direction. Worst-case fallback: scope §7.4.5 down to
  "single-domain on VQAv2" and frame cross-dataset failure as the
  §7.4.5 empirical contribution itself.

  Status: **Method 1 implemented and in flight as of 2026-04-30**;
  three new phases added to `scripts/e6_steering_vector.py`
  (`calibrate-subspace`, `smoke-subspace`, `sweep-subspace`), two new
  scripts (`e6_compute_subspace.py`, `analyze_e6_subspace.py`);
  pipeline running on GPU 0 (branch `e6-method1-subspace-projection`).
  Sweep on TallyQA + ChartQA subsets (n=100 each) queued; results to
  land in `docs/experiments/E6-steering-vector.md` when complete.
  Memory `project_next_cleaner_mitigation.md` rewritten to point at
  the multi-method tracker.

- **2026-04-29 (E6 Phase 1 PoC ✅ — anchor-agnostic mitigation works).**
  Phase 0 (calibration `v.pt` extraction, 230 s wall, 32 layers ×
  4096 d, 399 wrong-base + 1000 all-base pairs) + Phase 0.5 (wiring
  smoke, 4/10 changed) + Phase 1 (n=200 stratified × 4 conditions × 42
  steered cells + baseline = 34,400 records, 6,455 s wall = 1.8 h on
  H200) all landed on `llava-next-interleaved-7b` from existing E5c
  VQAv2 (a, m) S1 pairs. **8 / 42 cells pass design-doc selection rule
  (df ≤ 0.9 · df₀, em_b/em_d/em_a ≥ baseline − 0.02, mdist guard).**

  **Chosen cell:** L30 / α=1.0 / v_wrong (smallest |α| tiebreaker
  among −14 %-class passers).

  | metric | baseline | E6 chosen | Δ |
  |---|---:|---:|---:|
  | df_a (C-form) | 0.1915 | **0.1643** | **−14.2 % rel** |
  | adopt_a | 0.1206 | 0.1214 | +0.08 pp |
  | em(target_only) | 0.585 | 0.580 | −0.5 pp ✓ |
  | em(neutral d) | 0.570 | 0.565 | −0.5 pp ✓ |
  | em(anchor a-S1) | 0.540 | 0.540 | invariant |
  | mdist | 1.105 | 1.095 | −0.01 (clean) |

  E4 (LLaVA-1.5-7b mid-stack-cluster panel, df −14.6 % rel) and E6
  (llava-next-interleaved-7b, this PoC, df −14.2 % rel) match in effect
  size at vastly different deployment cost: E4 needs anchor-token span
  at inference (research demo); E6 applies a fixed residual offset
  universally at L30 (deployable). **The §7.4.5 deployability gap is
  closed at PoC scale.** Full results writeup at
  `docs/experiments/E6-steering-vector.md` (governing experiment
  markdown; will be updated in-place as Phase 2 numbers land per user
  instruction 2026-04-29).

  Other Phase 1 findings: (i) `cos(v_wrong, v_all) ∈ [0.96, 0.99]`
  across all L — wrong-base filter mostly scales magnitude, doesn't
  change direction; Phase 1 confirms cells pair up similarly across
  the v-var axis. (ii) ‖v[L]‖ peaks at L=30 (near final layer of the
  32-layer Qwen-7B backbone), *not* at the E1b mid-stack L=16 of the
  CLIP-ViT panel — residual-stream encoding profile differs from
  attention encoding profile. (iii) `adopt_a` barely moves; df_a
  drops — mitigation operates on graded pull, not categorical
  adoption (matches §1 headline). (iv) L24/α=4 backfires at +14 % df
  — α-overshoot regime to flag. (v) L24/α=2/v_wrong achieves the
  largest df reduction (−16.7 %) but breaks em_b (−4.9 pp), exactly
  the failure mode the deployability check catches.

  Status changes: §6.5 E6 row flipped from `☐ design v2` to
  `✅ Phase 1 PoC`; §7 P1 entries updated to point at Phase 2a (full
  VQAv2) and Phase 2b (cross-dataset deployability), with Phase 2c
  (LLaVA-1.5-7b head-to-head port) as P3 optional.

- **2026-04-29 (E6 anchor-agnostic steering-vector mitigation —
  branch + design doc, post-review v2).** New mitigation track opened.
  Branch `e6-steering-vector-mitigation` cut from master at commit
  `6eee87f`. Design doc at
  `docs/experiments/E6-steering-vector-design.md` (commits `0471e52`
  initial, `0588ff6` first revision, current commit second revision)
  — motivation grounded in the **calibration-vs-inference label
  axis**: E4 (and the N1/N2 digit-bbox sharpenings brainstormed
  alongside E6) all need anchor-image labels at inference, making
  them research demonstrations rather than deployable interventions.
  E6 (ActAdd-class residual-stream offset, train-free) calibrates
  `v_{L*}` once from E5c S1 wrong-base (a, m) pairs and applies a
  fixed offset universally at inference — **zero anchor labels needed
  at deploy time**. Mechanistic justification: E1d's single-layer
  null is on the *attention pathway only*; residual-stream
  interventions are not refuted (Turner 2023 / Rimsky 2024 lineage).

  **PoC model = llava-next-interleaved-7b** (post-review v2 flip from
  original LLaVA-1.5-7b). Drives: §3.3 main-panel coherence,
  pre-existing E5c VQAv2 calibration data on disk (399 wrong-base S1
  pairs verified), free Phase 2b cross-dataset deployability check
  (E5c TallyQA + E5e ChartQA + E5e MathVista all on same model). E4
  same-model head-to-head dropped from critical path; if reviewers
  ask, Phase 2c ports to LLaVA-1.5-7b cheaply (~1 day).

  PoC plan: Phase 0 calibration extraction (~15–20 min H200, leverages
  existing E5c sids) → Phase 0.5 wiring smoke (10-pair, ~10 min) →
  Phase 1 (L × α × v-var) sweep on n=200 stratified (~5–7 h H200) →
  Phase 2a full VQAv2 (~20–40 h H200) + Phase 2b cross-dataset
  (~5–10 h × 3 datasets) only if PoC clears the selection rule
  (df ↓ ≥ 10 % rel, em(b) / em(d) invariant ± 2 pp, fluency guard).
  E4 not retracted — kept as §7.4 mechanism story; E6 added as §7.4.5
  deployability story. N1/N2 reframed as mechanistic analysis tools
  under §7.2. Roadmap §6.5 (new E6 row), §7 (new P1 entry) updated.

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
