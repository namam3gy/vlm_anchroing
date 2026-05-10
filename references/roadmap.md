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
| **H4** | Reasoning / thinking-mode reduces anchoring | thinking-on `df` ≤ thinking-off `df` | ❌ Falsified, *amplification*. E5e γ-β on Qwen3-VL-8B Instruct vs Thinking on MathVista (`E5e-mathvista-reasoning-evidence.md`): ×2.9 df all-base, ×12.7 df on correct-base. H2 wrong > correct asymmetry collapses in thinking mode. Same direction as VLMBias / Wang LRM-judging. |
| **H5** | "No-hedging" prompt amplifies anchor pull on uncertain items | `direction_follow` increases under strengthen | ⚠ Suggestive (gemma3-27b-it strengthen `mean_distance_to_anchor` = 2617 → hallucination, not anchor pull). Folded into §"strengthen anomaly" caveat |
| **H6** | Cross-modal failures decouple into two orthogonal axes — `anchor-pull` vs. `multi-image distraction` | `adopt_rate(a)` and `acc_drop_d_vs_b` perfectly correlated → H6 fails | ✅ 6-model main matrix (5 datasets): gemma3 family = anchoring corner (high adopt, no distraction); llava-onevision-7b + qwen2.5-vl-{7b,32b} = distraction corner; llava-interleave-7b mixed. See `docs/insights/_data/H6_2axis_per_model.csv` and `docs/figures/H6_2axis_scatter_5dataset.png`. |
| **H7** ⚙ | `direction_follow_rate` is monotonic with `pred_b`-token logit / probability — i.e. uncertainty modulates anchor pull on a **continuous** confidence scale, of which wrong/correct (H2) is a coarse projection | `direction_follow_rate` flat across confidence quartiles | ✅ Confirmed for non-reasoning panel — `L1-confidence-modulation-evidence.md` reports `entropy_top_k` Q4 − Q1 mean df = +0.152 on E5b/E5c/E5e, 23/35 anchor cells fully monotone. **Boundary case**: H7 monotonicity *collapses* under reasoning mode (`E5e-mathvista-reasoning-evidence.md` §3.1) — deserves §6 prose paragraph distinguishing "uncertainty-modulated graded pull" from "reasoning-induced graded pull". |

## 3. Status snapshot — where we are (2026-05-09 — 5-round paper review loop complete)

### 3.0b Post-review state (2026-05-09 — Solid Findings, top of band; Main contingent on bridge experiment)

5-round iterative review-revise loop on `docs/paper/emnlp_draft_ko.md`
shipped via PR-equivalent merge into master (commit `86fb66a`,
phase2 branch tip `8ffdc2d`). Methodology / writing / novelty /
adversarial / bar-raiser personas + author-reviser between rounds.
Paper 516 → 604 lines net (+88).

**P1 close-outs (2026-05-10, branch `worktree-paper+p1-defense-r4`):**
P1-3 paired-bootstrap CI on §6.2.3 Table 6 (B = 10,000) and P1-6 §A.5
27-cell pilot grid 4-metric heatmap aggregation both shipped — closing
R4 MAJ-4, R4 MAJ-6 (Bonferroni-20), and R4 CRIT-2 (cherry-pick concern).
Δem(b) emerges as the paper's multiplicity-robust headline (5/5
sign-clean under both 95 % and Bonferroni-20 corrected CIs); Δdf(a) is
sample-size-bound to PlotQA n=2,306 as the only CI-strong cell. 27-cell
heatmap shows the em-deal-breaker rule was non-binding (no cell rejected)
and the chosen cell ranks first by combined |Δdf(a)| under the same ex
ante rule, addressing reviewer cherry-pick concerns.

- **Final tier verdict (3-way convergent):** Solid Findings, top of band.
  Borderline / weak-accept Main contingent on §8.4 item 1 (γ-β
  residual-stream bridge experiment) landing positive in next revision.
- **Highest-leverage next action:** P0-1 — γ-β Thinking-mode trace
  amplitude on K=8 subspace at L=26 (cheap form ~2 H100-day; clean
  form +~4 H100-day). Bar-raiser explicitly named this as the
  experiment that would tip Findings → Main.
- **Sharpest 5-year-citable finding loop converged on:** §5.2 → §6.4
  predict-then-verify chain ("multi-layer redundancy predicts
  single-direction mitigation failure cross-dataset"); now framed
  as the paper's *이론적* contribution at 4 callsites (abstract /
  §1.3 / §1.5 (4a) / §8.1).
- **17 DEFER items** consolidated in
  [`docs/insights/plan_post_review_2026-05-09.md`](../docs/insights/plan_post_review_2026-05-09.md)
  with priority + estimate + dependency + acceptance criteria.
  P0 (2 items) for tier-shift; P1 (4 items) for adversarial-defense
  rigor; P2 (1 item) for cross-architecture verification; P3
  (3 items) for camera-ready hardening; P4 (7 items) for future
  submissions.
- **Review trail tracked.** Pipeline (`.claude/agents/paper-reviewer-*.md`,
  `.claude/agents/paper-reviser.md`, `.claude/commands/paper-review-loop.md`)
  and 11 review/response/summary files at `docs/paper/reviews/*.md`
  committed for reproducibility (selective `.gitignore` overrides).

### 3.0a Phase 1 P0 v3 final state (2026-05-04)

Phase 1 P0 v3 substantively complete. See §10 changelog 2026-05-04 entry
for the full commit chain. Headlines:

- **Branch + master pushed** to origin (`phase1/p0-baseline-recalibration`).
- **6-model main matrix (post-InternVL3 removal 2026-05-10; was 7-model with InternVL3) × 5-dataset** at `docs/insights/_data/main_panel_5dataset_summary.md`. Models: llava-onevision-7b (Main), qwen2.5-vl-7b, gemma3-4b, qwen2.5-vl-32b, gemma3-27b, llava-interleave-7b.
- **Stage 4-final mitigation** (Phase B, commit `9f9dfa0`): cell L=26 K=8 α=1.0 ships. **Free-lunch**: avg Δdf = -2.9pp, **Δem(a) = +3.9pp benefit, Δem(b) = +8.8pp recovery** on wrong-base sids — paired-sids generator `scripts/build_e6_stage4_summary.py` → `docs/insights/_data/stage4_final_per_dataset.csv`. Earlier "em(a) -2.4pp cost" framing was a hand-copy error, corrected 2026-05-04 → paper §7.4 task #38.
- **Phase D §7.1-7.3** (commit `c556fb6`): 24/24 cells on disk. New finding via `scripts/analyze_cross_dataset_peaks.py` — OneVision peak layer is **dataset-dependent** (L=27 on Plot/Tally, L=14 on Info/VQAv2).
- **Phase E E1d causal ablation** (commit `2d11876`): OneVision × {Tally, Info, Chart, Math} 4/4. ChartQA + MathVista re-ran with per-dataset susceptibility CSVs after PlotQA-CSV-reuse bug.
- **llava-next-interleaved-7b dropped** from main panel (commit `0e7998e`) — low native resolution, not informative for chart/figure datasets.
- **E5b 5-stratum cross-dataset extension** (Phase 2, 2026-05-04 10:31): OneVision Main on 4 datasets (MathVista, ChartQA, InfoVQA, PlotQA), 12-cond × 4 = 85,258 records. adopt monotonic decay S1→S5 on every dataset; df gentle decay -0.04 to -0.06 on 3/4 (MathVista flat); em stable. Plausibility-window claim replicates at full GT range on a second architecture (the original E5b was llava-interleave on GT≤8). Doc: `docs/insights/E5b-cross-dataset-onevision.md`, generator: `scripts/build_e5b_5strat_decay_summary.py` → `docs/insights/_data/e5b_5strat_decay_per_dataset.{csv,md}`.

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

**Phase 1 target rows** (now mostly ✅ shipped — see §3.0a top of section):

| Experiment | Dataset | Conditions | Models | Status |
|---|---|---|---|---|
| `experiment_e7_plotqa_full` | PlotQA test V1 | b/a/m/d (S1) | 6-model panel (post-InternVL3 removal 2026-05-10) | ✅ shipped 2026-05-03; standalone evidence doc `docs/insights/E7-plotqa-infovqa-evidence.md` 2026-05-04 — 6-model wrong-base df ranking, gemma3 anti-scaling 4B (0.395) > 27B (0.227), 5/6 models show em(a) > em(b) free-lunch baseline |
| `experiment_e7_infographicvqa_full` | InfographicVQA val | b/a/m/d (S1) | 6-model panel (post-InternVL3 removal 2026-05-10) | ✅ shipped 2026-05-03; covered jointly in `E7-plotqa-infovqa-evidence.md` — gemma3 anti-scaling reverses (4B 0.324 < 27B 0.350), free-lunch claim doesn't generalise here (mixed em deltas) |
| E5b/c PlotQA + InfoVQA + ChartQA + MathVista 3-model | 4 datasets | b + 5×a-strat + 5×m-strat + d | same 3 | 🟡 Phase 2 — OneVision Main shipped 2026-05-04 (4/4 datasets, 12-cond × 4 = 85,258 records). adopt monotonic decay S1→S5 on every dataset; df gentle decay (-0.04 to -0.06) on 3/4 (MathVista flat); em stable. Plausibility-window claim replicates at full GT range on a second architecture. Doc: `docs/insights/E5b-cross-dataset-onevision.md`, generator: `scripts/build_e5b_5strat_decay_summary.py`. 3-model expansion still pending (defer; OneVision is the §5 headline) |
| E6 Subspace recalibration on PlotQA + InfoVQA pooled, 5-dataset full-range eval | 5 datasets | b/a/m/d (S1) + cell sweep | Main only | ✅ shipped — chosen L=26 K=8 α=1.0; Stage 4-final eval done. **Bonus em(b) +9.2pp** finding for paper §7.4 task #38 |
| Phase D §7.1-7.3 cross-dataset attention | 4 datasets (Tally/Plot/Info/VQAv2) | b/a/m/d (S1) | 5-panel + OneVision = 6 models | ✅ shipped 2026-05-03 — 24/24 cells; cross-dataset peaks via `analyze_cross_dataset_peaks.py` |
| Phase E E1d causal ablation OneVision × {Tally, Info, Chart, Math} | 4 datasets | sweep × 6 modes | OneVision Main | ✅ shipped 2026-05-03 + 2026-05-04 chart/math recovery |

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
| MathVista (γ-β) reasoning-mode | MathVista | b/a/m/d (S1) | qwen3-vl-8b-instruct + qwen3-vl-8b-thinking | ✅ landed 2026-04-28; full evidence doc `docs/insights/E5e-mathvista-reasoning-evidence.md` 2026-05-04 — all-base ratio ×1.6 adopt / ×2.9 df hides a much stronger correct-base ×12.7 df amplification (instruct df_correct=0.021 → thinking df_correct=0.267); H2 wrong > correct asymmetry collapses in thinking, evidence that H7 confidence-monotonicity (L1) breaks down with chain-of-thought |
| VQAv2 4-condition (b/a/m/d) | VQAv2 | full grid cross-model | TBD | ☐ P1 (kept, time-permitting) |

### 3.2 Mechanistic runs

| Experiment | Models | n | Status |
|---|---|---|---|
| E1 attention-mass (historical 6-model panel) | gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b | 200 stratified | ✅ — pre-2026-05-10 panel; InternVL3 removed from active mech panel 2026-05-10 |
| E1b per-layer localisation | same 6 (historical) | 200 | ✅ — 4 archetypes (SigLIP-Gemma early, mid-stack cluster CLIP-ViT/InternViT/ConvNeXt, Qwen-ViT late, FastVLM late text-stealing) |
| E1d causal ablation | same 6 (historical) | 200 | ✅ — single-layer null on 6/6; upper-half multi-layer **−4.0 to −10.5 pp** Δdf on 6/6 |
| E1-patch (digit-pixel attention) | gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b (4 perfect-square archetypes, n=400 each) | analysis + 2026-04-29 extension extraction | ✅ — peak digit/anchor 0.468–0.631 (+24 to +40 pp above fair share) on every panel model |
| **Phase D cross-dataset E1** (4 panel × 4 datasets + OneVision × 4 datasets, 2026-05-03; InternVL3 cell preserved on disk for audit but removed from active mech panel 2026-05-10) | gemma4-e4b, llava-1.5-7b, convllava-7b, fastvlm-7b + OneVision | 200 stratified per (model, dataset) | ✅ 20/20 active cells (+ 4 archived InternVL3 cells), commit `c556fb6`. Cross-dataset peak layer comparison via `scripts/analyze_cross_dataset_peaks.py` |
| **Phase E E1d cross-dataset** (OneVision × 4 datasets, 2026-05-03 + chart/math recovery 2026-05-04) | OneVision Main only | 200 stratified per dataset, 6 modes | ✅ 4/4 cells, commits `7a27750` + `2d11876`. Per-(model, mode) summary at `outputs/causal_ablation/_summary/per_model_per_mode.csv` |
| E4 mitigation Phase 1 (sweep, historical 3-model) | llava-1.5-7b, convllava-7b, internvl3-8b (historical; InternVL3 removed from active panel 2026-05-10) | 200 × 7 strengths | ✅ |
| E4 mitigation Phase 2 (full validation, historical 3-model) | same 3 (historical) | 17,730 | ✅ |
| E4 generalisation to other archetypes | gemma4-e4b, qwen2.5-vl-7b, fastvlm-7b | TBD | ☐ P3 |
| **E6 Subspace mitigation** (Phase B Stage 4-final, 2026-05-03) | OneVision Main | n=5000 wrong-base × 5 datasets | ✅ chosen L=26 K=8 α=1.0; commit `9f9dfa0`. Stage 4-final eval table at `docs/insights/_data/main_panel_5dataset_summary.md` |

### 3.3 Headline numbers (post-2026-04-28 re-aggregation)

Full tables (standard-prompt 7-model panel, E5b distance sweep,
E5c digit-pixel causality 2/3 models, E5e ChartQA+TallyQA 3-model,
E1d / E4 mechanism summary) live in
[`docs/insights/headline-numbers.md`](../docs/insights/headline-numbers.md).
Quick orientation:

- **VQAv2 main panel:** `adopt(a)` 0.021–0.066, `df(a)`
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
- **E4 mitigation:** LLaVA-1.5 / ConvLLaVA mid-stack-cluster
  Phase 2 `df` reduction −14.6 % / −9.6 % rel; em ↑;
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
| **E5e MathVista (γ-β)** | reasoning-mode VLM × MathVista — Qwen3-VL-8B-Instruct vs. Qwen3-VL-8B-Thinking (separate weights), 4-cond S1, max_new_tokens=512, runner is `</think>`-aware | ✅ landed 2026-04-28; full evidence doc `docs/insights/E5e-mathvista-reasoning-evidence.md` (2026-05-04). Headline (C-form, S1 anchor arm, all-base, n=365): instruct adopt(a)=0.074 / df(a)=0.102, thinking adopt(a)=0.117 / df(a)=0.291 (×1.6 adopt, ×2.9 df). **Wrong / correct split**: instruct df(a) wrong=0.256 / correct=0.021; thinking df(a) wrong=0.327 / correct=0.267 — correct-base ratio ×12.7. H2 wrong > correct asymmetry collapses in reasoning mode → first cell where the H7 continuous-confidence monotonicity (`L1-confidence-modulation-evidence.md`) is empirically violated |
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
| **L3** | per-confidence-quartile `adopt_rate` and `direction_follow_rate` table, model × dataset; compare to A1 binary split | ✅ 695,004 (sample × arm) records over 85 anchor cells (5-dataset × 7-model expansion 2026-05-04); `_data/L1_*.csv` |
| **L4** | report — pick the proxy + quartile shape with cleanest monotone trend; lift over A1 | ✅ `docs/insights/L1-confidence-modulation-evidence.md` — under `log_prob_sum` Q4 − Q1 mean df = **+0.191** (51/85 monotone, 60 %); `cross_entropy` is the paper-clean default at +0.156 (43/85). New §2.E in L1 doc. (Note: the InternVL3-8b H7 reversal previously surfaced here was retired with the InternVL3 panel removal 2026-05-10 — see §10 changelog.) |
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
| **E1-patch perfect-square panel — 5-model (shipped 2026-05-03; InternVL3 removed from active panel 2026-05-10 — see §10 changelog)** | digit-pixel-patch attention. Active 5-model panel: **gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, convllava-7b, fastvlm-7b**. (llava-interleave-7b dropped 2026-05-04 — low resolution.) | ✅ Phase D 20/20 active cells (+ 4 InternVL3 cells preserved on disk for audit). Headline: digit/anchor ratio +24–40 pp above fair share at peak layer on every panel model. |
| **OneVision Main extraction via AnyRes** | OneVision in §7.1-7.3 mech panel via `_compute_anchor_bbox_mass` extension to AnyRes per-image bbox routing + lite_eager monkey-patch (`scripts/extract_attention_mass.py:_install_lite_eager_attention`) to fit OOM. Image-anchor mass computed per crop, then composed. | ✅ Phase D 4-dataset cells (Plot/Tally/Info/VQAv2) shipped 2026-05-03 |
| **Cross-dataset peak comparison** | per-(model, dataset) peak layer comparison from Phase D cells. New finding: **OneVision peak is dataset-dependent** (L=27 PlotQA/TallyQA, L=14 InfoVQA/VQAv2). | ✅ `scripts/analyze_cross_dataset_peaks.py` shipped 2026-05-03 |
| **E1-patch causal ablation (digit-bbox region zero-mask) — Phase 3** | causal ablation restricted to digit-bbox region (not full anchor span). Patch-level surgical ablation. | ☐ Phase 3 — implementation depends on E1-patch attention path + bbox JSON pipeline |
| ~~**InternVL3-8b in main panel**~~ (revised 2026-05-04) | ~~Confirmed 27×27 perfect-square; routes through standard `_compute_anchor_bbox_mass`. Now in 5-model panel.~~ | ❌ **Removed 2026-05-10** — InternVL3-8b dropped from active mech panel and §3 main matrix. Outputs preserved on disk for audit. See §10 changelog. |
| **Qwen2.5-VL-7b non-perfect-square (appendix only)** | 17×23 non-square requires per-encoder routing not yet implemented. | ☐ appendix |

#### §7.4 E4 attention re-weighting mitigation

| ID | Experiment | Status |
|---|---|---|
| **E4 Phase 1 + 2 (existing 3-model mid-stack cluster, historical)** | mid-stack-cluster attention re-weighting (LLaVA-1.5 / ConvLLaVA / InternVL3 — InternVL3 cell preserved on disk for audit; not in active panel post-2026-05-10) | ✅ landed pre-restructure |
| **E4 + Main `llava-interleave-7b` (Phase 3)** | E4 sweep + full validation for Main model. Risk: Main may not fall in mid-stack cluster archetype (SigLIP ≠ CLIP-ViT/InternViT/ConvNeXt) → E4 may null-effect or backfire. Result drives §7.4 framing: if archetype-conditional, document as such; if cleanly transferred, headline strengthened. | ☐ Phase 3 — depends on E1-patch Main archetype assignment |
| **E4 §7.4 paper rendering** | report `direction_follow_rate` reduction, `exact_match` rise, `accuracy_vqa(b)` invariance side by side; the "free lunch" framing | ✅ `docs/insights/paper-section-7-4-mitigation-free-lunch.md` (2026-04-29) — needs Phase 3 update with Main extension |

#### §7.4.5 E6 Subspace deployable mitigation

| ID | Experiment | Status |
|---|---|---|
| **E6 (historical) — Tally-only N=5000 calibration, gt ∈ [0,8] eval** | Subspace L31_K04_α=1.0 clears 4-dataset selection rule. df −46% to −56% on TallyQA/ChartQA/VQAv2/MathVista; em +0.9 to +3.3 pp. Caveat: gt ∈ [0,8] restriction made result look like partial solution. | ✅ landed 2026-05-01 — historical baseline (superseded) |
| **E6 Phase 1 — Pilot grid + recalibration on PlotQA + InfoVQA pooled, full gt range, 5-dataset** | New calibration target: PlotQA+InfoVQA pooled n5k. **27-cell pilot grid** (L∈{25,26,27} × K∈{2,4,8} × α∈{0.5,1.0,2.0}) on OneVision Main. Aggregator `scripts/analyze_e6_pilot_cells.py` with em-drop dealbreaker rule + wrong-base baseline filter. | ✅ shipped 2026-05-03 — chosen cell **L=26 K=8 α=1.0** |
| **E6 Stage 4-final eval (chosen cell × 5 datasets)** | Apply L=26 K=8 α=1.0 to OneVision on 5-dataset full gt range, n=5000 wrong-base subset per dataset. Compare baseline vs mitigation arm directly (same predictions.jsonl, two cells). Δ-table generator: `scripts/build_e6_stage4_summary.py`. | ✅ shipped 2026-05-03 (commit `9f9dfa0`). Paired wrong-base avg Δdf=-2.9pp, **Δem(a)=+3.9pp benefit, Δem(b)=+8.8pp recovery → free-lunch** (paper task #38; earlier "-2.4pp em(a) cost" framing was a hand-copy error, retracted 2026-05-04). |
| **E6 Pilot validation (2026-05-01, llava n=200, historical)** | Existing Tally-calibrated subspace tested on PlotQA + InfoVQA pilots: PlotQA gt∈[1,8] Δdf −60%, em +3.85pp; InfoVQA gt∈[1,8] Δdf −24%, em +1.09pp. | ✅ historical |

### 6.6 §8 — Future work (scope only)

| ID | Direction | Status |
|---|---|---|
| **F1 (preferred)** | LLM/VLM architectural diff — same anchor delivered as text to LLM vs. as image to VLM, compare layer-wise integration profile (§7-style attention) | ✅ ideation paragraph drafted — `docs/insights/paper-section-8-f1-future-work.md` (2026-04-29) |
| **F2** | image-vs-text anchor — anchor image described as text and given to the same VLM; effect-size delta | ☐ ideation only |
| **F3** | Reasoning-mode VLM at scale — Qwen3-VL thinking, etc., on E5e cross-dataset matrix | ☐ scope only (γ-β is the minimal §8 stake) |

## 7. Pending work — Phase-structured priority queue (2026-05-04 update)

Phase 1 = paper consistency push (Main matrix at canonical setup). Phase 2 =
breadth-strengthening (digit-pixel causality 5-strat × 5 datasets). Phase 3 =
mechanism-Main alignment (E1-patch + E4 + Main). Phase 4 = paper polish.
Within phase, P0 blocks the phase target, P1 is opportunistic.

**As of 2026-05-04** — **Phase 1 P0s all complete or in-flight wrap**.
Master queue + recoveries shipped Phase B/C/D/E/G/H/I/J. 6-model main
panel × 5-dataset matrix 30/30 ✅ (post-InternVL3 removal 2026-05-10 —
panel re-baselined to 6 active models without internvl3-8b).
Mitigation chosen cell L=26 K=8 α=1.0 + Stage 4-final eval
landed (commit `9f9dfa0`). §7.1-7.3 cross-dataset attention 24/24
landed (commit `c556fb6`). Phase E E1d 4/4 landed (commits `7a27750` +
`2d11876` recovery). Memory + roadmap updated 2026-05-04 (commit `6d8dac4`).

### Phase 1 — paper consistency push ✅ COMPLETE

| Pri | Task | Status |
|---|---|---|
| **P0** | `experiment_e7_plotqa_full` 6-model panel | ✅ shipped |
| **P0** | `experiment_e7_infographicvqa_full` 6-model panel | ✅ shipped |
| **P0** | E6 Subspace recalibration + Stage 4-final eval | ✅ shipped (chosen L=26 K=8 α=1.0; commit `9f9dfa0`) |
| **P0** | Section §3.3 + §5 + §6 reaggregation against 5-dataset matrix | ✅ shipped — `docs/insights/_data/main_panel_5dataset_summary.md` (gitignored) |
| **P0 (NEW, 2026-05-04)** | Phase D §7.1-7.3 cross-dataset attention | ✅ shipped 24/24 |
| **P0 (NEW, 2026-05-04)** | Phase E E1d causal ablation OneVision × 4 datasets | ✅ shipped 4/4 (incl chart/math 2026-05-04 recovery) |
| **P0 (NEW, 2026-05-04)** | DataLoader prefetch infrastructure (`run_experiment.py`) | ✅ shipped, opt-in via VLM_ENABLE_PREFETCH=1 |
| **P0 (NEW, 2026-05-04)** | gitignore expansion for `docs/insights/_data/` etc + history rewrite | ✅ shipped via filter-repo + force-push |

### Phase 2 — breadth strengthening (Phase 1 results in hand)

| Pri | Task | Where | Estimate |
|---|---|---|---|
| **P1** | E5b/c (5-stratum + digit-mask) on PlotQA + InfoVQA × 3-model panel | §6.3 §5 | ~10h wall |
| **P1** | E5b/c on ChartQA + MathVista × 3-model panel (currently absent) | §6.3 §5 | ~10h wall |
| ~~**P1**~~ | ~~§6 confidence-modulated reaggregation across new 5-dataset matrix~~ | ~~§6.4~~ | ✅ landed 2026-05-10 (branch `paper/p4-13-section6-reaggregation`) — L1_*.csv was already on the 5-dataset × 7-model panel; this pass closes the figure + paper-prose drift left over. Fixes: (i) `paper_L1_confidence_quartile.png` regen against `cross_entropy` proxy (was filtering on the renamed/absent `entropy_top_k`); (ii) §6.2 worked-example table in `docs/paper/sections/06_confidence.md` (Q2/Q3/Q4 numbers + "wrong-base S1" mislabel — the CSV cell is all-base by quartile); (iii) §6.3 adopt/df gradient inline numbers on the same cell; (iv) emnlp_draft_ko §4.4 Insight 2 same gradient + made the worked-example cell explicit; (v) L1 evidence doc historical banner pointing readers to §2.E for canonical 5-dataset numbers. Verified γ-β ×12.7 (§6.5) + InternVL3 reversal table (§6.6) + headline +0.156 cross_entropy / 51 of 85 log_prob_sum (§6.4) all already match canonical CSV. |

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
| **P0 (NEW, 2026-05-08)** | E8 Mitigation capability-preservation regression test on OneVision Main | §7.4.5 | ✅ shipped — verdict: STRICT_FREE_LUNCH. **Extended 2026-05-09 to 8 benchmarks** (MME + AMBER follow-up): n_total = 27,097; macro Δ = +0.31 pp; HallusionBench +2.21 pp [+1.14, +3.28], **AMBER +0.19 pp [+0.05, +0.33]** (second hallu axis with CI excluding zero), POPE −0.06 pp [−0.21, +0.09], MME −0.13 pp [−0.76, +0.51] |
| ~~**P1 (2026-05-08)**~~ | ~~E8 follow-up: add MME + AMBER to capability panel~~ | §7.4.5 | ✅ shipped 2026-05-09 (folded into the P0 row above); driver gained `evaluate()`-rating-tolerance + aggregator merge subcommand (commit `31621f1`) |
| **P1 (NEW, 2026-05-08)** | E8 follow-up: MMMU-DEV-VAL with LLM-judge (multi-discipline reasoning, ~$1-2 GPT-4o-mini cost) | §7.4.5 | deferred until paper §7.4.5 prose locked; reviewer pre-check value justifies cost at that point |
| **P1** | §7.4.5 paper prose update (Tally-cal headline → PlotQA+InfoVQA-cal headline at full gt range) | `docs/paper/sections/07_*.md` | ✅ shipped 2026-05-08 — §7.4.5 + §7.5 + §7.6 + Capability Preservation subsections cohesive on L26_K8_α=1.0, 5-dataset paired-sids deltas, free-lunch, 6-bench E8 macro Δ +0.41 pp |
| **P1** | §3 / §5 / §6 paper prose update for 5-dataset matrix | `docs/paper/sections/0[3-6]_*.md` | ✅ shipped 2026-05-04 batch 1 (§3.6 / §4 / §5 / §6 5-dataset rewrites); cross-section drift to §1 / §8 closed 2026-05-08 (this changelog entry) |
| **P1** | Citation verification — every 2026 arXiv ID in `references/project.md` and §2 paper draft must resolve to a real paper | §9 caveat | ✅ shipped 2026-05-08 — 9/9 arXiv IDs verified, 3 venue tags resolved (NAACL 2025 ✅, HCAIR ICLR 2026 ✅, EMNLP Findings ❌ for CIVET); audit doc closed |
| **P3** | Image-vs-text anchor (F2) follow-up paper | §6.6 | future |
| ~~**P3**~~ | ~~InternVL3 + Qwen2.5-VL E1-patch non-square (appendix only)~~ | ~~§6.5 §7~~ | ❌ InternVL3 retired from active panel 2026-05-10 (see §10 changelog); Qwen2.5-VL non-square route still appendix-only |

### Phase 5 — paper review-driven hardening (NEW 2026-05-09, post 5-round loop)

Source: 5-round review loop (`docs/paper/reviews/_final_summary.md`)
+ post-review plan
([`docs/insights/plan_post_review_2026-05-09.md`](../docs/insights/plan_post_review_2026-05-09.md)).
Paper currently *Solid Findings, top of band*; tier-shift to Main
contingent on P0-1 bridge experiment.

| Pri | ID | Task | Where | Estimate | Tier impact |
|---|---|---|---|---|---|
| **P0** | P0-1 | γ-β residual-stream bridge experiment (cheap form) — project Qwen3-VL-Thinking trace residuals onto V_K[L=26]; test amplitude growth predicts ×12.7 correct-base df ratio | §4.6 + §6.2 + §8.4 item 1 | ~2 H100-day | **Tier-shifter** (bar-raiser signature ask). Single highest-leverage move. |
| **P0** | P0-2 | Eigenvalue spectrum of `D[:, L=26, :]` rank-8 elbow check | §6.4 + new figure | ~4 H100-hour | Theoretical contribution upgrade if elbow clean. |
| ~~**P1**~~ | ~~P1-3~~ | ~~Paired-bootstrap CI on §6.2.3 Table 6 (B=10,000)~~ | ~~`scripts/build_e6_stage4_summary.py` extension~~ | ✅ landed 2026-05-10 (branch `worktree-paper+p1-defense-r4`) | Closes R4 MAJ-4 + R4 MAJ-6 (Bonferroni-20). New script `scripts/build_e6_stage4_bootstrap_ci.py`; canonical CSV/MD `docs/insights/_data/stage4_final_per_dataset_ci.{csv,md}` + raw draws `_data/stage4_final_bootstrap_draws.npz`; insight `docs/insights/E6-stage4-paired-bootstrap-ci.md`. Δem(b) sign-clean 5/5 under Bonferroni-20; Δdf(a) sign-clean 1/5 (PlotQA only); InfoVQA Δdf 95 % CI [−4.7, +3.4] confirms inconclusive fence with paper's prior paired-Wilson estimate (~10 % within actual half-width 0.0406). |
| **P1** | P1-4 | CAA at K=1 + ITI at attention-head — actual Table 7 rows | §6.5 + new evidence doc | ~3 H200-day | Closes R4 MAJ-5 (structural Note → empirical). |
| **P1** | P1-5 | Random-K=8 baseline for §6.3 (Alt-1 falsification) | §6.3 Insight 1.5 | ~2 H100-day | Closes R4 CRIT-3. |
| ~~**P1**~~ | ~~P1-6~~ | ~~§A.5 27-cell pilot grid 4-metric heatmap aggregation~~ | ~~§A.5 + new canonical CSV~~ | ✅ landed 2026-05-10 (branch `worktree-paper+p1-defense-r4`) | Closes R4 CRIT-2 (cherry-pick concern). New script `scripts/aggregate_e6_pilot_grid.py`; canonical CSV `docs/insights/_data/E6_pilot_grid_27cells.csv` + `_selection_replay.md`; figures `docs/figures/E6_pilot_grid_{plotqa,infographicvqa}_heatmap.png`; insight `docs/insights/E6-pilot-grid-aggregation.md`. Em-deal-breaker rule non-binding on the grid (no cell rejected); chosen cell #17 ranks first by combined `|Δdf(a)|` under the same ex ante rule. |
| **P2** | P2-7 | E6 cross-architecture replication on Qwen2.5-VL-7B (different encoder archetype) | §6.6 + §1.4 framing | ~10 H200-day | Partial close of R4 CRIT-1 (N=1 → N=2). |
| **P3** | P3-8 | Paraphrase robustness (5 prompts × 5 datasets) | §A.X + §8.2 | ~3 H200-day | Defuses single-prompt critique. |
| **P3** | P3-9 | Closed-source defuse (~500 sample on GPT-4o or Gemini 2.5) | §3.6 + §4.* | ~1-2 day + ~$15 API | Defuses open-only critique. |
| **P3** | P3-10 | Encoder-family promotion to top-line contribution (camera-ready prose) | §1 + §1.5 + §5 | ~half-day | Camera-ready polish. |

**Recommended sprint ordering** (per plan §"Recommended execution sequence"):

- **Week 1 (Findings hardening):** ~~P1-3~~ ✅ + ~~P1-6~~ ✅ + P0-1 cheap + P1-5 in parallel. (P1-3 + P1-6 landed 2026-05-10.)
- **Week 2 (Main shift):** P0-2 + P1-4 + start P2-7.
- **Week 3 (Main consolidation):** Finish P2-7. If P0-1 cheap was positive, run clean form.
- **Camera-ready:** P3-8 + P3-9 + P3-10.

**Don't-touch protect-list (R5 bar-raiser):** (a − m) calibration contrast, single-model 6-callsite hedge, §6.2.3 reframing, Δem(non-anchored) ≥ 0 clause, §1.5 (1) hedge stack, §5.3 dataset-dependent peak self-disclosure. (~~§4.7 InternVL3 boundary case~~ retired 2026-05-10 with InternVL3 panel removal — see §10 changelog.)

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
- **`InternVL3-8b` prose-leak parse loss (historical caveat).** ~30 % of
  records drop out of E4 Phase-1 valid-triplet count because InternVL3
  emits prose ("based on…") truncated at `max_new_tokens=8`. No longer
  active — InternVL3-8b removed from active paper panel 2026-05-10 (see
  §10 changelog). Caveat preserved for audit only.
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

- **2026-05-10** — InternVL3-8b removed from active paper architecture
  (paper drafts, insights, scripts, references, configs, _data CSVs).
  `outputs/<exp>/internvl3-8b/` preserved for audit. H6 cluster
  re-anchored on 6-model main panel via
  `docs/insights/_data/H6_2axis_per_model.csv` (gemma anchoring corner
  vs llava-onevision-7b + qwen2.5-vl distraction corner). Branch
  `worktree-paper+remove-internvl3` PR #21.

- **2026-05-10 (§4.1 PlotQA swap + §4.4 6-bin headline switch + §C.1/§C.3/§C.4 appendix expansion — paper-tier reorganization).** Two structural changes to paper draft `docs/paper/emnlp_draft_ko.md` + 5 sections files. (1) **§4.1 single-dataset depth panel = PlotQA** (was VQAv2). 7-model panel (Gemma3-4b/27b, OneVision *(Main)*, InternVL3, Qwen2.5-VL-7b/32b + LLaVA-Interleave-7b with resolution caveat); df range +0.059~0.325 (vs VQAv2 +0.085~0.274); wrong > correct asymmetry df-기준 +7.4~34.4 pp on 7/7 모델 (PlotQA) — VQAv2의 +6.9~19.6 pp adopt-기준 동일 패턴이 chart-numeric stimulus에서 *증폭*되어 재현. Legacy VQAv2 panel은 새 §C.1 cross-stimulus replication appendix로 이전. New §C.3 = TallyQA + InfoVQA cross-dataset replication (Insight 1 능력↔끌림 역상관 / Insight 2 wrong > correct asymmetry 3/3 dataset robust). New script `scripts/_analyze_section41_swap.py`, canonical CSV `docs/insights/_data/section41_swap_analysis.csv` (gitignored), new figure `docs/figures/paper_4_1_PlotQA_correct_vs_wrong_df.png` via `scripts/_build_figure_4_1_plotqa.py`. (2) **§4.4 L1 confidence headline switched 4-quartile → 6-bin.** Aggregate over 85 anchor cells: mean df B6−B1 = +0.182 (`cross_entropy`) / +0.231 (`log_prob_sum`) — vs 4-bin +0.156 / +0.191, 6-bin +17~21 % larger gap (extreme bins capture confidence endpoints more accurately). Strict monotonicity 기준 변경: ≥ 4 of 5 bin-pair strict ↑ on **52-60 / 85 cells** (relaxed, 1 dip 허용) replaces 4-bin "fully strict 3/3 51/85". Worked example switched to PlotQA × LLaVA-OneVision-7b *(Main)* — B1=B2=0 broad floor → B3-B5 sigmoid rise → B6 saturation, df 0.000 → 0.000 → 0.028 → 0.128 → 0.238 → 0.289 (B6-B1 gap +28.9 pp). New §C.4 binning robustness appendix — 4-bin vs 6-bin sign-preservation table on 85 cells + InternVL3 chart-stack reversal 5/5 sign-preserved (§6.6 boundary case 4-bin Q decomposition retained for original investigation context). `scripts/analyze_confidence_anchoring.py` parameterized on `--n-bins`, new CSVs `_data/L1_confidence_quartile_long_6bin.csv` + `_data/L1_proxy_monotonicity_6bin.csv` + `_data/L1_proxy_comparison_6bin.csv`. Figure 5 (`paper_L1_confidence_quartile.png`) regenerated 6-bin. **Downstream paper updates (16 callsites total)**: abstract / §1.3 (i)-(iii) / §1.5 #1 + #3 / §4.1 6 cross-refs / §4.5 Insight 1 / §4.6 InternVL3 boundary case / Conclusion / §C.1 footer / §E intro + sections/01_intro.md (4 refs) + sections/03_method.md (1 ref + M2 historical preserved) + sections/05_distance_digitpixel.md (1 ref) + sections/06_confidence.md (full 6-bin rewrite §6.1-§6.7, dual-form §6.6) + L1 evidence doc 2026-05-10 update header. Figure 4 (`paper_cross_dataset_summary.png`) also regenerated to fill 6-model × 5-dataset 30-cell heatmap (was showing 4×4 stale visual due to filter bug + missing model rows). Script update: `scripts/build_paper_figures.py:fig_cross_dataset_summary` + `fig_L1_confidence_quartile`.

- **2026-05-10 (§6 confidence reaggregation drift fix — P4-13).** Branch
  `paper/p4-13-section6-reaggregation`. The L1_confidence_quartile_long.csv
  + L1_proxy_*.csv canonical artefacts already covered the 5-dataset ×
  7-model main matrix from the 2026-05-04 reaggregation; this pass
  closes the figure + paper-prose drift that survived. (1) Audit trail
  built against canonical CSV: γ-β ×12.7 (§6.5), InternVL3 reversal
  table (§6.6), headline +0.156 cross_entropy / 51 of 85 log_prob_sum
  monotone cells (§6.4) all match — no edits needed. (2) Drift fixes:
  (a) `scripts/build_paper_figures.py:fig_L1_confidence_quartile`
  filtered on the renamed/absent `entropy_top_k` proxy → switched to
  paper-default `cross_entropy`, reduced to the worked-example single
  cell so the figure and §6.2 table tell the same story; figure
  regenerated. (b) `docs/paper/sections/06_confidence.md` §6.2 worked
  example: Q2/Q3/Q4 numbers were stale (Q2 df 0.062 → 0.044, Q3 df
  0.158 → 0.137, Q3 adopt 0.137 → 0.159, Q3 em_b 0.42 → 0.488, Q4
  em_b 0.34 → 0.284) and the "wrong-base S1" framing was a mislabel
  — the L1_* CSV cell is all-base by confidence quartile. Fixed both,
  added an in-paper CSV-row pointer for traceability. (c) §6.3
  inline gradient updated to match. (d) `docs/paper/emnlp_draft_ko.md`
  §4.4 Insight 2 same gradient updated + made the worked-example cell
  explicit instead of leaving bare numbers. (e) L1 evidence doc gets
  a §0 historical banner pointing readers to §2.E for canonical
  5-dataset numbers (per advisor: defer the §0–§2.4 single-model
  rewrite to a separate doc-rewrite branch — this PR scope is
  paper-table truth-from-CSV). **Paper file edits (b)/(c)/(d) are
  local-only — `docs/paper/*` is gitignored, so this changelog
  entry is the recovery record if the local working copy is lost.
  The CSV-row pointer added to §6.2 (`experiment_e5c_vqa,VQAv2,
  llava-next-interleaved-7b,a,S1,cross_entropy`) is the verification
  anchor — re-applying the fix is a single grep against
  `_data/L1_confidence_quartile_long.csv`.** PR opened against
  master.

- **2026-05-10 (P4-12 OneVision E1d analyzer fix closed).** `scripts/analyze_causal_ablation.py` 의 두 stratification 버그가 수정되어 OneVision Main Phase E (5 dataset × n=200 stratified) 결과가 §5.2 / §5.3 / §E.2로 통합. **Headline:** single-layer ablation 5/5 null on OneVision (max |Δdf| = 1.5 pp on InfoVQA, 모든 95 % CI overlap 0) — 6-mech panel의 6/6 null과 일관, multi-layer redundancy claim의 *확장 검증*. Upper-half ablation은 6-mech panel의 균일 −4 ~ −10.5 pp significant와 달리 OneVision에서는 5/5 null at n=200 (point estimates ∈ [−3.9, +0.4] pp) — §5.3 dataset-dependent peak (Plot/Tally L=27 vs Info/VQAv2 L=14)와 일관 heterogeneity, §6.2 subspace-projection 도구 선택의 *layer-uniform attention re-weighting 한계*라는 mechanism-level 동기 보강. 버그 두 건: (i) `_build_triplets`의 base/anchor join에 dataset key 누락 (commit `a7e391c`); (ii) per-run dataset detection 부재 (commit `de1f94e`). 문서 통합: `docs/insights/E1d-causal-evidence.md` 2026-05-10 update block, paper Abstract / §1.3 / §5.2 / §5.3 / §8.2 / §E.2, `docs/insights/plan_post_review_2026-05-09.md` P4-12 mark closed, `docs/paper/reviews/_final_summary.md` item 9 mark closed. PR: `paper/p4-12-onevision-e1d-analyzer-fix` (#16).

- **2026-05-10 (P1-3 paired-bootstrap CI + P1-6 27-cell pilot grid
  aggregation — Phase 5 adversarial-defense rigor batch 1).** Branch
  `worktree-paper+p1-defense-r4`. Two R4 follow-ups closed in parallel
  on the existing OneVision Main canonical runs (no new GPU compute).
  - **P1-3 (R4 MAJ-4 + MAJ-6 close).** Paired-bootstrap CI for §6.2.3
    Table 6 with B = 10,000 (sid-paired resampling, per-arm denominators
    recomputed each bootstrap so adopt's `pb ≠ anchor` and df's
    `pa ≠ pb` clauses shift correctly per arm). 95 % equal-tail
    percentile + Bonferroni-20 corrected (99.75 %, family = 5 datasets ×
    4 metrics). New script `scripts/build_e6_stage4_bootstrap_ci.py`;
    canonical artefacts `docs/insights/_data/stage4_final_per_dataset_ci.{csv,md}`
    + raw draws `_data/stage4_final_bootstrap_draws.npz` (20 arrays,
    B = 10,000 each); insight cousin
    `docs/insights/E6-stage4-paired-bootstrap-ci.md`. Sign-clean count:
    Δadopt(a) 2/5 at 95 %, Δdf(a) **1/5 (PlotQA n=2,306 [−6.9, −3.4])**,
    Δem(a) 3/5 at 95 % / 2/5 under Bonferroni, **Δem(b) 5/5 sign-clean
    at both 95 % and Bonferroni-20**. **InfoVQA Δdf 95 % CI = [−4.7, +3.4]**
    — `inconclusive fence` confirmed with real CI numbers (paper's earlier
    paired-Wilson half-width estimate ~±0.04 to ~±0.06 was within ~10 %
    of the actual half-width 0.0406; sanity gate passed before
    committing). Paper edits: §6.2.3 Table 6 + reframing paragraph
    rewritten with CI numbers; abstract Δdf qualifier rewritten with
    PlotQA cell + Δem(b) Bonferroni-robust call-out; §8.2 P1-3 deferred
    bullet struck; §8.4 item 5 narrowed to CAA·ITI only; English
    sister-section `docs/paper/sections/07_mechanism_mitigation.md`
    Table 6 + Three properties paragraphs synchronised.
  - **P1-6 (R4 CRIT-2 close).** 27-cell pilot grid (L ∈ {25,26,27} ×
    K ∈ {2,4,8} × α ∈ {0.5,1.0,2.0}) on PlotQA + InfoVQA n=250 calibration
    pilots aggregated into a 4-metric heatmap (Δadopt(a) / Δdf(a) /
    Δem(a) / Δem(b)) per calibration dataset. New script
    `scripts/aggregate_e6_pilot_grid.py`; canonical CSV
    `docs/insights/_data/E6_pilot_grid_27cells.csv`; selection-rule
    replay markdown `_data/E6_pilot_grid_27cells_selection_replay.md`;
    figures `docs/figures/E6_pilot_grid_{plotqa,infographicvqa}_heatmap.png`
    (4 metrics × 3 layers × K-α heatmap each, chosen cell starred);
    insight `docs/insights/E6-pilot-grid-aggregation.md`. **Em-deal-breaker
    rule (Δem(a) ≤ −6 pp on either calib) is non-binding on the grid**
    — no cell rejected (PlotQA min Δem(a) = −1.2 pp on cell #19, InfoVQA
    min = +0.4 pp on cell #1). Chosen cell #17 (L=26, K=8, α=1.0) ranks
    first by combined |Δdf(a)| (mean −4.4 pp; #2 #8 at −3.2 pp; 1.2 pp
    margin) — direct response to reviewer cherry-pick concerns. Paper
    edits: §A.5 stub replaced with full 4-metric aggregation table +
    top-5 ranking + heatmap figure pointers + binding-clause analysis;
    §6.2.2 prose updated with §A.5 cross-link; §8.2 P1-6 deferred bullet
    struck; English sister-section §6.2.2 cross-link added.
  - **No GPU compute consumed** — both tasks ran on existing
    `outputs/e6_steering/llava-onevision-qwen2-7b-ov/` predictions
    (chosen-cell sweeps + pilot grid jsonls; canonical OneVision Main
    artefacts unchanged). Reproducible notebook
    `notebooks/E6_phase5_p1_3_p1_6_demo.ipynb`. Total work ≈ 2 h
    end-to-end (script + run + paper integration + roadmap).

- **2026-05-09 ~21:40 (5-round paper review loop + post-review plan +
  selective gitignore overrides for tracked review trail).**
  Multi-agent review-revise pipeline shipped end-to-end on
  `docs/paper/emnlp_draft_ko.md`. Methodology / writing / novelty /
  adversarial / bar-raiser personas + author-reviser between
  rounds. Paper 516 → 604 lines net (+88).
  - **Pipeline infrastructure** (commit `8ffdc2d`):
    `.claude/agents/paper-reviewer-{methodology,writing,novelty,aggressive,bar-raiser}.md`
    + `paper-reviser.md`; `.claude/commands/paper-review-loop.md`.
    Each reviewer enforces "no vague critique" + canonical-CSV
    verification + § + verbatim + suggested-fix bar; reviser uses
    EDIT / PARTIAL EDIT / REBUT / DISAGREE / DEFER classification.
    `.gitignore` selective overrides allow `.claude/agents/paper-*`
    and `.claude/commands/paper-review-loop.md` and
    `docs/paper/reviews/*.md` to be tracked while keeping rest of
    `.claude/` and `docs/paper/` local.
  - **Round-by-round** (review + response files at `docs/paper/reviews/round{1..5}_*.md`):
    - **R1 methodology** — 8 must-fix incl 4 CRIT data errors
      (Table 5 E4 swap, Table 3 TallyQA gemma3-27b values, ActAdd
      +57 % untraceable, abstract 5×7=85 arithmetic). All headline
      numbers (E6 Table 6, E8 Table 8, γ-β ratios, L1 51/85)
      verified clean against canonical `_data/`.
    - **R2 writing** — 11 must-fix; 6 forced Korean coinages
      stripped (이분법 / 주력 / 발화 / 회귀); abstract "입증" →
      "강하게 뒷받침"; Figure 5 caption / 4.4 Insight 2 contradiction
      fixed.
    - **R3 novelty** — VLMBias factual correction (visual objects,
      not class labels); §2 *Activation steering and concept erasure*
      paragraph added (CAA / ITI / LEACE); strict-free-lunch 4-clause
      formal definition + Chand et al. 2025 No-Free-Lunch precedent;
      4 references added.
    - **R4 aggressive** — CRIT-1 N=1 model on E6 chain hedged across
      6 callsites as case study; §A.4 FLUX seed=1729 surfaced
      (3-round defer closed); §A.5 27-cell pilot grid label;
      §7 Bonferroni-6 robustness; 4 items honestly DEFERred with
      GPU-hour estimates.
    - **R5 bar-raiser** — predict-then-verify chain (§5.2 → §6.4)
      sharpened as theoretical contribution at 4 callsites; new
      §8.4 후속 작업 with bridge experiment as lead item.
  - **Final tier verdict (3-way convergent):** Solid Findings, top
    of band. Borderline / weak-accept Main contingent on §8.4 item 1
    bridge experiment (γ-β residual-stream amplitude on K=8 subspace,
    cheap ~2 H100-day / clean ~1 H200-week).
  - **5-year-citable finding:** §5.2 → §6.4 predict-then-verify
    chain ("multi-layer redundancy predicts single-direction
    mitigation failure cross-dataset"), framed as paper's
    *이론적* contribution.
  - **Post-review plan**
    [`docs/insights/plan_post_review_2026-05-09.md`](../docs/insights/plan_post_review_2026-05-09.md):
    17 DEFER items consolidated into Phase 5 priority queue
    (P0 × 2 tier-shift, P1 × 4 rigor, P2 × 1 cross-arch, P3 × 3
    camera-ready, P4 × 7 future). Priority + cost + dependency +
    acceptance criteria per item. Recommended 3-week execution
    sequence ending in P0-1 bridge experiment + paper update.
  - **Bar-raiser don't-touch protect-list (7 items):** (a − m)
    contrast / 6-callsite hedge / §6.2.3 reframing / Δem(b) clause /
    §1.5 (1) hedge stack / §5.3 self-disclosure / §4.7 boundary
    case. Phase 5 work must respect.
  - **Note on dispatch:** custom `paper-reviewer-*` and
    `paper-reviser` agents were dispatched as `general-purpose`
    subagents reading the persona file content (workaround during
    initial registration race). After pod restart they register
    properly via `.claude/agents/`. Future runs of
    `/paper-review-loop` should invoke them via `subagent_type`
    directly. Review quality and audit trail are intact; output
    files substantive (16-50 KB per review).

- **2026-05-09 ~20:35 (E8 follow-up — MME + AMBER added, 8-bench panel).**
  Phase 4 P1 follow-up shipped on isolated worktree branch
  `worktree-phase4-mme-amber` (PR #14). Two new held-out benchmarks
  added to the capability eval at the same chosen cell
  (L = 26, K = 8, α = 1.0) on OneVision-7b Main:
  - **MME** (n = 2,374, 14 categories incl. the Count subset that
    directly exercises the number-anchor failure mode) —
    Δ = −0.13 pp, 95 % CI = [−0.76, +0.51], essentially neutral.
  - **AMBER** (n = 14,216, contamination-clean Nov 2023 multi-dim
    hallucination — largest sample on the panel) —
    **Δ = +0.19 pp, 95 % CI = [+0.05, +0.33], CI excludes zero.**
    Second hallucination axis after HallusionBench to land a
    statistically significant positive Δ.
  - **8-bench merged final**: macro Δ = +0.31 pp (was +0.41 on
    6-bench; AMBER's tight CI on n = 14,216 pulls macro down without
    breaching any threshold). Verdict: **STRICT_FREE_LUNCH** preserved.
    Contamination-resistant floor of the panel rises from n = 1,500
    (MMStar alone) to n = 18,090 (MMStar + MME + AMBER).
  - **MME per-category breakdown**: Count subset (n = 60), the in-domain
    analogue of the number-anchor failure mode, shows **Δ = 0.00 pp
    exact** — every paired prediction matches between baseline and
    +mit. Existence (n = 60) likewise Δ = 0.00 pp at ceiling. Direct
    evidence the mitigation acts on cross-modal anchor pull, not
    counting capability itself.
  - **Driver hardening (commit `31621f1`)**: `dataset.evaluate()`
    wrapped in try/except so VLMEvalKit's `MME_rating` /
    `AMBER_rating` helpers (which raise on broken pair structure /
    aggregate-stat edge-cases AFTER writing the per-question
    `_auxmatch.xlsx`) no longer block per-question score extraction.
    Caught at smoke time when MME pair-rating raised IndexError on
    the random 50-q sub-sample.
  - **Aggregator gained `merge` subcommand** so the 6-row final
    survives the 8-row extension without re-running the original
    benchmarks (later inputs override earlier on duplicates;
    +3 unit tests; 15 passing total).
  - **Memory `feedback_vlmevalkit_quirks.md` extended to 5 quirks**
    (post-side-file rating helpers raising; MME absolute-score
    reporting caveat).
  - Files committed: `scripts/run_capability_eval.py`,
    `scripts/aggregate_capability_eval.py`,
    `tests/test_capability_eval.py`,
    `configs/capability_eval_mme_amber.yaml`,
    `docs/insights/E8-capability-preservation-evidence.md`,
    `docs/insights/{phase1-p0-v3-summary,headline-numbers}.md`,
    `docs/experiments/E8-capability-preservation.md` (new). Wall
    time: 2 h 8 min on a single H200, sequential, $0 (no LLM-judge).

- **2026-05-08 ~21:30 (Phase 4 P1 paper polish — cross-section
  consistency pass + venue-tag verification).**  Phase 4 P1 batch
  shipped (paper polish, write phase).
  - **§1 / §8.5 cross-section drift resolved against E8 STRICT_FREE_LUNCH.**
    §1 abstract now carries the 6-benchmark capability-preservation
    headline (macro Δ = +0.41 pp, HallusionBench Δ = +2.21 pp 95 % CI
    excluding zero, POPE Δ = −0.06 pp 95 % CI [−0.21, +0.09]).
    §1.4 mechanism+mitigation paragraph extended with the same.
    §1.6 contribution #5 lists E8 explicitly. §8.5 conclusion
    rewritten to lead with E6 deployable + E8 capability preserved.
  - **§1.3 confidence-claim numbers re-aligned to §6 actual figures.**
    Old "Q4-Q1 = +15.2 pp" / "23 of 35 cells" (legacy 4-dataset
    4-model panel) replaced with paper-default
    `cross_entropy` Q4-Q1 = +15.6 pp (43 / 85 cells) and
    `log_prob_sum` +19.1 pp (51 / 85, 60 %) on the 5-dataset ×
    7-model panel. §1.6 contribution #3 now lists the six-model
    Phase 1 P0 v3 main panel + supplementary llava-interleave
    cell + cross-dataset E5e + γ-β reasoning-mode pair.
  - **§5.4 stale "pending gemma3-27b-it E5c" wording removed.** Cell
    landed 2026-04-29 (VQAv2 a−m = +5.7 pp, TallyQA a−m = +2.1 pp).
    Table extended to include the cell; prose rewritten to reflect
    3-model panel resolution.
  - **§4.4 sample-sizes table extended with E8 capability eval row**
    (10,507 questions × 2 variants = 21,014 generations).
  - **Citation venue-tag audit closed (2026-05-08).** Three
    arXiv-2025+ venue tags from `docs/insights/citation-audit-2026-05.md`
    verified via WebFetch on arXiv abs pages:
    NAACL 2025 ✅ for 2502.08193 (Wang-Zhao-Larson typographic);
    HCAIR @ ICLR 2026 ✅ for 2505.15392 (Huang anchoring);
    EMNLP Findings 2025 ❌ for 2506.05146 (CIVET) — paper is arXiv
    preprint only, no named venue. §2 paper draft tag removed
    (arXiv:2506.05146, 2025); `references/project.md` "What EMNLP
    Main demands" strategic argument softened (CIVET no longer cited
    as a "Findings" example, instead as a behavioral-probing-only
    arXiv example in the same class). Audit doc updated with verified
    statuses + new action items pruned to non-arXiv reference checks
    (Jones&Steinhardt, Echterhoff, Goh, Hagendorff, Mussweiler&Strack,
    Tversky&Kahneman, Jacowitz&Kahneman).
  - Files committed: `references/project.md`,
    `docs/insights/citation-audit-2026-05.md`. Paper-section edits
    are local-only (`docs/paper/sections/01/02/04/05/08_*.md`,
    gitignored per existing convention).

- **2026-05-08 ~20:45 (E8 follow-up: POPE added to the panel).**
  Sixth held-out benchmark — POPE (object-existence hallucination
  diagnostic, n=5127) added to the capability eval as a complementary
  hallucination axis to HallusionBench (illusion/depth). **Result:
  Δ=−0.06pp, 95 % CI=[−0.21, +0.09]** — tight CI essentially pins the
  effect to zero. 6-benchmark macro Δ = **+0.41 pp**, verdict still
  STRICT_FREE_LUNCH. POPE is the largest single benchmark on the panel,
  so its tight CI dominates the noise-floor estimate.
  - Driver fix landed (commit `23fe5bc`): VLMEvalKit's YORN
    `evaluate()` short-circuits on existing `_auxmatch.xlsx`, so the
    self-test pass's 2-row auxmatch poisoned the full sweep's
    extraction (driver reported n=2 despite full 5127-question
    inference). Fixed by wiping `out_dir/<variant>/<bench>` at run
    start.
  - Memory file `feedback_vlmevalkit_quirks.md` extended to four
    quirks (YORN cache + 3 prior); insight doc + paper §7.4.5
    sub-section regenerated with the 6-row table.

- **2026-05-08 ~04:38 (E8 capability-preservation regression test).**
  New Phase 4 P0 shipped. Spec
  `docs/superpowers/specs/2026-05-08-mitigation-general-capability-design.md`
  + insight doc `docs/insights/E8-capability-preservation-evidence.md`.
  - VLMEvalKit (commit `97ce037`) pinned as a dep; LLaVA-OneVision-HF
    backend chosen over LLaVA-NeXT (avoids `llava` dep conflict; same
    Qwen2 weights at the L=26 hook site).
  - New `LLaVAOneVisionMitigated` subclass installs the chosen-cell hook
    at construction; `vlm_anchor.hooks.make_subspace_projection_hook`
    now the single source of truth (e6_steering_vector.py keeps a
    1-line shim).
  - Driver `scripts/run_capability_eval.py` orchestrates per-benchmark
    interleaving (RealWorldQA → OCRBench → HallusionBench → MMStar →
    MMBench-DEV-EN, fast-first). Aggregator
    `scripts/aggregate_capability_eval.py` ships with pre-registered
    thresholds (per-bench Δ ≥ -1.0pp, macro Δ ≥ -0.5pp), 12 unit tests
    cover hook math + verdict logic + threshold-pinning.
  - **Result: STRICT_FREE_LUNCH on full sweep (~1.5h H200, no LLM-judge).**
    Macro Δ = +0.50pp; per-bench Δ ∈ [-0.80, +2.21]; HallusionBench
    Δ=+2.21pp 95% CI=[+1.14, +3.28] **excludes zero — statistically
    significant positive**. §7.4.5 free-lunch claim (originally
    Δdf ≤ 0 ∧ Δem(a) ≥ 0 ∧ Δem(b) ≥ 0 within anchoring family) extends
    to general VLM capability.
  - Pipeline cross-check vs lmms-lab model card published numbers:
    MMStar 61.67 vs 61.7 (essentially identical match); RealWorldQA
    +3.5pp, MMBench +1.24pp, OCRBench match. Strong evidence of HF
    mirror weight equivalence at the Qwen2 LM layer where the hook
    operates.

- **2026-05-04 ~17:30 (Phase 2 insight mining batch 1).** Audit pass on
  experiments that had outputs but no full insight write-up. Landed:
  - **E5e γ-β reasoning evidence doc** (`docs/insights/E5e-mathvista-reasoning-evidence.md` +
    `notebooks/E5e_reasoning_ablation.ipynb` +
    `_data/experiment_e5e_mathvista_reasoning_per_cell.csv`). Wrong-base / correct-base
    split surfaced — correct-base df ratio is **×12.7** (instruct 0.021 → thinking 0.267),
    much stronger than the all-base ×2.9 headline. Direct violation of H7 confidence
    monotonicity in reasoning mode.
  - **E7 PlotQA + InfoVQA standalone evidence doc**
    (`docs/insights/E7-plotqa-infovqa-evidence.md` +
    `notebooks/E7_plotqa_infovqa.ipynb`). Surfaces 3 findings the §3.3
    umbrella hides: (1) Gemma3 anti-scaling is **PlotQA-driven** and reverses
    on InfoVQA; (2) **InternVL3-8b shows H2 collapse** (wrong−correct df gap
    +0.008 PlotQA / +0.024 InfoVQA — panel-side analogue of the thinking-mode
    H2 collapse); (3) 6/7 PlotQA models show **em(a) > em(b) un-mitigated
    free-lunch**, motivating §7.4.5 E6 mitigation.
  - **§6 confidence quartile reaggregation on 5-dataset × 7-model matrix.**
    Ran `scripts/recompute_answer_span_confidence.py` on 172 jsonl files
    (added length-normalised proxies to runs lacking `answer_span_*` fields)
    + `scripts/analyze_confidence_anchoring.py`. Coverage 35 → 85 anchor
    cells; df Q4 − Q1 = +0.191 on `log_prob_sum` (51/85 monotone), +0.156
    on `cross_entropy` paper-default (43/85). New §2.E in
    `L1-confidence-modulation-evidence.md` documents **InternVL3-8b H7
    reversal** on PlotQA / ChartQA / InfoVQA — least-confident records
    anchor *less*, not more (Δ −0.089 to −0.156). Same model with
    panel-side H2 collapse. §2 H7 row updated from ☐ to ✅ with boundary
    cases noted; H4 row flipped to ❌ (γ-β amplification finding).
  - **OneVision Phase E E1d analyzer fix** (commits `a7e391c`, `de1f94e`):
    `analyze_causal_ablation.py` now emits per-(model, dataset) cells with
    OneVision dataset routing (hardcoded timestamp map + susceptibility-CSV
    auto-detect fallback). Reaggregated 5 OneVision Phase E run dirs to add
    M2 / C-form per-row flags.
  - **OneVision Phase E E1d INFERENCE bug** (commit `8895128`): bisected the
    "ablation no-op on 4/4 datasets" symptom to commit `7f8ebb6` (May 3 07:23 KST,
    "switch to SDPA"). Empirical verification on 5 chartqa sids: eager 3/5 differ
    vs sdpa 1/5 — SDPA dispatch silently drops the `attention_mask` bias from
    `_make_anchor_mask_hook`. The orphaned PlotQA run 20260503-002050 (eager
    pre-commit) has 26-36 % differing — the expected level. New
    `--attn-implementation` flag added; eager re-run completed
    (`scripts/_phase2_e1d_eager_rerun.sh` 17:40 → 21:35, ~3h55m on H200).
    **Final 4-dataset deltas vs baseline df** (`per_model_per_mode.csv`):
    TallyQA Δ_lower_half=+0.050 / Δ_all=−0.040; MathVista Δ_lower_half=+0.075 /
    Δ_all=−0.045; ChartQA Δ_lower_half=+0.026 / Δ_all=+0.006; InfoVQA all
    Δ ≤ |0.008| (minimal). 3/4 datasets reproduce classic E1d signature
    (mid-stack ablation amplifies, full ablation drops); InfoVQA's flat
    response is consistent with its Phase D peak at L=14 (the chosen
    `--peak-layer 27` setting routes ablation through the wrong band).
    `phase1-p0-v3-summary.md` caveat #2 retired.
  - Blast-radius audit: SDPA-mask-bias bug confined to `causal_anchor_ablation.py`.
    E4 attention re-weighting uses `build_eager_runner` explicitly. E6 family
    (steering / leace / cogbias / qao) hooks layer output, not attention_mask.
    §7.1-7.3 attention extraction (`extract_attention_mass.py`) uses
    `lite_eager` monkey-patch on a separate dispatch path that reads attention
    output rather than modifying input mask. None affected.

- **2026-05-04 ~10:31 (Phase 2 E5b 5-stratum cross-dataset, OneVision Main).**
  Queue script `scripts/_phase1_e5b_5strat_onevision_queue.sh` ran 4
  datasets (MathVista, ChartQA, InfoVQA, PlotQA) in S1→S5 12-cond
  (b + a/m × {S1..S5} + d) under `anchor_distance_scheme: relative`. Total
  85,258 records on 2 H200 with DataLoader prefetch + 2-shard sharding,
  ~3.5 h wall-clock. Per-cell aggregation via `analyze_e5e_wrong_correct.py`,
  decay summary via `scripts/build_e5b_5strat_decay_summary.py`. Insight
  doc: `docs/insights/E5b-cross-dataset-onevision.md`. **Headline**: adopt
  monotonic decay S1→S5 on every dataset (PlotQA: 8.7 % → 0.7 %, MathVista:
  10.5 % → 1.6 %, InfoVQA: 4.5 % → 0.2 %, ChartQA: 2.8 % → 0.4 %); df
  gentle decay (-0.04 to -0.06) on 3/4 datasets, MathVista flat; em stable
  across strata. Plausibility-window claim replicates at full GT range on
  a second architecture. Outputs gitignored under
  `outputs/experiment_e5b_5strat_<ds>_onevision/`. Master merge commit
  `16aebcf`.

- **2026-05-04 ~02:45 (Phase 1 P0 v3 substantively COMPLETE).** Single-session
  master queue execution + recoveries. All on origin (master + branch pushed
  after `git filter-repo` removed historical 113MB CSV from history, see
  below).

  **Phases B → J orchestrated** by `scripts/_phase1_post_pilot_master_queue.sh`
  (commits `9f9dfa0` → `8c7cc43` → master merge):
  - **Phase B** Stage 4-final mitigation: chosen cell **L=26 K=8 α=1.0**
    (PlotQA+InfoVQA pooled n5k, 5-dataset eval). df reduction works, em(a)
    -2.4pp avg (acceptable per em-drop dealbreaker), surprise **em(b) +9.2pp
    avg** recovery on wrong-base sids → paper §7.4 angle (#38).
  - **Phase D** §7.1-7.3 cross-dataset attention: 5-panel × 4 datasets +
    OneVision × 4 datasets = 24/24 cells. Cross-dataset peak comparison via
    `scripts/analyze_cross_dataset_peaks.py` reveals OneVision peak L=27 on
    PlotQA/TallyQA but L=14 on InfoVQA/VQAv2 → late-layer mechanism is
    **dataset-dependent**, not purely model-specific.
  - **Phase E** E1d causal ablation OneVision × {Tally, Info, Chart, Math}
    4/4 (Chart + Math re-ran 2026-05-04 02:00 with proper per-dataset
    susceptibility CSVs after PlotQA-CSV-reuse bug, commit `2d11876`).
  - **Phase G** new-model baselines: internvl3-8b (config patched
    `OpenGVLab/InternVL3-8B` → `-hf` mirror), qwen2.5-vl-32b-it, gemma3-4b-it
    × 5 datasets each.
  - **Phase H** qwen2.5-vl-7b §7.1-7.3 attention × 5 datasets.
  - **Phase J** branch merge → master + push to origin.

  **6-model main panel finalised**: llava-onevision-7b, qwen2.5-vl-7b,
  internvl3-8b, gemma3-4b, qwen2.5-vl-32b, gemma3-27b. **llava-next-interleaved
  dropped** (low native resolution, 2026-05-04 user decision). Summary at
  `docs/insights/_data/main_panel_5dataset_summary.md` (gitignored). Last
  cell internvl3-8b/TallyQA being rerun (2-shard + DataLoader prefetch
  ON, ETA ~07:00).

  **Pilot grid + cell selection** (commits `8fe1d81`, `dd17457`): 27-cell
  pilot (L∈{25,26,27} × K∈{2,4,8} × α∈{0.5,1.0,2.0}) on PlotQA+InfoVQA
  pooled n5k. `scripts/analyze_e6_pilot_cells.py` aggregator with em-drop
  dealbreaker rule + wrong-base baseline filter (was averaging over correct-
  base too, gave -45pp spurious dEM until fixed).

  **Infrastructure milestones (this session)**:
  - **DataLoader prefetch** in `run_experiment.py` (opt-in
    `VLM_ENABLE_PREFETCH=1`). Smoke-tested byte-equal modulo 10⁻⁵ tail-token
    logits. Used for internvl3 tally rerun.
  - **lite_eager attention monkey-patch** defensive `is_causal` fix
    (mirrors HF SDPA's dynamic logic).
  - **gitignore expansion**: `docs/insights/_data/`, `docs/paper/`,
    `docs/ppt/`, `docs/superpowers/plans/`, `docs/CHANGELOG.md` all
    untracked + history-rewritten via `git filter-repo`. Local files
    preserved; force-pushed master + feature branch.
  - **fast-tail watcher bug fix** (`scripts/_phase1_recover_internvl3_fast_tail.sh`):
    `find -name predictions.jsonl` was matching shard sub-files, firing
    kill trigger before merge step. Now uses `-maxdepth 2` + `grep -v _shards`.
  - **microsec + PID suffix** in `extract_attention_mass.py` output dir
    (avoids 2-process-same-second collision after gemma4-e4b Phase D
    corruption incident).

  **Recovery patterns established**: panel HF id corrections (panel models
  in master queue had wrong HF ids — `liuhaotian/llava-v1.5-7b` →
  `llava-hf/llava-1.5-7b-hf`, `ConvLLaVA/ConvLLaVA-Stage5-7B-LoRA` →
  `ConvLLaVA/ConvLLaVA-sft-1536`), parallel-write dir-collision fix.

- **2026-05-02 ~08:39 (Phase 1 P0 v3 multi-GPU restart).** GPU count expanded
  from 1 → 3 (GPU 0/1/2 available). Sharded run + parallel orchestrator
  added (commit `24e008c`):

  - `scripts/run_experiment.py` — `--shard-idx N --num-shards K --output-dir`
    flags slice samples round-robin (`samples[N::K]`) and route output;
    figures skipped under sharding.
  - `scripts/run_experiment_sharded.py` — fans out K subprocesses pinned via
    `CUDA_VISIBLE_DEVICES`, waits, concatenates predictions, recomputes
    summary into the shared `<ts>/predictions.{jsonl,csv}+summary.json`.
    Greedy decoding (temperature=0) keeps merge byte-identical to a
    single-process run; verified via 12-sample ChartQA smoke (46 records,
    0 field diffs, summary equal).
  - `scripts/_phase1_post_baseline_parallel.sh` — 5-stage orchestrator
    using GPUs 0/1/2:
    - **S1** TallyQA Main sharded 3-way (~3.5h vs ~10h single-GPU).
    - **S2** chart_base+math_base on GPU0, calib_plotqa on GPU1,
      calib_infovqa on GPU2 — all in parallel (~1h dominated by calib_plotqa).
    - **S3** SVD pooled subspace (CPU/fast).
    - **S4** 5 sweep-subspace cells distributed by cost: tally→GPU0,
      plotqa+chartqa→GPU1, infovqa+mathvista→GPU2 (~1.5h).
    - **S5** CPU finalization: recompute_confidence ×5 in parallel,
      per_cell ×5 sequential, confidence anchoring, 5-dataset summary.
  - All steps idempotent; safe to re-run after pod restart.
  - Wall-clock ~6-7h end-to-end vs ~12h sequential.
  - Legacy `_phase1_post_baseline.sh` retained (single-GPU, CUDA_VISIBLE_DEVICES=1
    hard-coded) for fallback.

  **v2 sharding extension** (commit `7433ee3`): `e6_steering_vector.py`
  calibrate-subspace + sweep-subspace phases now support sharding via the
  same `--shard-idx/--num-shards/--output-dir` flags. New drivers
  `run_calibrate_subspace_sharded.py` (K shards → torch.cat D_wrong/D_all
  → canonical SVD) and `run_sweep_subspace_sharded.py` (K shards →
  predictions.jsonl concat + dedup). Calibrate sharding is statistically
  equivalent (not byte-identical: round-robin slicing + per-shard cap
  produces at most num_shards-1 extra correct-base rows; D_wrong identical
  when wrong-base eligibility exhausted before cap). Top-K SVD basis
  robust to such perturbation; alignment verifiable via
  `check_subspace_alignment.py` (principal-angle cosines, threshold 0.99).
  Sweep sharding is byte-identical (greedy decoding + streaming records).
  `_phase1_post_baseline_parallel_v2.sh` orchestrates the full sharded
  pipeline (tally + calibs + sweep_tally); ~5h end-to-end. Skips Stage 1
  via `cell_done` if v1 has produced merged tally predictions.

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
