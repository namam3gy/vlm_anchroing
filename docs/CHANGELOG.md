# CHANGELOG — vlm_anchroing

> Older entries (pre-2026-04-28 second-pass) extracted from `references/roadmap.md §10`
> on 2026-04-29 to keep the roadmap lightweight. For the most recent ~5 entries,
> see `references/roadmap.md §10`.

- **2026-04-29 (references / docs slim-down + ko-mirror catch-up).**
  Session-start load was carrying ~1072 lines across `references/project.md`
  (209) + `references/roadmap.md` (863); cut to ~718 lines (~33 %) by
  splitting two heavy sub-trees out of the roadmap.
  (i) **roadmap §3.3 headline tables** (7-model VQAv2 main panel · E5b
  distance · E5c digit-pixel · E5e ChartQA+TallyQA · E1d / E4 mechanism
  summary, ~97 lines) extracted to
  `docs/insights/headline-numbers.md`; roadmap §3.3 now carries a 5-bullet
  orientation + pointer.
  (ii) **roadmap §10 changelog** (~475 lines / 40+ entries) compressed
  to 20 entries via 2026-04-24 / -25 / -26 / -27 cluster collapse
  (E1+E1b 6-model panel + H3 retired + Phase A all into one paragraph;
  E4 Phase 1 + Phase 2 + paired-anchor-damage retraction into one;
  E5b/c/M1/E5-ChartQA cluster into two), then the older 15 entries
  moved to **`docs/CHANGELOG.md`** with the most recent 5 left inline.
  (iii) **`project.md §0.4`** "E5b/E5c cross-model expansion in flight"
  → "qwen2.5-vl-7b 2/3 panel landed 2026-04-29; gemma3-27b-it remaining
  gap" (cross-model E5c expansion landed earlier this session).
  (iv) **`roadmap.md §3.1` row 1 + §3.3 footnote + §9 in-flight bullet**
  — stale post-M2 / post-C-form / post-γ flags refreshed.
  (v) **`roadmap.md §10` new 2026-04-29 doc-hygiene entry** captures
  paper-deck relocation (`docs/figures/` → `docs/ppt/`),
  `*.pptx` regenerable git-untracking, ko-mirror cache cleanup,
  Status-banner backfills.
  (vi) **ko mirrors** (`references/project_ko.md`, `roadmap_ko.md`) —
  structurally frozen at the pre-2026-04-28 paper-section-anchored
  restructure (project_ko has no §0; roadmap_ko has §3=Status / §6=Phase B
  shape). Headers updated to point to EN canonical for current paper
  outline; roadmap_ko §10 received 11 compressed Korean entries
  covering 2026-04-28 → 2026-04-29 (M2, C-form, γ-β thinking-amplifies,
  B안 propagation, paper §3 / §7.4 / §8 prose, L1 confidence, γ-α
  MathVista, E1-patch POC + 4-model extension, qwen2.5-vl E5c,
  gemma3-27b TallyQA, doc-hygiene). §1–§9 ko bodies left untouched —
  full structural re-sync deferred.
  No findings or numbers changed; doc-tree slim-down only.

- **2026-04-28 (overnight polish — number / citation / status audit).**
  Two-pass non-GPU polish session while gemma3-27b TallyQA E5e was on
  GPU. Outcomes (state now reflected in §3.x / §6.x / §7 / §9 / paper
  draft): (i) **Phase A `_data/A1_*.csv` regenerated** —
  `phase_a_data_mining.py::_resolve_model_runs` was alphabetically
  picking a 45-record qwen2.5-vl smoke run over the canonical 53,190-row
  full run; switched to "largest run with n ≥ 100", A1-A7 CSVs include
  all 7 models. (ii) **Paper §5.2 / §5.3 / §5.4 / §5.5 / §5.6 / §6 / §7
  numbers cross-checked** against `_data/A1_*`, `_data/E5b_*`,
  `_data/E5c_*`, `_data/L1_*`, E4 evidence — all verified or
  rewritten. §5.5 had 8/12 stale `df(a)/df(m)` cells (pre-C-form
  source); rewritten from `outputs/experiment_e5e_*_full/*/summary.json`.
  (iii) **§5.2 metric note** — moved-closer rate (pull-form) and
  C-form `direction_follow_rate` magnitudes differ by 0.5-2.6 pp on
  the 7-model panel; paper retains pull-form in §5.2 (Phase A's
  original definition), C-form in §3.3 / §5.1. (iv) **§3 prose C-form
  alignment** — `paper-section-3-problem-definition.md` (pre-C-form)
  carried the old `(pb−gt)·(pa−gt)` numerator + buggy
  "categorical-replace regime → df = 0 on MathVista" reading; both
  superseded with retraction note. (v) **§2 citation audit** — six
  arXiv IDs verified, three corrections (`2507.03123` AIpsych author
  Jin → Liu; `2508.20570` description Hufe et al. "Dyslexify" later-half
  CLIP attention-head circuit; Wang-Zhao-Larson NAACL 2025 added as
  `2502.08193`, replacing wrong `2508.20570` conflation). (vi) **§7
  priority queue refresh** — struck five landed P0s (M2-refactor,
  L1-L4, γ-α, γ-β, M2-tests, paper §3 / §7.4 / §8 prose); rebuilt
  around actually-blocking items. (vii) **γ-β instruct numbers fixed
  in three roadmap surfaces** — pre-C-form `adopt(a)=0.055 / df=0.094`
  → real `0.074 / 0.102`; ratios ×1.6 / ×2.9 (paper had been correct,
  roadmap was stale). (viii) `configs/experiment_e5c_*.yaml` +
  `scripts/run_e5b_e5c_cross_model_chain.sh` pre-staged for the
  qwen2.5-vl + gemma3-27b E5c launches.

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
  PPTX deck `docs/ppt/C_form_migration_report.pptx`). **All
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

- **2026-04-28 (paper-section prose for §3 / §7.4 / §8/F1).** First-draft
  paper-ready writeups under `docs/insights/`:
  `paper-section-3-problem-definition.md` (4-cond setup +
  JSON-strict prompt + four canonical metrics);
  `paper-section-7-4-mitigation-free-lunch.md` (Phase 2 free-lunch
  table — df↓ -5.8 to -17.7 % rel · em(a)↑ +0.49 to +1.30 pp ·
  em(b) invariant on 3/3 mid-stack-cluster · 10-22 % anchor-damage
  recovery with three caveats);
  `paper-section-8-f1-future-work.md` (LLM-vs-VLM architectural-diff
  paragraph, three named outcome patterns A/B/C). Roadmap §6.1 /
  §6.5 / §6.6 / §7 P2 status flips landed in same wave.
  Paper-summary deck pipeline (`scripts/build_paper_figures.py`,
  `build_paper_pptx.js`, four `paper_*.png`, speaker-notes
  `paper_summary_slides.md`) shipped alongside; heavy regenerable
  artefacts (`paper_summary.pdf`, `*.pptx`, `slide-*.jpg`) gitignored.

- **2026-04-28 (E1-patch POC, 2 archetypes — superseded 2026-04-29).**
  Driver `scripts/compute_anchor_digit_bboxes.py` + extended
  `extract_attention_mass.py --bbox-file` + analysis
  `analyze_attention_patch.py`. Headline: gemma4-e4b digit/anchor =
  0.631 (peak L9, +0.404 above fair share); llava-1.5-7b 0.468
  (peak L7, +0.241). Two qualitatively different attention pathways
  (Gemma globally typographic-inheritor vs LLaVA mid-stack-cluster).
  Superseded by the 2026-04-29 4-model perfect-square panel entry
  (top of changelog). Artefacts: `docs/insights/E1-patch-evidence.md`,
  `_data/E1_patch_*.csv`, `inputs/irrelevant_number_bboxes.json`.

- **2026-04-28 (L1 confidence-modulated anchoring evidence — paper §6).**
  `scripts/analyze_confidence_anchoring.py` on 10 logit-capturing runs
  (post-`5f925b2`: E5b / E5c / E5d / E5e), 112,008 (sample × arm)
  records over 34 cells. **`entropy_top_k` is the cleanest of three
  proxies tested.** Mean df Q4-Q1 = +0.128 (+0.152 post-C-form
  refresh); 18/34 anchor cells fully monotone Q1<Q2<Q3<Q4. Worked
  example (E5c VQAv2 S1, llava-interleave): Q1 most-confident →
  adopt 0.077 / df 0.040; Q4 most-uncertain → 0.147 / 0.113. Phase A's
  wrong/correct binary is a coarse projection. Artefacts:
  `docs/insights/L1-confidence-modulation-evidence.md`,
  `_data/L1_*.csv`. VQAv2 main panel logit re-run (no logit pre-`5f925b2`)
  queued under §6.4.

- **2026-04-28 (E5e MathVista γ-α — 3-model 4-cond full).**
  `experiment_e5e_mathvista_full.yaml`, 3 models × 385 base × 4 cond
  = 4,620 records, ~45 min on H200. Headlines (M2 wrong-base S1
  anchor): gemma3-27b adopt(a) = 0.194; llava +0.041; qwen +0.007;
  3/3 preserve `a > m`. **Pre-C-form driver bug had reported
  `df_M2 = 0` universally on MathVista ("categorical-replace"
  framing); retracted post-refactor —** real C-form numbers in §3.3
  (gemma3-27b df(a) = 0.332). E5d MathVista C3 FAIL diagnosis was a
  llava-specific small-n behaviour. Evidence:
  `docs/insights/E5e-mathvista-evidence.md`. New analyzer
  `scripts/analyze_e5e_wrong_correct.py` writes
  `_data/experiment_e5e_mathvista_full_per_cell.csv`.

- **2026-04-28 (M2 landed).** `metrics.py` refactored:
  `anchor_adoption_rate` uses `D_paired = (pb != anchor)` denominator;
  new `anchor_direction_follow_rate` requires `pa != pb` in numerator;
  legacy fields kept for audit. Per-row M2 flags
  (`pred_b_equal_anchor`, `pred_diff_from_base`,
  `anchor_direction_followed_moved`) added.
  `reaggregate_paired_adoption.py --apply --force` rewrote 53 run
  dirs. 15 metrics tests pass; M2 status in §8 → ✅. Stale
  qwen2.5-vl smoke-only run dir (`20260427-075523/`, n=25) deleted.
  Same-day C-form refactor on top of M2 fixed the
  `direction_follow_rate` formula (separate entry above).

- **2026-04-28 (roadmap restructured to be paper-section-anchored).**
  Aligned with `references/project.md §0` paper outline: §1 4-cond
  canonical (b/a/m/d); §2 H7 logit-confidence monotonicity;
  §3 status matrix consolidates prior tables; §4 documents M2; §5
  closes Phase A; §6 maps experiments to paper sections; §7 priority
  queue; §8 M2 in pending refactors; §9 / §10 updated.

- **2026-04-27 (E5b / E5c stratified results — anchoring is
  uncertainty-modulated AND plausibility-windowed).** Stratified runs
  on llava-interleave-7b (n=1000 base × dataset, VQAv2 + TallyQA,
  5 anchor strata + 5 masked-anchor strata + 1 neutral). **Two
  compounding gates:** (i) uncertainty (correct-base ≤ 0.10 across
  all strata; wrong-base 1.4-37× larger); (ii) plausibility window
  — wrong-base adopt peaks at S1 (VQAv2 0.130 / TallyQA 0.092) and
  decays to floor by S5 (TallyQA S5 = 0/346 — implausible anchors
  fully rejected). E5c digit-mask gap (anchor − masked) wrong-base
  S1 = +6.1 pp VQAv2 / +2.5 pp TallyQA → digit pixel is the
  operative cause; anchor background ≈ generic 2-image distractor on
  correct-base. Same period: E5 ChartQA full run on 3-model panel
  (qwen2.5-vl / qwen3-vl-8b / llava-interleave, 16,170 records each)
  — df higher (0.230-0.394) and adopt lower (0.015-0.022) than VQAv2;
  em invariant — "tilt vs replace" decomposition driven by whether
  target image carries a competing legible number. Detailed
  per-experiment sub-entries collapsed 2026-04-29.

- **2026-04-27 (M1 paired adoption metric landed).** `evaluate_sample`
  requires `base_prediction`; `anchor_adopted = (base_pred ≠ anchor)
  AND (pred == anchor)` (commits `bbcc418..ce1928a`). Driver threads
  target_only's prediction into subsequent conditions.
  `reaggregate_paired_adoption.py` re-computed adoption on 54
  predictions.jsonl files (35 standard + 13 causal_ablation + 6
  e4_mitigation); raw predictions preserved. Paired-adoption rates
  0.019-0.059 (vs. marginal 0.110-0.141; ~75-90 % rel reduction).
  df + accuracy unchanged. Superseded by M2 + C-form refactor on
  2026-04-28.

- **2026-04-25/-26 (E4 mitigation — Phase 1 sweep + Phase 2
  cross-cluster).** Mid-stack-cluster (LLaVA-1.5 / ConvLLaVA /
  InternVL3) attention re-weighting at upper-half locus identified
  by E1d. **Phase 1 (n=200, 7 strengths × 3 cond):** all 3 hit
  ≥10 % rel df reduction with em flat or rising; per-model `s*`
  required (LLaVA -3.0 / ConvLLaVA -2.0 / InternVL3 -0.5 — order of
  magnitude apart). **Phase 2 (88,650 records / model, pull-form;
  C-form refresh in 2026-04-28 B안 entry):** LLaVA -17.7 % rel df,
  ConvLLaVA -10.6 %, InternVL3 -5.8 %; em rises +0.49 to +1.30 pp;
  em(target_only) invariant; paired anchor-damage -3.55 to -9.34 pp;
  recovery 10.2-21.7 %. **Anti-correlation:** lower df₀ → smaller
  relative reduction (InternVL3 lowest both). ConvLLaVA fluency-tail
  (mean_dist 2.99 → 53.54 on treated cell) noted for paper —
  switch to median + fluency-degraded count. Earlier "InternVL3 has
  no anchor damage" reading retracted via paired
  intersection-of-valid-cells analysis; `_paired_anchor_damage`
  integrated in `analyze_e4_mitigation.py`.

- **2026-04-25 (E1d causal anchor-attention ablation, 6-model panel,
  n=200).** Three findings: (i) **single-layer ablation null on
  6/6** at E1b peak AND at layer 0 (`Δ direction_follow ∈
  [-0.032, +0.020]` peak / `[-0.027, +0.005]` layer 0) —
  multi-layer redundancy confirmed; rules out "different single
  layer is causal site"; (ii) stack-wide ablation reduces df 11-22 pp
  universally but breaks fluency on 3/6; (iii) **upper-half attention
  ablation is the single architecture-blind mitigation locus** —
  reduces df on 6/6 (-5.5 to -11.5 pp pull-form, -4.0 to -10.5 pp
  C-form post-B안), fluency-clean on 4/6 (mid-stack cluster + Qwen).
  Mid-stack cluster identified as highest-leverage E4 prototype
  target. Sub-finding: ConvLLaVA + LLaVA-1.5 share E1b peak/mechanism
  but respond opposite to lower-half ablation (Δ -0.120 vs +0.165) —
  same-attention-signature ≠ same-causal-structure. Writeups:
  `docs/experiments/E1d-causal-ablation.md`,
  `docs/insights/E1d-causal-evidence.md`.

- **2026-04-24 (Phase A + E1 / E1b 6-model attention panel + H3
  retired + roadmap created).** **Phase A complete:** A1 graded-pull
  asymmetry +6.9 to +19.6 pp on 7/7 models
  (`docs/insights/00-summary.md`, A1 / A2 / A7); per-digit asymmetry;
  cross-model Spearman ρ 0.15-0.31. **E1 attention-mass 4-model
  panel** extended to 6 with ConvLLaVA (CLIP-ConvNeXt) + FastVLM
  (FastViT). Three E1 claims at 4-model scale: anchor>neutral
  attention 4/4; H2 wrong>correct attention asymmetry falsified 4/4;
  A7 susceptible>resistant 3/4 (Gemma inverts at step 0,
  typographic-attack inheritance). **E1b per-layer localisation —
  4 archetypes:** SigLIP-Gemma early L5 +0.050 (text-stealing);
  CLIP-ViT/InternViT/ConvNeXt mid-stack cluster L14-16 +0.020
  (text-stealing); Qwen-ViT late L22 +0.015 (target-stealing); FastVLM
  late L22 +0.047 (text-stealing, panel-largest A7 gap with n=75
  caveat). **H3 "ConvNeXt < ViT" definitively falsified** at both
  behavioural (E2 pilot adoption=0.156 inside CLIP/SigLIP CI) and
  per-layer level — ConvLLaVA precisely replicates LLaVA-1.5
  (`docs/insights/E1c-h3-falsified.md`). H6 added: cross-modal
  failure decomposes into anchor-pull + multi-image-distraction axes.
  E2+E3 full 4-model deferred per user; E1d / E4 prioritised.
  `data.load_number_vqa_samples` now skips undecodable images
  (`000000000136.jpg` PIL bug). Bilingual `_ko.md` mirror convention
  adopted (retired 2026-04-27, commit `84f9341`). Roadmap created
  same day with 7 main models full + 5 new models integrated +
  3 dataset extensions at smoke-only.
