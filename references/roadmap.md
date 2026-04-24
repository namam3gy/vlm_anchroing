# references/roadmap.md — Cross-modal Anchoring in VLMs

**Single source of truth.** Read this first before any work in `vlm_anchroing/`. Update §3 (status), §5/§6 (checklists), and §10 (changelog) at the end of every task. Do **not** duplicate this content into other docs — link to sections of this file instead.

- **Target venue:** EMNLP 2026 Main (current scope per `references/project.md` is Findings-tier; the roadmap below is built around the Main-tier moves the plan recommends).
- **Compute envelope:** 8×H200 (one shared with a vLLM Qwen2.5-32B server → ~60 GB usable per GPU), one month.
- **Research-plan companion:** `references/project.md` is the candid feasibility review. Treat it as the prior; this file is the operational plan that follows from it.

---

## 1. Research definition

**Phenomenon.** A VLM is given a numerical VQA question + the target image. We *also* hand it an irrelevant second image that contains a rendered digit (the **anchor**). Does the anchor systematically pull the model's numeric answer?

**Three within-sample conditions** (`build_conditions` in `data.py`):

| Condition | Inputs | Role |
|---|---|---|
| `target_only` | target image | baseline accuracy |
| `target_plus_irrelevant_neutral` | target + irrelevant image with **no** digits | controls for "second image distracts" |
| `target_plus_irrelevant_number` | target + image containing one digit | the anchor manipulation |

The neutral arm separates **anchoring** from **mere distraction**. Anything visible in `number > neutral` is the anchoring signal.

**Why it matters.** No published work delivers a stand-alone rendered-number image as a cross-modal anchor in a multi-image VLM prompt and measures regression-style shift toward the anchor on open numerical VQA. The closest neighbours (VLMBias, Typographic Attacks, FigStep, Tinted Frames, the LLM-anchoring lineage) each differ on at least one core axis. See §1 of `references/project.md` for the full novelty matrix.

## 2. Hypotheses (anchored to falsifiable predictions)

| ID | Hypothesis | Falsifier | Evidence so far |
|---|---|---|---|
| **H1** | An anchor digit pulls the prediction beyond the neutral-distractor baseline. | `direction_follow_rate(number)` ≤ chance(0.5) **and** `mean_distance_to_anchor(number)` ≈ random pairing baseline. | ✅ all 7 models in main run: direction-follow ∈ [0.247, 0.348], adoption ∈ [0.110, 0.140]. |
| **H2** | Anchoring is **asymmetric**: stronger on items the model originally got wrong (= subjective uncertainty proxy). | Stratify pairs by `target_only`-correctness; expect adoption_wrong > adoption_correct. If equal, H2 fails. | ✅ **Refined.** A1 (Phase A): adoption gap is ≈ 0 across all 7 models, but the *graded* `moved_closer_rate` gap is **+6.9 to +19.6 pp** wrong > correct in every model. The bias is uncertainty-modulated *graded pull*, not categorical capture. See `docs/insights/A1-asymmetric-on-wrong.md`. |
| **H3** | Vision-encoder family modulates susceptibility. ConvNeXt/encoder-free should be *less* susceptible than CLIP/SigLIP-ViT (typographic-attack inheritance). | If ConvLLaVA / EVE / DINO-VLM show statistically equivalent direction-follow gap to CLIP-ViT VLMs, H3 fails. | ⚠️ **Pilot (2026-04-24) does not support the simple form.** ConvLLaVA-7B `adoption=0.156` falls inside the CLIP/SigLIP cluster CI (LLaVA-1.5 = 0.181). See `docs/experiments/E2-pilot-results.md`. **Replaced by H6 (below).** |
| **H6** | Cross-modal failures separate into two orthogonal axes: **anchor-pull** (uncertainty-modulated, encoder-mediated) and **multi-image distraction** (encoder-architecture-mediated, hits accuracy without encoding the anchor). Different encoder families sit at different points on this 2D plane. | If `adoption_rate` and `acc_drop_vs_target_only` are perfectly correlated (single failure mode), H6 fails. | ✅ **Strongly suggested by pilot.** InternVL3 = high acc_drop / low adoption; LLaVA-1.5 = low acc_drop / high adoption; ConvLLaVA = both. The two-axis decoupling is the new headline candidate. Needs full-run CIs to confirm. |
| **H4** | "Thinking" / instruction-tuned reasoning reduces anchoring (System-2 suppression). | Same model family with vs. without reasoning trace shows equal direction-follow. | ❌ Untested. *Note*: VLMBias + LRM-judging literature shows reasoning can *amplify* some biases — write the experiment to be agnostic to direction. |
| **H5** | Strengthening the prompt ("output a number, do not hedge") increases anchor pull on uncertain items but also induces large-number hallucination on others. | Compare `experiment_anchor_strengthen_prompt` vs `experiment` per item. | ⚠️ Suggestive: `mean_distance_to_anchor` in strengthen run reaches 2617 for gemma3-27b-it (vs. 4.4 in the same model's standard run) → model fabricates huge numbers under pressure. Needs proper analysis. |

## 3. Status — what's been run

### 3.1 Completed full runs (17,730 samples each)

| Experiment | Models | Output dir |
|---|---|---|
| `experiment` (standard prompt, VQAv2 number subset) | gemma3-27b-it, gemma4-31b-it, gemma4-e4b, llava-next-interleaved-7b, qwen2.5-vl-7b-instruct, qwen3-vl-8b-instruct, qwen3-vl-30b-it | `outputs/experiment/<model>/<run>/` |
| `experiment_anchor_strengthen_prompt` (no-hedging system prompt) | same 7 | `outputs/experiment_anchor_strengthen_prompt/<model>/<run>/` |

Per-record fields available (see `predictions.jsonl` keys): `condition`, `irrelevant_type`, `anchor_value`, `prediction`, `ground_truth`, `standard_vqa_accuracy`, `exact_match`, `anchor_adopted`, `anchor_direction_followed`, `numeric_distance_to_anchor`. Per-token logits also captured (commit `5f925b2`).

### 3.2 Smoke runs only (50 samples)

| Experiment | Models | Output dir |
|---|---|---|
| `experiment_tallyqa` | qwen2.5-vl-7b-instruct, qwen3-vl-8b-instruct, llava-next-interleaved-7b | `outputs/experiment_tallyqa/` |
| `experiment_chartqa` | qwen2.5-vl-7b-instruct (only) | `outputs/experiment_chartqa/` |
| `experiment_mathvista` | qwen2.5-vl-7b-instruct (only) | `outputs/experiment_mathvista/` |

### 3.3 Models integrated but not yet in any full run

These are wired into `build_runner` and pass the smoke test. Pilot (`outputs/experiment_encoder_pilot/`, 1,125 sample-instances each, 2026-04-24) is complete; full 17,730 runs queued pending user signoff:

- `llava-1.5-7b` — CLIP-ViT vanilla baseline. Pilot adoption=0.181 (highest of 11 models).
- `internvl3-8b` — InternViT family. Pilot adoption=0.066 (lowest), acc_drop=0.355 (highest). The "distraction-not-anchoring" outlier.
- `fastvlm-7b` — Apple FastViT. Pilot acc=0.483 (low — verify parse-rescue rate).
- `convllava-7b` — pure ConvNeXt encoder. Pilot adoption=0.156 (close to LLaVA-1.5; H3 in simple form not supported).

### 3.4 Headline numbers from completed runs

Standard prompt, VQAv2 number subset, 17,730 samples per model. `direction_follow` is the rate at which the model's prediction moves toward the anchor relative to its `target_only` answer.

| Model | acc(target_only) | acc(neutral) | acc(number) | adoption(number) | direction_follow(number) | mean dist→anchor |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.505 | 0.541 | 0.123 | 0.320 | 4.17 |
| llava-interleave-7b | 0.619 | 0.577 | 0.576 | 0.134 | 0.348 | 3.20 |
| gemma3-27b-it | 0.628 | 0.623 | 0.633 | 0.141 | 0.305 | 4.45 |
| qwen2.5-vl-7b | 0.736 | 0.708 | 0.711 | 0.110 | 0.248 | 5.03 |
| qwen3-vl-8b | 0.751 | 0.709 | 0.715 | 0.127 | 0.258 | 3.43 |
| gemma4-31b-it | 0.749 | 0.723 | 0.741 | 0.116 | 0.239 | 6.16 |
| qwen3-vl-30b-it | 0.759 | 0.709 | 0.707 | 0.120 | 0.280 | 3.70 |

Two patterns visible without further work:
1. **Direction-follow ≈ 24–35 %** even though adoption (exact match to anchor) is only 11–14 %. The bias is *gradient*, not categorical — most cases are "pulled toward" the anchor, not "set to" the anchor. This is exactly the signature an anchoring effect should leave.
2. **Strong models anchor *less* on direction-follow** but accept the anchor digit at similar rates. Counter to Lou & Sun (2024) "stronger LLMs anchor more" — needs a careful per-pair re-analysis before claiming a finding.

### 3.5 Strengthen-prompt anomaly (preliminary)

Under `experiment_anchor_strengthen_prompt`, three Gemma models show pathological `mean_distance_to_anchor`: gemma3-27b-it 2617.13, qwen2.5-vl-7b 1519.91, gemma4-31b-it 511.75 (vs. 3–6 in the standard prompt). The "must output a number" instruction is inducing large-number hallucination, *not* anchor adoption. Filter or report with a robust statistic before using these runs for any quantitative claim.

## 4. Experiment configuration (current defaults)

- VQAv2 number subset: `answer_range=8`, `samples_per_answer=400`, `require_single_numeric_gt=True` → 17,730 sample-instances per model.
- 5 irrelevant sets per question (one number-image and one neutral-image per set, each anchor digit ∈ {0..9} sampled).
- Greedy decoding, `max_new_tokens=8`.
- JSON-only system prompt (`{"result": <number>}`). The strengthen variant adds explicit "no hedging" instructions.
- Seed 42.

## 5. Phase A — deep re-analysis of existing data (no new compute)

These extract evidence already present in `outputs/experiment/` and `outputs/experiment_anchor_strengthen_prompt/`. Each is one analysis script + one short insight markdown under `docs/insights/`.

| ID | Question | Output | Status |
|---|---|---|---|
| **A1** | **Asymmetric anchoring on wrong cases.** Stratify pairs by `target_only` correctness; compute adoption / direction-follow / pull magnitude separately for originally-correct vs. originally-wrong. Report per model + pooled. This is the H2 test and the paper's strongest hook. | `docs/insights/A1-asymmetric-on-wrong.md` | ✅ Done — adoption symmetric, graded `moved_closer_rate` +6.9–19.6 pp |
| **A2** | **Per-anchor-value pull.** For each anchor digit 0–9, compute mean signed shift `pred(number) − pred(target_only)` and adoption rate. Are some digits stickier (0, round numbers, model-frequent priors)? Plot pull vs. anchor. | `docs/insights/A2-per-anchor-digit.md` | ✅ Done — anchors 1/2/4 sticky, 7/8 inert; LLaVA × anchor=2 = 0.30; Qwen3-VL anti-anchoring on digits 3/6/7 |
| **A3** | **Question-type stratification.** Use the existing `question_type` column ("how many", "what number", etc.). Does anchoring differ across question forms? Note: VQAv2's question_type is coarse — confirm taxonomy first. | folded into `00-summary.md` (no signal at this granularity) | ✅ Done (negative — defer until ChartQA/TallyQA data) |
| **A4** | **Per-pair shift distribution.** Move beyond means: histogram of `pred(number) − pred(target_only)`. Where is the mass? Bimodal (full adoption + no change) or graded? This decides whether H1 reads as "discrete capture" or "graded pull". | folded into `00-summary.md` | ✅ Done — strongly bimodal (≥75% no change) + thin pull-toward-anchor tail |
| **A5** | **Strengthen vs. standard prompt.** Per item, compare both prompts. Test H5: does the strengthen prompt amplify *anchor-driven* shifts on uncertain items, or does it just inflate hallucination? Use median + IQR, not means, given the §3.5 outliers. | folded into `00-summary.md` | ✅ Done — only gemma3-27b moves substantially (+17.4 pp adoption); not universal |
| **A6** | **Failure-mode taxonomy.** Bucket each `number`-condition prediction into: (a) exact anchor copy, (b) graded pull toward anchor, (c) unchanged from `target_only`, (d) anchor-orthogonal hallucination, (e) refusal/non-numeric. Report per model. | folded into `00-summary.md` | ✅ Done — `_data/A6_failure_modes.csv` |
| **A7** | **Cross-model item agreement.** For the items a model anchors on, do other models also anchor? Build a per-item susceptibility score, correlate across models. If susceptibility is item-driven (high cross-model correlation), the bias is *content*-mediated; if model-driven (low correlation), it's *parameter*-mediated — directly informs H3. | `docs/insights/A7-cross-model-agreement.md` | ✅ Done — Spearman ρ ∈ [0.15, 0.31]; partly content-driven; same-family Qwen3-VL pair = 0.30 |

**Deliverable for Phase A:** four insight files (`00-summary.md`, `A1-…`, `A2-…`, `A7-…`) + numeric artifacts under `docs/insights/_data/`. A3/A4/A5/A6 are folded into the summary because the underlying signal didn't earn its own writeup (negative or supporting). All 7 analyses ran from a single script: `scripts/phase_a_data_mining.py`.

## 6. Phase B onward — new experiments

### Tier 1 (Main-tier acceptance lever, per `references/project.md`)

| ID | Experiment | Compute estimate | Why it matters | Status |
|---|---|---|---|---|
| **E1** | **Attention-mass analysis.** `output_attentions=True` on a stratified subset. Compute attention to anchor-image-tokens vs. target-image-tokens vs. text-tokens, per layer, per condition. | Hours per model on existing 7. | Directly answers reviewer "why does this happen?". Cheapest mechanistic move. | ✅ **6-model panel + per-layer localisation done** (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b; n=200 each). Three E1 claims settled; four E1b archetypes identified: SigLIP-Gemma early+large (L5, δ +0.050, text-stealing); mid-stack cluster of CLIP-ViT+InternViT+ConvNeXt (L14–16, δ ~+0.020, text-stealing); Qwen-ViT late+moderate (L22, δ +0.015, target-stealing); FastViT late+large+strongest-A7 (L22, δ +0.047, text-stealing, A7 gap +0.086 with n=75 caveat). **H3 "ConvNeXt < ViT" definitively falsified** — ConvLLaVA replicates LLaVA-1.5 exactly. Writeups: `docs/experiments/E1-preliminary-results.md` (E1 4-model original), `docs/experiments/E1b-per-layer-localisation.md` (E1b full 6-model writeup) + `docs/insights/E1b-per-layer-localisation.md` (distilled insight). Remaining: head-level sparsity, causal test. |
| **E2+E3** (combined) | **Full 17,730 grid for all 4 newly-integrated models** (ConvLLaVA, LLaVA-1.5, InternVL3, FastVLM). Pilot at 1,125 done — see `docs/experiments/E2-pilot-results.md`. The full run (a) tightens CIs around the H6 two-axis hypothesis, (b) captures per-token logits enabling deferred A1 logit-margin re-analysis, (c) gives 11-model panel for the paper. | ~1 day per 7B model on H200; 4 models in sequence. | Resolves H6 two-axis hypothesis at scale. | **⏸ Deferred** (user 2026-04-24): proceeding with pilot data only; E1 attention + mitigation prioritised first. Re-evaluate after E1 results inform whether full 4-model panel still needed. |
| **E4** | **Mitigation prototype.** Pick the simplest intervention that the E1 attention analysis suggests — most likely contrastive decoding on number-image vs. no-image, or attention re-weighting that down-scales anchor-image tokens. Target: ≥ 10 % reduction in direction-follow with ≤ 2 pp accuracy drop. | A few days; depends on E1. | The single most reliable Findings → Main lever. | ☐ |

### Tier 2 (paper hardening)

| ID | Experiment | Notes | Status |
|---|---|---|---|
| **E5** | **Multi-dataset full runs.** TallyQA + ChartQA at full scale (current 50-sample smoke is not enough). MathVista as a stretch goal. | TallyQA is the cleanest counting domain; ChartQA gives an in-image-number conflict (especially compelling — anchor competes with a legible number in the target image). | ☐ |
| **E6** | **Closed-model subset.** GPT-4o or Gemini-2.5 on a ~500-sample stratified slice. Defuses the "open-only" reviewer complaint. | Token cost only. | ☐ |
| **E7** | **Paraphrase robustness.** 3–5 question-prompt rephrasings × bootstrap CI × multiple-comparison correction. | Required before claiming any per-model effect. | ☐ |
| **E8** | **Position effect.** Anchor image as image[0] vs. image[1]. Some VLMs are known to weight images positionally. | Can be sub-experiment of E2/E3. | ☐ |

### Tier 3 (optional, only after Tiers 1–2)

| ID | Experiment | Notes | Status |
|---|---|---|---|
| E9 | Anchor-value range sweep beyond 0–9 (10s, 100s). | Tests whether the bias scales with anchor magnitude or saturates. | ☐ |
| E10 | Layer-wise logit lens — *when* the anchor enters the prediction. | Complements E1. | ☐ |
| E11 | Human baseline (~50 Prolific participants on a small condition matrix). | Disproportionately credibility-positive for psychology-framed papers. | ☐ |
| E12 | Thinking-VLM comparison on a model that supports both modes (Qwen3-VL with/without reasoning). | The H4 test. | ☐ |

## 7. Ordering and decision points

1. **Phase A (done)** — Phase-A insights settled (`docs/insights/`). Output: A1 confirmed H2 in graded form; A2/A7 informed mechanism plan.
2. **Phase B (current)** — pilot done, E2+E3 full runs deferred per user 2026-04-24. Active sequence: **E1 (attention mass on existing 7 models) → E4 (mitigation prototype) → re-evaluate full 4-model run**. Pilot already provides enough H6 evidence to prototype mitigation; only fall back to E2+E3 full grid if E1 attention can't separate the two failure modes mechanistically.
3. **Phase C** — Tier-2 hardening (E5 multi-dataset, E7 paraphrase robustness, E6 closed-model subset) once E1+E4 produce a publishable result.
4. **Phase D** — write-up. ARR May 25 deadline (per `references/project.md` §"realistic one-month plan").

**Decision triggers** (write the answer down when the trigger fires):
- After A1: is the asymmetry real and large (≥ 10 pp gap)? If no, fall back to either H3 or H5 as headline.
- After E1: does the anchor draw disproportionate attention? If yes → E4 is straightforward; if no → mitigation must target the LLM side instead.
- After E2: does ConvLLaVA show meaningfully lower direction-follow? If yes, H3 becomes a paper section; if no, drop H3 and consolidate around H2 + mechanism.

## 8. File-system conventions for research artifacts

Created lazily — only when the first artifact of that type is written.

```
research/
  insights/                    # phase-A re-analysis outputs (one md per insight)
    00-summary.md
    A1-asymmetric-on-wrong.md
    ...
  experiments/                 # plans + result writeups for E1..E12
    E1-attention-mass.md
    E2-convllava-full.md
    ...
  scripts/                     # one-off analysis scripts that produce insights
                               # reusable scripts go to ../scripts/
```

This roadmap stays at project root and is the **only** doc that lists status across the whole program. Insight and experiment docs each focus on one thing.

## 9. Known caveats (carry these into every analysis)

- **Strengthen-prompt distance outliers** (§3.5) — robustly trim or use medians.
- **Anchor digit ∈ 0–9 vs. answer support 0–8** — the anchor distribution is wider than the GT distribution. When computing "moved toward anchor", control for the fact that anchors 9 can never be the correct answer in this subset.
- **Broken VQA image** (`inputs/vqav2_number_val/images/000000000136.jpg`) — file body is filesystem garbage, not a JPEG. The loader (`vlm_anchor.data.load_number_vqa_samples`) now calls `Image.verify()` and silently skips undecodable files, so it no longer crashes runs. The questions.jsonl entry referencing that image_id is now dropped on load (one fewer sample).
- **`fastvlm-7b` prose outputs** — the model often emits prose despite the JSON-only prompt. `extract_first_number` rescues most cases but the parse-failure rate is non-zero; report it explicitly.
- **Shared GPU** — same machine runs a vLLM `Qwen2.5-32B` server on port 8000 (~55 % VRAM). Effective per-GPU budget for this project ≈ 60 GB.
- **Citation hygiene** — `references/project.md` flags some 2026 arXiv IDs that may not resolve. Verify each cite before any submission.

## 10. Changelog

- **2026-04-24** — Roadmap created. Status reflects: 7 models × full VQAv2 (standard + strengthen prompts) done; 5 new models integrated but not yet in main runs; 3 dataset extensions at smoke-only. Phase A queued.
- **2026-04-24** — Phase A complete. Headline (H2): anchoring is uncertainty-modulated **graded pull**, not categorical capture (`docs/insights/A1-asymmetric-on-wrong.md`). Per-digit asymmetry confirmed (A2). Cross-model correlations 0.15–0.31 (A7) → both encoder and content matter, motivating E1+E2. A3/A4/A5/A6 folded into `00-summary.md`. Decision triggers in §7 fired — Phase B order unchanged.
- **2026-04-24** — E2 pilot (n=1,125 × 4 models) complete. **H3 in simple "Conv < ViT" form not supported** — ConvLLaVA adoption=0.156 falls inside the CLIP/SigLIP cluster CI. **New H6 added**: cross-modal failures decompose into two orthogonal axes (anchor-pull vs. multi-image distraction). InternVL3 = pure distraction (low adoption, high acc_drop), LLaVA-1.5 = pure anchoring (high adoption, low acc_drop), ConvLLaVA = both. Two-axis framing replaces "encoder family universally matters" as the candidate paper headline. See `docs/experiments/E2-pilot-results.md`. Full 17,730 runs for all 4 models queued, awaiting user signoff.
- **2026-04-24** — Bug fix: `vlm_anchor.data.load_number_vqa_samples` now calls `Image.verify()` and silently skips undecodable images. Prevents the `000000000136.jpg` PIL crash from killing future multi-day runs.
- **2026-04-24** — User decision (option 2): defer E2+E3 full 4-model run; prioritise E1 attention extraction + E4 mitigation. Pilot data + Phase A is sufficient to prototype mitigation. Re-open E2+E3 only if E1 cannot mechanistically separate anchor-pull from multi-image distraction. Phase B sequence in §7 updated.
- **2026-04-24** — Bilingual docs convention adopted. Every md under references/roadmap.md or research/ now has a `_ko.md` Korean mirror. English `.md` is canonical (Claude reads/edits it first); Korean version updated in lockstep. Memory entry: `feedback_bilingual_docs.md`.
- **2026-04-24** — E1 extended to 4 encoder families (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b; n=200 each). **Three claims settled at 4-model scale:** (i) anchor>neutral attention robust 4/4 (answer-step mean +0.004 to +0.007, CI excludes 0); (ii) H2 `wrong>correct` attention asymmetry falsified 4/4 — uncertainty does not modulate mean anchor attention; (iii) A7 `susceptible>resistant` holds 3/4 at answer step, inverts in Gemma-SigLIP (which also concentrates signal at step 0, consistent with typographic-attack inheritance). Candidate 3-claim paper structure emerges: anchor notice (attention) is robust; anchor pull (behaviour) is encoder-modulated; uncertainty modulates pull (Phase A) but not attention. `docs/experiments/E1-preliminary-results.md`.
- **2026-04-24** — **E1b per-layer localisation done** (same 4 models × n=200). Peak layer differs sharply by encoder family: SigLIP-Gemma **layer 5/42** (12 % depth, δ +0.050, spike flanked by anchor/target trade-off layers), Qwen-ViT **layer 22/28** (82 %, δ +0.015, A7 gap +0.025 with bottom-decile CI including zero), CLIP-ViT (LLaVA-1.5) **layer 16/32** and InternViT (InternVL3) **layer 14/28** (both mid, δ ~+0.019). Layer-averaged E1 numbers were hiding a ~3× concentration at a single layer. **Second axis — budget decomposition:** at peak, Gemma/LLaVA-1.5/InternVL3 pull anchor mass from *text* (δ_text −0.014 to −0.038), Qwen pulls from *target image* (−0.010, text −0.005). Two distinct mechanisms: text-stealing vs target-stealing. **Candidate E4 intervention sites per family, to be tested:** Gemma → input-side pre-layer-5 KV/projection patch (denies text→anchor pull); Qwen → late-stack anchor attention re-weighting layer 22±2 gated by susceptibility (returns mass to target); CLIP/Intern → mid-stack ~14–16 (returns mass to text — less ideal, still testable). These are observational conjectures; E4 will test whether any of them actually reduce `direction_follow`. `docs/experiments/E1b-per-layer-localisation.md` (detailed) + `docs/insights/E1b-per-layer-localisation.md` (distilled).
- **2026-04-24** — **E1 inputs_embeds-path extension done; 6-model panel complete.** Added ConvLLaVA (ConvNeXt encoder, inputs_embeds generate path) and FastVLM (FastViT, -200-marker expansion path) to the attention extraction pipeline in `scripts/extract_attention_mass.py` via new `EagerConvLLaVARunner` / `EagerFastVLMRunner` subclasses. Full n=200 runs complete for both. **Two key new findings:** (i) **H3 "ConvNeXt < ViT" is definitively falsified at the per-layer level** — ConvLLaVA's peak layer is L16 (same as LLaVA-1.5), mechanism is text-stealing (identical), magnitude +0.022 (within 20 % of LLaVA-1.5). Three encoders (CLIP-ViT, InternViT, ConvNeXt) now form a tight "mid-stack text-stealing" cluster. (ii) **FastVLM is a new archetype:** late peak (L22, matching Qwen depth) + text-stealing budget (−0.034, matching Gemma kind) + Gemma-level magnitude (+0.047) + panel-largest A7 gap (+0.086, with n=75 and wide CI caveat). Two published VLM failure modes — typographic attack and anchor-vs-target budget confusion — appear to co-fire in FastVLM. The 3-archetype story (from the 4-model E1b) refines to 4 archetypes. E4 design can now proceed with per-family intervention sites; the mid-stack cluster is the highest-leverage target (one intervention could generalise to three encoders). See `docs/experiments/E1b-per-layer-localisation.md` for the updated 6-model panel.
