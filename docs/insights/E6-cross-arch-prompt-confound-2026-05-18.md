# E6 cross-arch on Qwen2.5-VL — Phase 2 confound discovery: prompt-format dependence

**Date:** 2026-05-18.
**Branch:** `worktree-e6-cross-arch-qwen25vl` (PR #54).
**Status:** Phase 2 Stage-4 result on Qwen2.5-VL is *confounded* by response-format dependence; cross-arch claim **deferred** to Option K (raw-number prompt re-run on 3-model panel).
**Affected paper sections:** §6.4 (new cross-arch subsection), §8 Limitations "Mitigation scope". Plausibly §6 main result framing.

## Background

Phase 0 + Phase 1.5 confirmed cross-arch *recipe portability* on Qwen2.5-VL-7B-Instruct:
- Peak L\*_qwen = 26 (identical to OneVision)
- Phase 1.5 chosen cell L=26 K=8 α=1.0 (identical to OneVision; rank 1 with +0.92pp margin over rank 2)

Phase 2 Stage-4 5-dataset evaluation, however, produced near-zero mitigation magnitude:

| Dataset | n_paired | Δdf | Δem(b) | 95 % CI on Δdf |
|---|---:|---:|---:|---|
| TallyQA | 5000 | −0.04 pp | +0.80 pp | [−0.38, +0.30] |
| PlotQA | 925 | **+1.30 pp** ⚠️ | +0.54 pp | [−0.11, +2.70] |
| InfoVQA | 222 | −0.45 pp | −0.45 pp | [−2.70, +1.80] |
| ChartQA | 152 | −1.32 pp | +0.66 pp | [−3.29, 0.00] |
| MathVista | 139 | 0 pp | +0.72 pp | [−2.88, +2.88] |
| **mean** | | **−0.10 pp** | **+0.45 pp** | |

Compare OneVision Stage-4 mean Δdf = **−2.9 pp**; magnitude ratio ≈ **30×**. PlotQA even shows backfire trend. Bar-raiser conditional-Main accept (cross-arch instantiation positive) would NOT be met by this result alone.

## Diagnostic chain (B → D → C → E)

### B — Pilot α-sweep robustness (free analysis on existing Phase 1 data)

`docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv`. L26_K08 cells across α ∈ {0.5, 1.0, 2.0}:

| α | PlotQA Δdf | InfoVQA Δdf | pooled mean Δdf |
|---:|---:|---:|---:|
| 0.5 | −4.95 pp | −0.53 pp | −2.74 pp |
| **1.0 (chosen)** | **−4.95 pp** | **−2.42 pp** | **−3.68 pp** |
| 2.0 | **+11.78 pp** | **+8.90 pp** | **+10.34 pp** catastrophic backfire |

α=2.0 sign-flip implies narrow Goldilocks zone. Pilot dDF on chosen α=1.0 was decent (−3.68 pp pooled), so cell selection itself was not faulty.

### D — OneVision vs Qwen2.5-VL baseline-vs-mitigation identical-prediction ratio (free analysis on existing Stage-4 data)

| Dataset | OneVision identical-pred % | Qwen2.5-VL identical-pred % |
|---|---:|---:|
| MathVista | 21–24 % | **95–99 %** |
| ChartQA | 56–62 % | **97–100 %** |
| InfoVQA | 60–63 % | **96–99 %** |
| PlotQA | 62–63 % | (similar pattern expected) |
| TallyQA | 62–68 % | (similar pattern expected) |

OneVision hook changes 35–80 % of predictions; Qwen2.5-VL changes 1–4 %. ~20× weaker effective hook intervention with the *same* (L, K, α) cell.

### C — Logit margin diagnostic on Qwen2.5-VL ChartQA n=20 a-arm

Script: `scripts/diagnose_qwen25vl_logit_margin.py`.
Output: `outputs/e6_steering/qwen2.5-vl-7b-instruct/_diagnostic_logits/{per_step.jsonl,summary.json}`.

Per-step analysis (20 sids):

| Step | Chosen token | logit Δ on chosen | base top1-top2 margin | mit margin | argmax changed |
|---:|---|---:|---:|---:|---:|
| 0 | `{"` (JSON open) | **+2.99** | 6.27 | 4.44 | 0/20 |
| 1 | `result` | −0.04 | 10.13 | 10.09 | 0/20 |
| 2 | `":` | −0.18 | 15.34 | 15.24 | 0/20 |
| 3 | ` ` (space) | −0.05 | 9.87 | 9.87 | 0/20 |
| **4** | **digit (20/20)** | **+0.0000** | 3.70 | 3.69 | **0/20** |
| **5** | **digit (17/20)** | −0.04 | 5.37 | 5.35 | **0/20** |
| 6 | `}` | −0.04 | 8.37 | 8.33 | 0/20 |
| 7 | `<|im_end|>` | +0.04 | 11.67 | 11.67 | 0/17 |

**Key finding**: Hook fires at prefill, directly modifies *step 0 token logits*. Step 0 chosen token is `{"` (JSON wrapper start), NOT the digit. The digit token (step 4) sees ≈ 0 logit shift. The hook’s prefill modification fails to propagate to the digit decision through KV-cache indirect channel.

### E — Same diagnostic on OneVision ChartQA n=20 a-arm

Output: `outputs/e6_steering/llava-onevision-qwen2-7b-ov/_diagnostic_logits/{per_step.jsonl,summary.json}`.

| Step | Chosen token | logit Δ on chosen | base margin | mit margin | argmax changed |
|---:|---|---:|---:|---:|---:|
| **0** | **digit (20/20)** | **+0.019** | **2.66** | **1.81** | **3/20 = 15 %** |
| 1 | `0` digit / `<|im_end|>` | −1.63 | 6.51 | 5.72 | 3/20 |
| 2 | `<|im_end|>` mostly | −1.60 | 8.67 | 9.00 | 2/14 |
| 3 | `<|im_end|>` | +0.13 | 6.75 | 6.88 | 0/1 |

**OneVision generates digit at step 0**. Hook’s direct prefill modification hits the digit decision directly. Top1–top2 margin narrows from 2.66 → 1.81 — close-decision boundary cells flip.

→ **Hook architecture is IDENTICAL between models. The difference is RESPONSE FORMAT.**

## Full-population response format check

A-arm (`target_plus_irrelevant_number_S1`) raw_text classification across 5 datasets × 6 models on existing baselines (~45k records per model):

| Model | PlotQA | InfoVQA | ChartQA | TallyQA | MathVista |
|---|---|---|---|---|---|
| **OneVision (Main)** | 98.1 % raw | 56.8 % raw / 43 % other | 94.0 % raw | 33.7 % raw / **64.6 % other** | 66.6 % raw / 27.1 % json |
| **LLaVA-Interleave** | 97.1 % raw | 94.1 % raw | 93.7 % raw | 99.1 % raw | 50.1 % raw / 41 % other |
| **Qwen2.5-VL-7B** | 98.6 % JSON | 100 % JSON | 99.5 % JSON | 100 % JSON | 100 % JSON |
| **Qwen2.5-VL-32B** | 99.5 % JSON | 100 % JSON | 100 % JSON | 100 % JSON | 99.7 % JSON |
| **Gemma3-4b** | 83.1 % JSON | 94.9 % JSON | 88.3 % JSON | 79.6 % JSON | 49.3 % JSON / 44 % other |
| **Gemma3-27b** | **100 % JSON** | **100 % JSON** | **100 % JSON** | **100 % JSON** | **100 % JSON** |

System prompt (identical across 5 datasets):
```
You are a visual question answering system.
Return valid JSON only in the form {"result": <number>}.
Use a numeric JSON value for <number>, not a string.
Do not output any other keys, words, explanation, or markdown.
If uncertain, still output the single most likely number in that JSON format.
```

**Family split is clean**:
- LLaVA family (OneVision + Interleave): JSON instruction widely ignored, raw-number output dominates.
- Qwen2.5-VL + Gemma3-27b: JSON instruction strictly followed (≥99 % across all datasets).
- Gemma3-4b: mostly JSON but variable.

The current §6 main result on OneVision is implicitly built on LLaVA family’s instruction-non-compliance. Cross-arch transfer to any instruction-compliant model fails at the **prefill-only hook → digit-decision-token** distance gap.

## Mechanism summary

```
[system] [user start] [image tokens N] [question] [user end] [assistant start] ← hook modifies this position
                                                              │
                                                              ↓ direct channel (layer 27 → norm → lm_head)
                                                       step 0 logits ────┐
                                                                          │
                          if step 0 == digit (LLaVA family raw output)   │
                          → digit logit shifts → flip possible           │
                                                                          │
                          if step 0 == "{"  (Qwen / Gemma JSON output)   │
                          → JSON token shifts but digit is at step 4 ──┐│
                                                                       ▼▼
                          step 4 sees hook only via KV-cache at layer 27
                          attention to position [seq_len-1]; weight ≈ 0
                          → digit logit shift ≈ 0 → no flip
```

## Implications for paper

Three layers of impact:

1. **Cross-arch instantiation deferred**: Phase 2 result on Qwen2.5-VL (mean Δdf = −0.10 pp, 1 of 5 datasets CI-clean) is *prompt-format-confounded* and not a valid cross-arch test. R4 CRIT-1 / bar-raiser conditional Main remain open.

2. **§6 main result reframed**: OneVision E6 mitigation magnitude (Δdf −2.9 pp, Δem(b) +8.8 pp, etc.) is *contingent on LLaVA family's raw-number response format*. If §6 main is to claim *general inference-time intervention*, prompt-format must be controlled.

3. **§8 Limitations / §8.4 follow-ups**: Single-shot prefill hook works only when *step 0 = answer token*. Response format dependence is a structural limit of the hook design.

## Option K (queued for new branch `worktree-e6-prompt-controlled-rerun`)

**Goal**: Disentangle prompt-format confound from architectural cross-arch transfer.

**Plan**:
1. Edit configs/experiment_e7_*.yaml + experiment_e5e_*.yaml: replace JSON-strict prompt with raw-number prompt.
2. Re-run baseline E5e/E7 predictions on 3-model panel: OneVision + Qwen2.5-VL-7B + Gemma3-27b.
3. Re-run Phase 0 (calibrate-subspace, SVD, peak-pick) for each of OneVision + Qwen2.5-VL + Gemma3-27b on the new baseline preds.
4. Re-run Phase 1 (45-cell pilot — or possibly trimmed grid given Phase 1.5 result robustness) for each.
5. Re-run Phase 2 Stage-4 5-dataset paired-bootstrap CI for each.
6. Compare: does cross-arch magnitude transfer when prompt is controlled?

**Predicted outcomes**:
- (a) **All 3 models show similar magnitudes**: prompt-format was the only confound. Cross-arch recipe fully portable. Paper §6.4 lands as "raw-number-prompt cross-arch instantiation succeeds; previous §6 result is prompt-equivalent under raw-number".
- (b) **OneVision similar, Qwen + Gemma weaker**: prompt confound partially explains but encoder/architectural difference remains. Paper §6.4 documents partial transfer + Δmagnitude attribution.
- (c) **OneVision much weaker too without JSON wrapper**: raises questions about §6 main itself; mitigation effect was partly tied to JSON-format’s downstream token diversity. Paper §6 main needs redrafting.

**Budget** (3-GPU pod):
- Baseline re-run: ~1.5 h × 3 models = ~4.5 h
- Phase 0 per model: ~0.5–1 h × 3 = ~3 h
- Phase 1 per model (could trim to 27-cell L ∈ {25,26,27} given Phase 1.5 robustness): ~2 h × 3 = ~6 h
- Phase 2 per model (5 datasets full): ~12 h × 3 = ~36 h (parallelisable across separate model runs)
- Total: ~50 h wall (~2 days) if serial; ~24 h wall if parallelised across 3 sequential model phases.

## Recovery anchors

- Diagnostic script: `scripts/diagnose_qwen25vl_logit_margin.py` (commit on branch `worktree-e6-cross-arch-qwen25vl`).
- Phase 2 raw outputs (confound-affected; keep for audit only):
  - `outputs/e6_steering/qwen2.5-vl-7b-instruct/sweep_subspace_<ds>_plotqa_infovqa_pooled_chosen/predictions.jsonl`
  - `docs/insights/_data/stage4_final_per_dataset{_ci,}_qwen.{csv,md}`
  - `docs/insights/_data/stage4_final_bootstrap_draws_qwen.npz`
- Phase 1 (45-cell pilot, JSON-prompt): `outputs/e6_steering/qwen2.5-vl-7b-instruct/pilot_grid_{plotqa,infographicvqa}_n250/predictions.jsonl` + `docs/insights/_data/pilot_grid_cell_selection_qwen25vl.csv`. Cell selection result (chosen L=26 K=8 α=1.0) is *prompt-independent finding* — should survive Option K re-run.
- Diagnostic per-step records: `outputs/e6_steering/{llava-onevision-qwen2-7b-ov,qwen2.5-vl-7b-instruct}/_diagnostic_logits/per_step.jsonl`.

## Cross-references

- Memory [[project_e6_mitigation]] (or equivalent), [[feedback_em_drop_dealbreaker]], [[feedback_qao_q_d_alignment]].
- Plan doc: `docs/experiments/E6-cross-arch-design.md`.
- Roadmap: `references/roadmap.md` §3.0f (Phase 1.5 pre-decision) + §10 changelog 2026-05-18.
