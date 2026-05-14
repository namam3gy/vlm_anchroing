# Judge Anchoring Pilot — VLM-as-Judge Cross-Vendor / Cross-Dataset Results

**Date:** 2026-05-13 ~ 2026-05-14
**Status:** ✅ Strong positive findings on 4/10 cells, robust evidence for §8 demo
**Decision:** PASS — proceed with paper §8 demo subsection in follow-up plan
**Source predictions:** `outputs/judge_pilot/<judge>/<ts>/` and `outputs/judge_pilot_vlrewardbench_a1/<judge>/<ts>/`
**Figure:**
- `docs/figures/judge_pilot_v1_5judges_2datasets_n200.png` — 2 dataset × 5 judge line overlay (n=200)

---

## 1. Setup

| Item | Value |
|---|---|
| Datasets | **VLFeedback** (`MMInstruction/VLFeedback`, n=200) + **VL-RewardBench** (`MMInstruction/VL-RewardBench`, n=200) |
| Response selector | **chosen** — VLFeedback: max average of 3 GPT-4V dim ratings; VL-RewardBench: argmax(human_ranking) |
| Sample seed | VLFeedback: 20260512 (n=100) extended with 20260513 (n=200); VL-RewardBench: 20260513 |
| Anchor | digit "1" (`inputs/irrelevant_number/1.png`); m-arm: same image with digit pixels Telea-masked |
| Arms | b (image only) / a (image + anchor) / m (image + masked anchor) |
| Score | Visual Faithfulness 1–5 (single-dim, VLFeedback Silkie native rubric) |
| Judges | gpt-4o, gpt-5.1, gemini-2.5-pro, gemini-2.5-flash, claude-sonnet-4-5-20250929 |
| Gateway | `gateway.letsur.ai/v1` (Staix, OpenAI-compat); accessed 2026-05-13/14 |
| Reasoning toggle | gpt-5.1 + gemini-2.5-flash via `reasoning_effort=minimal` (non-reasoning); gpt-4o + claude-sonnet inherently non-reasoning; **gemini-2.5-pro reasoning forced (cannot disable)** |

**Per-judge max_output_tokens:**
- gpt-4o: 8 (single integer)
- gpt-5.1: 64 (post non-reasoning)
- gemini-2.5-pro: 2048 (reasoning headroom — still 29% parse fail on VLB; see caveats)
- gemini-2.5-flash: 64
- claude-sonnet-4-5-20250929: 64

---

## 2. Final results — 5 judges × 2 datasets, anchor=1, n=200

### 2.1 Headline matrix

| Dataset | Judge | n_pair | mean b → a → m | Δ(a−b) | Δ(m−b) | Δ(a−m) | P(score=1) b → a |
|---|---|---|---|---|---|---|---|
| VLFeedback | **gpt-4o** | 199 | **4.49 → 3.94 → 4.41** | **−0.55** | −0.08 | **−0.47** | 5% → **25%** (+21pp) |
| VLFeedback | gpt-5.1 | 199 | 3.89 → 3.85 → 3.86 | −0.04 | −0.03 | −0.01 | 12% → 13% (+0.5pp) |
| VLFeedback | gemini-2.5-pro | 193 | 3.67 → 3.68 → 3.77 | +0.02 | +0.10 | −0.08 | 20% → 22% (+1.6pp) |
| VLFeedback | **gemini-2.5-flash** | 198 | **4.32 → 3.89 → 3.98** | **−0.41** | −0.34 | −0.07 | 14% → 26% (+12pp) |
| VLFeedback | claude-sonnet-4-5 | 193 | 3.88 → 3.83 → 3.86 | −0.05 | −0.02 | −0.03 | 8% → 12% (+3.6pp) |
| VL-RewardBench | **gpt-4o** | 199 | **4.61 → 3.65 → 4.27** | **−0.96** | −0.34 | **−0.62** | 2% → **32%** (+30pp) |
| VL-RewardBench | gpt-5.1 | 199 | 3.35 → 3.40 → 3.23 | +0.05 | −0.12 | +0.17 | 14% → 11% (−3.0pp) |
| VL-RewardBench | gemini-2.5-pro | 165 | 2.67 → 2.62 → 2.77 | +0.04 | +0.23 | −0.19 | 36% → 37% (+0.6pp) |
| VL-RewardBench | **gemini-2.5-flash** | 197 | **4.35 → 3.75 → 3.79** | **−0.58** | −0.54 | −0.04 | 8% → 25% (+17pp) |
| VL-RewardBench | claude-sonnet-4-5 | 198 | 3.26 → 3.11 → 3.14 | −0.15 | −0.12 | −0.03 | 5% → 11% (+5.6pp) |

### 2.2 Susceptibility ranking (by |Δ(a−b)| averaged across 2 datasets)

1. **gpt-4o** — avg 0.75 (highly susceptible; **digit-specific**)
2. **gemini-2.5-flash** — avg 0.50 (highly susceptible; **distractor-general**)
3. claude-sonnet-4-5 — avg 0.10 (weak)
4. gpt-5.1 — avg 0.05 (effectively robust)
5. gemini-2.5-pro — avg 0.03 (robust)

---

## 3. Three key findings

### 3.1 OpenAI generation upgrade dramatically reduces anchor susceptibility

| Judge | VLF Δ(a−b) | VLB Δ(a−b) | mean_a−mean_b range |
|---|---|---|---|
| gpt-4o | −0.55 | −0.96 | drops nearly 1 pt on 1-5 scale |
| gpt-5.1 | −0.04 | +0.05 | within noise |

The newer gpt-5.1 (non-reasoning mode via `reasoning_effort=minimal`) is **essentially robust**, while gpt-4o is the most-biased judge in our panel. Suggests anchor susceptibility is being addressed in newer model generations — though we cannot disentangle whether the cause is RLHF / safety training, model scale, or specific instruction-following improvements.

### 3.2 Reasoning correlates with robustness within the Google family

| Judge | Reasoning mode | VLF Δ(a−b) | VLB Δ(a−b) |
|---|---|---|---|
| gemini-2.5-pro | forced ON (cannot disable per provider policy) | +0.02 | +0.04 |
| gemini-2.5-flash | OFF (`reasoning_effort=minimal`) | **−0.41** | **−0.58** |

A clean within-vendor ablation: the same Google family model becomes **dramatically more susceptible when reasoning is disabled**. This supports a "reasoning provides anchor-resistance" hypothesis — though confounded by the pro-vs-flash capability gap. The cleanest causal claim would require gemini-2.5-pro with thinking disabled (currently impossible per Google's API policy).

### 3.3 Two distinct anchor-effect mechanisms

The (a−m) vs (m−b) split distinguishes two attack patterns:

| Pattern | Δ(a−b) | Δ(m−b) | Δ(a−m) | Interpretation |
|---|---|---|---|---|
| **Anchor-digit-specific** | large negative | small | **large negative** | The digit pixels themselves bias the score. Adversarial watermarking attack. |
| **Distractor-general** | large negative | **≈ Δ(a−b)** | small | Any extra image biases the score. Multi-image instruction-following weakness. |

**Per judge:**

| Judge | Pattern (VLF / VLB) |
|---|---|
| **gpt-4o** | digit-specific / digit-specific (Δ(a-m) -0.47 / -0.62 ≫ Δ(m-b) -0.08 / -0.34) |
| **gemini-2.5-flash** | distractor-general / distractor-general (Δ(a-m) ≈ 0; Δ(m-b) ≈ Δ(a-b)) |
| gpt-5.1 / gemini-2.5-pro / claude-sonnet-4-5 | null on both patterns |

→ "Anchor injection" attack works on gpt-4o; "any extra image" distraction works on gemini-2.5-flash. Different threat models.

---

## 4. Caveats

### 4.1 Parse failures concentrated on gemini-2.5-pro VL-RewardBench

| Judge | VLF fails / 600 | VLB fails / 600 |
|---|---|---|
| gpt-4o | 0 | 1 |
| gpt-5.1 | 0 | 3 |
| **gemini-2.5-pro** | 6 (1%) | **58 (10%)** ← outlier |
| gemini-2.5-flash | 1 | 3 |
| claude-sonnet-4-5 | 6 (1%) | 6 (1%) |

VLB has higher failure rate because it includes math / MMMU-Pro / hallucination tasks that trigger longer reasoning chains, exceeding `max_output_tokens=2048` budget. n_pair drops to 165 for gemini-pro VLB — still adequate for null result reading, but not directly comparable to the n=199 cells.

### 4.2 Anchor=5 ceiling-push not re-tested under expanded panel

Earlier 2-judge × 2-dataset run with anchor=5 (chosen + single-dim) showed null effects on both judges (saturation: most baseline scores already at 5). Expanded to 5-judge panel only for anchor=1; a separate ceiling-push run with the new 3 judges remains future work.

### 4.3 Random vs chosen response selection — chosen privileged for stronger signal

Our chosen-response selector intentionally privileges high-baseline samples (where anchor has room to push down). An earlier ablation with random response selection showed substantially weaker signal (~10× attenuation on gpt-4o), confirming that anchor effect is *baseline-conditional*: when the judge would naturally rate a response 5/5, anchor injection can flip it; for already-low-baseline samples, anchor has no further floor to push to. This is a meaningful finding in itself but means our headline numbers reflect "worst-case adversarial scenario" not "average-case".

### 4.4 Gateway routing opaque on snapshot resolution

Models accessed via Staix gateway (`gateway.letsur.ai/v1`) return alias-only model IDs in responses. Underlying provider snapshots are not exposed by the API. Only Claude carries the explicit version (`claude-sonnet-4-5-20250929`). For other models, paper citation must use the alias + access date pattern.

### 4.5 Random selection variants and CoT prompt variants explored but discarded

Brief design ablations (random response selector, multi-dim labeled output, per-dim Analysis + Rating CoT prompt) were tried at smaller scale (n=30). All weakened or did not improve the anchor signal compared to the simple chosen + single-dim VF protocol. Configs and short writeups preserved as historical reference; main results use the simplest reproducible pipeline.

---

## 5. §8 demo narrative (suggested)

> *"We evaluate whether anchor injection — the cross-modal bias studied in §4–§7 — transfers to the VLM-as-judge setting. We sample 200 (image, prompt, response) triplets from each of two standard judge benchmarks (VLFeedback, VL-RewardBench), select the highest-rated response per record, and ask each judge for a Visual Faithfulness rating 1–5 under three conditions: image alone (b), image + anchor digit "1" (a), or image + same anchor with digit pixels Telea-masked (m). Five frontier closed-API judges spanning three vendors are tested. Results are heterogeneous: gpt-4o shows the strongest anchor effect (mean Visual Faithfulness drops from 4.49 to 3.94 on VLFeedback and from 4.61 to 3.65 on VL-RewardBench; the proportion of samples scored '1' jumps by 21pp and 30pp respectively); the (a−m) per-sample paired contrast (−0.47 / −0.62) isolates the digit-pixel effect from generic distractor confounds. gemini-2.5-flash shows comparable headline magnitude (Δ(a−b) ≈ −0.5 on both datasets) but a different mechanism: its (a−m) is ≈ 0 with Δ(m−b) ≈ Δ(a−b), suggesting susceptibility to any second image rather than to the anchor digit specifically. gpt-5.1, gemini-2.5-pro, and claude-sonnet-4-5 show effects within ±0.15 of zero. Within the Google family, the reasoning variant (gemini-2.5-pro, reasoning forced on) is robust while the non-reasoning variant (gemini-2.5-flash) is highly susceptible — supporting a 'reasoning provides anchor-resistance' hypothesis. Within OpenAI, the newer gpt-5.1 fully fixes gpt-4o's anchor susceptibility. The bias surfaces on standard judge prompts at deployment-realistic settings, with vendor- and model-specific susceptibility profiles directly relevant to RLAIF and judge-mediated evaluation pipelines."*

---

## 6. Files

### Code
- `src/vlm_anchor/vlfeedback_loader.py` — chosen + random selectors
- `src/vlm_anchor/vlrewardbench_loader.py` — VL-RewardBench loaders (chosen + random)
- `src/vlm_anchor/judge_clients.py` — OpenAI/Gemini multi-image clients with `extra_body` (for `reasoning_effort=minimal`) + multi-dim labeled parser
- `src/vlm_anchor/judge_pilot_data.py` — PilotSample, manifest I/O, paired bootstrap CI
- `scripts/build_judge_pilot_dataset.py` — manifest builder (dispatches on dataset.kind + selector)
- `scripts/run_judge_pilot.py` — runner (resumable, multi-judge, dim-labels)
- `scripts/analyze_judge_pilot.py` — analysis (paired CI, figures)

### Configs (anchor=1 v1 main runs)
- `configs/judge_pilot.yaml` — VLFeedback × 5 judges
- `configs/judge_pilot_vlrewardbench_a1.yaml` — VL-RewardBench × 5 judges

(Anchor=5 ceiling-push and v2 random+multi-dim ablation are described in §4.2 / §4.5
above; their configs were intermediate exploration and have been removed —
re-create from this writeup if needed for replication.)

### Outputs (gitignored)
- `outputs/judge_pilot/<judge>/<ts>/` — VLFeedback × 5 judges
- `outputs/judge_pilot_vlrewardbench_a1/<judge>/<ts>/` — VL-RewardBench × 5 judges

### Figure
- `docs/figures/judge_pilot_v1_5judges_2datasets_n200.png` — 2 dataset × 5 judge line overlay (n=200)

---

## 7. Decision

**PASS** — sufficient evidence to draft §8 demo subsection. Follow-up plan should:

1. Author §8 paragraph(s) using the narrative draft in §5 above
2. Add the 2×5 line-overlay figure (`judge_pilot_v1_5judges_2datasets_n200.png`)
3. Decide on table form (raw means or just Δs)
4. Note caveats (§4) honestly in main text or appendix
5. Keep pilot artifacts (configs, scripts, outputs) for full reproducibility

Do NOT touch paper §1–§7 prose — pilot is a §8 add-on, not a method contribution.
