# Headline numbers (C-form re-aggregation, 2026-04-28)

> Extracted from `references/roadmap.md §3.3` on 2026-04-29 to keep the
> roadmap lightweight. Re-aggregate from raw `predictions.jsonl` via
> `scripts/reaggregate_paired_adoption.py` if numbers drift.

All numbers below use the canonical M2 metrics from `roadmap.md §4` with the
`direction_follow_rate` numerator in **C-form**: `(pa-pb)·(anchor-pb) > 0`.
Pre-refactor results archived at `outputs/before_C_form/`; side-by-side
deltas in `docs/insights/C-form-migration-report.md`. Adopt and
exact-match are unchanged by the refactor; only `direction_follow*`
columns moved.

## Standard-prompt VQAv2 number subset, 17,730 samples / model

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

## E5b distance sweep — `llava-interleave-7b` only (cross-model in flight)

Wrong-base subset, `adopt_rate` (M2):

| stratum | VQAv2 (n_eligible) | TallyQA (n_eligible) |
|---|---:|---:|
| S1 | **0.134** (313) | **0.098** (265) |
| S2 | 0.030 | 0.006 |
| S3 | 0.010 | 0.003 |
| S4 | 0.010 | 0.000 |
| S5 | 0.003 | 0.000 |

Pattern: sharp peak at S1, decay to noise floor by S5 (cross-dataset).

## E5c digit-pixel causality — 3-model panel ✅

Wrong-base S1 paired conditional adoption (`adopt_cond` from
`docs/insights/_data/E5c_per_cell.csv`, 2026-04-29; the M2 marginal
form is in the main panel rows above). The a − m gap reads off whether
inpainting the digit pixel removes the effect:

| dataset | model | anchor S1 | masked S1 | gap (a − m) |
|---|---|---:|---:|---:|
| VQAv2 | llava-interleave-7b | 0.129 | 0.068 | **+6.1 pp** |
| VQAv2 | gemma3-27b-it | 0.138 | 0.082 | **+5.7 pp** |
| VQAv2 | qwen2.5-vl-7b | 0.070 | 0.066 | +0.4 pp |
| TallyQA | llava-interleave-7b | 0.110 | 0.084 | **+2.6 pp** |
| TallyQA | gemma3-27b-it | 0.074 | 0.053 | +2.1 pp |
| TallyQA | qwen2.5-vl-7b | 0.033 | 0.037 | −0.5 pp |

(gemma3-27b-it TallyQA at `max_samples=300` per §9; full n=1000
stratified is infeasible on this model.) Direction-follow `df_cond`
mirrors the ranking: VQAv2 wrong-base S1 df(a)/df(m) = 0.280/0.221
on gemma3 (+5.99 pp), 0.208/0.155 on llava (+5.3 pp), 0.148/0.163
(~0) on qwen2.5-vl. Cross-model expansion now confirms the
expected rank: largest pull (llava) → largest gap; mid-panel
(gemma3) → mid gap; floor (qwen2.5-vl) → floor gap. The
digit-pixel-causality claim from §5.4 generalises across all three
models on VQAv2 and across the two pulled models on TallyQA, with
qwen2.5-vl serving as the negative control.

## E5e S1-only 4-condition full — 3-model panel × ChartQA + TallyQA

All-base, S1 anchor / masked, C-form (numbers cross-checked against
`outputs/experiment_e5e_*_full/<model>/<ts>/summary.json` and
`docs/insights/_data/experiment_e5e_*_per_cell.csv` 2026-04-29):

| dataset | model | acc(b) | acc(a) | adopt(a) | adopt(m) | df(a) C-form | df(m) C-form |
|---|---|---:|---:|---:|---:|---:|---:|
| ChartQA | gemma3-27b-it | 0.217 | 0.218 | **0.037** | 0.022 | **0.096** | 0.079 |
| ChartQA | llava-interleave-7b | 0.113 | 0.110 | **0.028** | 0.009 | **0.152** | 0.115 |
| ChartQA | qwen2.5-vl-7b | 0.255 | 0.253 | **0.017** | 0.013 | **0.051** | 0.046 |
| TallyQA | gemma3-27b-it | 0.237 | 0.236 | **0.027** | 0.016 | **0.073** | 0.060 |
| TallyQA | llava-interleave-7b | 0.236 | 0.233 | **0.026** | 0.014 | **0.066** | 0.056 |
| TallyQA | qwen2.5-vl-7b | 0.230 | 0.226 | **0.011** | 0.011 | **0.029** | 0.030 |

TallyQA × gemma3-27b-it cell landed 2026-04-29 (inference completed
2026-04-28 23:28; C-form re-aggregation 2026-04-29 via
`reaggregate_paired_adoption.py`). Wrong-base S1 cell from the per-cell
CSV: `adopt(a) = 0.059`, `df(a) C-form = 0.152`, `adopt(m) = 0.034`,
`df(m) = 0.133` — matching the panel-leading TallyQA pattern (graded
tilt with anchor > masked, S1 distance window).

## Mechanistic / mitigation summary

E1d upper-half ablation: **−4.0 to −10.5 pp** `direction_follow` (C-form)
on 6/6 models; fluency-clean on 4/6 (mid-stack cluster + Qwen).

E4 Phase 2 full mid-stack-cluster (C-form): `direction_follow_rate`
reduction LLaVA-1.5 **−14.6 %** rel, ConvLLaVA **−9.6 %**, InternVL3
**−5.8 %**; `exact_match` rises +0.49 to +1.30 pp; `accuracy_vqa(b)`
invariant — anchor-condition specific.
