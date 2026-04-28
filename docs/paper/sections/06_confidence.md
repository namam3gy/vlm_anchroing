# §6. Confidence-modulated anchoring

## 6.1 The wrong/correct binary is a coarse projection

Phase-A (§5) split records by whether `target_only` was correct
(`base_correct`) and showed wrong-base direction-follow exceeds
correct-base by +6.9 to +19.6 pp on every model. This is a binary
projection of a richer underlying structure: model confidence on the
base prediction is continuous (entropy of the answer-token
distribution), and anchor pull modulates with it.

We test this by capturing per-token logits + softmax + top-k
distributions on every `target_only` row (post-commit `5f925b2`
runs: E5b, E5c, E5d, E5e). Three confidence proxies are defined on
the answer-token distribution:

- `softmax_top1_prob`: top-1 softmax probability.
- `top1_minus_top2_margin`: logit gap between top-1 and top-2.
- `entropy_top_k`: Shannon entropy over the captured top-k
  probabilities.

For each (model × dataset × cond_class × stratum) cell, we sort the
base-prediction rows by the chosen proxy and split into four equal
quartiles (Q1 = most confident, Q4 = least confident), then compute
the M2 `adopt_rate` and C-form `direction_follow_rate` per quartile.

## 6.2 Result: monotonic gradient on direction-follow

`entropy_top_k` is the cleanest of the three proxies:

| Proxy | mean(adopt Q4 − Q1) | mean(direction_follow Q4 − Q1) | fully monotone Q1<Q4 cells |
|---|---:|---:|---|
| **`entropy_top_k`** | **+0.044** | **+0.152** | **23 / 35 (df), 10 / 35 (adopt)** |
| `softmax_top1_prob` | +0.036 | +0.108 | 15 / 34 (df), 5 / 34 (adopt) |
| `top1_minus_top2_margin` | +0.017 | +0.013 | 8 / 34 (df), 7 / 34 (adopt) |

(Q4 − Q1 means: rate at the least-confident quartile minus rate at
the most-confident quartile. Positive means anchor pull is larger on
less-confident base predictions.)

Worked example — E5c VQAv2 wrong-base S1 on llava-interleave-7b,
the cell whose binary version contributed Phase-A's +7.2 pp
moved-closer gap:

| quartile | base correctness `mean(exact_match_b)` | anchor adopt | direction-follow C-form |
|---|---:|---:|---:|
| Q1 (most confident) | 0.92 | 0.043 | 0.032 |
| Q2 | 0.72 | 0.084 | 0.080 |
| Q3 | 0.42 | 0.149 | 0.137 |
| Q4 (least confident) | **0.34** | **0.172** | **0.210** |
| **Δ (Q4 − Q1)** | −0.58 | +0.130 | **+0.178** |

The +0.178 quartile gap is a finer-grained projection of the +0.072
binary wrong−correct gap that Phase-A reported on the same dataset
× model (Phase-A used `moved_closer_rate`, a distance-based metric
that approximates direction-follow on the bulk of records). The
quartile structure separates *confidently wrong* (Q4-correct mass)
from *lucky correct* (Q1-correct mass), recovering the binary signal
plus residual within-stratum resolution.

## 6.3 Reading

The continuous monotonicity rules out a "categorical capture under
high uncertainty" reading. If the anchor were simply *replacing* the
prediction whenever the model is uncertain, we would expect
`adopt(Q4) >> adopt(Q3) ~ adopt(Q2) ~ adopt(Q1)` — a step. We see a
gradient on adopt (0.043 → 0.084 → 0.149 → 0.172) and an even
cleaner gradient on direction-follow (0.032 → 0.080 → 0.137 → 0.210).
The picture is consistent with a Mussweiler-Strack
selective-accessibility account: the anchor enters the search-space
as a comparison candidate; it gets *blended into* the answer in
proportion to how much the model is leaning on its prior versus on
the question's content.

## 6.4 Cross-dataset cross-model picture

The +0.152 (Q4−Q1, entropy) gradient holds across:

- **Datasets**: VQAv2 (E5b, E5c), TallyQA (E5b, E5c), ChartQA (E5d,
  E5e), MathVista (E5d, E5e — including γ-α and the γ-β reasoning
  pair).
- **Models**: LLaVA-Interleave-7b, Qwen2.5-VL-7b-Instruct,
  Gemma3-27b-it (E5e), plus the γ-β Qwen3-VL-8b
  Instruct/Thinking pair on MathVista.

23 of 35 (model × dataset × stratum) cells are fully monotone Q1 < Q2
< Q3 < Q4 on direction-follow. The cells that violate strict
monotonicity have either small denominators (E5d ChartQA-validation
~50 base questions per quartile) or near-zero base anchor signal
(qwen2.5-vl-7b on TallyQA correct-base, where the adopt-rate floor
itself is at noise).

## 6.5 The γ-β confirmation: reasoning shifts the same gradient harder

Qwen3-VL-8b-Thinking on MathVista is the strongest single
demonstration of confidence-modulated anchoring on a single
model-pair contrast. Compared to Qwen3-VL-8b-Instruct, the Thinking
checkpoint has:

- *Lower* acc(b) (0.196 vs 0.216) — higher mean base-prediction
  entropy.
- *Higher* adopt(a) (0.117 vs 0.074, ×1.6) and *higher*
  direction-follow C-form (0.291 vs 0.102, ×2.9).

The same gradient that L1's quartile analysis demonstrates *within*
each cell shows up *across* model variants: the model with the
higher base-prediction entropy on a fixed dataset has the larger
anchor pull. §7-§8 take up the question of why reasoning *raises*
mean entropy here rather than lowering it.

## 6.6 §6 summary

The Phase-A wrong/correct binary (§5.2) is a coarse projection of a
continuous structure: anchor pull is monotonic with answer-token
entropy across 23 of 35 cells in our panel and across the
single-pair contrast between Qwen3-VL Instruct and Thinking on
MathVista. The right grounding of "uncertainty modulates anchor pull"
is **graded blending in proportion to base-prediction entropy**,
not categorical capture under threshold.
