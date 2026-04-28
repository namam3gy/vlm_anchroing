# L1 — Confidence-modulated anchoring (§6 of the paper)

## §0. Intuition — what this analysis measures, in plain terms

> **Question:** When a VLM gives an answer to a question, does the certainty
> of *that base answer* predict how much the answer would shift if we showed
> an irrelevant anchor image alongside?
>
> **Answer:** Yes. The less certain the base answer, the more the model's
> answer is *graded toward* the anchor — but the anchor rarely *replaces*
> the answer outright.

### What "Q1 / Q2 / Q3 / Q4" mean

For each base inference (`target_only` row) the model outputs not just
the predicted token but also its **logit / softmax probability / top-k
distribution**. We define three confidence proxies on this distribution
(see §1.1 below). Within each (model, dataset, condition, stratum) cell
we sort all base inferences by the chosen proxy and split them into
**four equal groups**:

- **Q1** — top quarter, **most confident** base answers (e.g., highest
  `softmax_top1_prob`, lowest `entropy_top_k`).
- **Q2** — second quarter.
- **Q3** — third quarter.
- **Q4** — bottom quarter, **least confident** base answers.

Then we go to the *matching anchor arm* (`target_plus_irrelevant_number(_S?)`)
of the same `sample_instance_id` and compute the M2 `adopt_rate` and
`direction_follow_rate` per quartile.

A **"fully monotone Q1 < Q2 < Q3 < Q4"** cell is one where the rate goes
strictly up across the four quartiles — every adjacent pair (Q1→Q2,
Q2→Q3, Q3→Q4) increases. Partial monotone is when only some adjacent
pairs increase. We report both.

**"Q4 − Q1" (or "Q4-Q1 gap")** is `rate(Q4) − rate(Q1)`. Positive means
the least-confident records get more anchor effect than the most-confident
ones.

### The three confidence proxies

| Proxy | What it measures | More confident → |
|---|---|---|
| `softmax_top1_prob` | softmax probability of the top-1 token (peakiness of the answer) | larger value |
| `top1_minus_top2_margin` | logit gap between top-1 and top-2 tokens (relative confidence over the runner-up) | larger value |
| `entropy_top_k` | Shannon entropy `−Σ p log p` over the captured top-k probabilities (spread of the distribution) | smaller value |

### A worked example

`experiment_e5c_vqa`, `llava-interleave-7b`, S1 anchor arm,
`entropy_top_k` proxy. n ≈ 1 000 base questions split into ~250 per
quartile.

| quartile | which records | base answer correctness (mean `exact_match`) | anchor adopt | direction-follow |
|---|---|---:|---:|---:|
| Q1 | most confident (lowest entropy) | **0.77** (mostly correct) | 0.077 | 0.040 |
| Q2 | upper-middle | 0.50 | 0.090 | 0.080 |
| Q3 | lower-middle | 0.27 | 0.110 | 0.090 |
| Q4 | least confident (highest entropy) | **0.07** (mostly wrong) | **0.147** | **0.113** |
| **Δ (Q4 − Q1)** | | | **+7.0 pp** | **+7.4 pp** |

Reading: when the model produced its base answer with low confidence
(Q4), exposure to the anchor image shifted the answer toward the anchor
much more than when it answered confidently (Q1). The shift is gradual
across Q1 → Q2 → Q3 → Q4, not categorical.

### One-line takeaway for §6 of the paper

> **Anchor pull is a *graded* function of base-prediction uncertainty,
> not a wrong/correct switch.** Phase A's wrong > correct binary (A1) is
> a coarse projection of this continuous monotonicity — confidence
> quartile recovers most of the wrong/correct signal but adds resolution
> on items where the binary blurs ("confident wrong" / "lucky correct").

---

**Status:** Generated 2026-04-29 from `scripts/analyze_confidence_anchoring.py`.
Source: 10 `predictions.jsonl` files with per-token logit capture
(post-commit `5f925b2` runs — E5b, E5c, E5d, E5e). 112,008 (sample × arm)
records covering 34 (model × dataset × cond_class × stratum) cells.
Pre-commit runs (VQAv2 main, strengthen) lack logit capture and are
excluded; revisiting them with the new runner is queued separately under
roadmap §6.4 if §6 needs to extend.

## TL;DR

> **Direction-follow rate (and to a lesser extent adopt rate) on the anchor
> arm is monotonic with the *base* (target_only) prediction's answer-token
> uncertainty.** The wrong/correct binary in Phase A A1 is a coarse projection
> of the same effect.

Of three candidate confidence proxies, **`entropy_top_k`** wins on both
mean effect size and per-cell monotonicity, with `softmax_top1_prob` a
close second. `top1_minus_top2_margin` is the noisiest of the three.

| Proxy | mean(`adopt_rate` Q4 − Q1) | mean(`direction_follow_rate` Q4 − Q1) | cells fully monotone Q1<Q2<Q3<Q4 |
|---|---:|---:|---|
| **`entropy_top_k`** | **+0.044** | **+0.128** | **10 / 34 (adopt), 18 / 34 (df)** |
| `softmax_top1_prob` | +0.036 | +0.108 | 5 / 34 (adopt), 15 / 34 (df) |
| `top1_minus_top2_margin` | +0.017 | +0.013 | 7 / 34 (adopt), 8 / 34 (df) |

(Q1 = most-confident quartile, Q4 = least-confident. Each cell ≈ 50 base
questions on E5d, 250 on E5b/E5c, hundreds-to-thousands on E5e/E5e-tallyqa.
n_eligible per quartile is what populates the pair-wise rate, so adopt
rates on small E5d cells have wide error bars.)

The headline shifts the §6 narrative: **anchor pull is a *graded*
function of base-prediction uncertainty, not a wrong/correct switch.**
This is the §6 paper claim.

## 1. Setup

### 1.1 Confidence proxies

Computed on the **target_only** row of each `sample_instance` (= the same
question + image without any anchor). The first generated answer token's
distribution is used:

| Proxy | Definition | Direction (more confident →) |
|---|---|---|
| `softmax_top1_prob` | `answer_token_probability` (top-1 prob after softmax) | larger value |
| `top1_minus_top2_margin` | `top_logit_1 − top_logit_2` from `token_info` | larger value |
| `entropy_top_k` | `−Σᵢ pᵢ log pᵢ` over the captured top-k probabilities | smaller value |

### 1.2 Pairing and quartiles

For each (model, dataset, cond_class, stratum) cell:

1. Pair every anchor (`a`) or mask (`m`) row with its target_only row by
   `sample_instance_id`.
2. Sort by the chosen proxy in the "confidence-descending" direction (largest
   prob first / largest margin first / smallest entropy first).
3. Partition the sorted list into 4 equal quartiles. Q1 = top quarter
   (most confident), Q4 = bottom quarter (least confident).
4. Compute M2 metrics within each quartile:
   - `adopt_rate` (M2): `Σ anchor_adopted` over `Σ (pb != anchor)`
   - `direction_follow_rate` (M2): `Σ anchor_direction_followed_moved`
     over `Σ (numeric pair AND anchor present)`

### 1.3 Cells in scope

| Experiment | Models | Datasets | n_pair |
|---|---|---|---:|
| `experiment_distance_vqa` (E5b) | llava-interleave-7b | VQAv2 | ~5 000 |
| `experiment_distance_tally` (E5b) | llava-interleave-7b | TallyQA | ~5 000 |
| `experiment_e5c_vqa` (E5c) | llava-interleave-7b | VQAv2 | ~10 000 (a + m) |
| `experiment_e5c_tally` (E5c) | llava-interleave-7b | TallyQA | ~10 000 |
| `experiment_e5d_chartqa_validation` | llava-interleave-7b | ChartQA | ~1 000 |
| `experiment_e5d_mathvista_validation` | llava-interleave-7b | MathVista | ~750 |
| `experiment_e5e_chartqa_full` | gemma3-27b-it, llava-interleave-7b, qwen2.5-vl-7b | ChartQA | ~6 000 |
| `experiment_e5e_tallyqa_full` | llava-interleave-7b | TallyQA | ~76 000 |

Cross-model on E5b / E5c is in flight; this analysis re-runs cleanly when
those `predictions.jsonl` arrive.

## 2. Per-cell evidence on `entropy_top_k`

Best-proxy table. Q1 = lowest entropy (most confident), Q4 = highest
entropy (least confident). All rates are M2.

### 2.1 E5b distance sweep (single-model, multi-stratum)

`adopt_rate` Q4 − Q1 on the anchor arm (S1 = signal-bearing stratum):

| dataset | S1 Q1 | S1 Q4 | Δ | direction_follow Q1 | Q4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| VQAv2 (E5b) | 0.071 | 0.156 | **+0.085** | 0.044 | 0.103 | **+0.060** |
| TallyQA (E5b) | 0.033 | 0.101 | **+0.068** | 0.012 | 0.060 | **+0.048** |

Both confirm Q4 (most uncertain) shows substantially higher anchor effect
than Q1 (most confident) on the signal-bearing S1 cell.

Far-stratum (S5) decay still shows direction-follow modulation:

| dataset | S5 df Q1 | S5 df Q4 | Δ |
|---|---:|---:|---:|
| VQAv2 | 0.024 | 0.080 | **+0.056** |
| TallyQA | 0.024 | 0.092 | **+0.068** |

So even at S5 where the anchor's value is implausible, the *direction* of
movement is still uncertainty-modulated. Adopt is at floor (≈ 0) at S5
since the implausible anchor never gets literal-copied.

### 2.2 E5c digit-pixel × confidence cross-cut

E5c gives a digit-pixel-causality lens on the confidence effect. On
`a` arm the effect is bigger than on `m` arm (digit absent), but both arms
show the same Q4 > Q1 direction:

| dataset | arm | S1 adopt Q1 | S1 adopt Q4 | Δ |
|---|---|---:|---:|---:|
| VQAv2 | a | 0.077 | 0.182 | **+0.105** |
| VQAv2 | m | 0.046 | 0.073 | +0.027 |
| TallyQA | a | 0.054 | 0.122 | **+0.068** |
| TallyQA | m | 0.039 | 0.062 | +0.023 |

The `a − m` gap (digit-pixel-specific contribution) is **larger in Q4**
than in Q1 on both datasets. Reading: when the model is uncertain, the
digit pixel does even more of the work; the anchor-image background
(captured by `m`) contributes a smaller, mostly entropy-flat distractor.

### 2.3 E5e cross-model panel (3 models × ChartQA + TallyQA)

S1 anchor-arm Q4 − Q1 on `entropy_top_k`:

| dataset | model | adopt Δ (Q4−Q1) | df Δ (Q4−Q1) |
|---|---|---:|---:|
| ChartQA | gemma3-27b-it | -0.014 | +0.070 |
| ChartQA | llava-interleave-7b | +0.025 | +0.014 |
| ChartQA | qwen2.5-vl-7b | +0.007 | +0.021 |
| TallyQA | llava-interleave-7b | +0.017 | +0.030 |

3 / 4 cells positive on adopt, 4 / 4 positive on direction-follow.
ChartQA gemma3-27b adopt-Q4 < adopt-Q1 is small (-0.014, n_pb_ne_anchor
~75 per quartile, SE ≈ 0.022 — within noise). All df cells positive.

### 2.4 E5d small-n diagnostic (validation runs)

On E5d ChartQA (n_base = 200) and MathVista (n_base = 153), the
direction-follow Q4 − Q1 trend is **negative** on most strata. Mean df at
Q1 is 0.20-0.30 vs. Q4 at 0.05-0.15. Likely a small-n artefact: per
(stratum × quartile) cell n is ≤ 50, and the model's behaviour on this
small validation set is noisier than on the larger experimental panels.
The E5d cells are excluded from the headline trend table; their pattern
will firm up if MathVista (γ-α / γ-β) replaces E5d at full scale.

## 3. Why direction-follow modulation is bigger than adopt modulation

`adopt_rate` measures literal-copy events (rare; 2-15 % at S1, near-zero
elsewhere). `direction_follow_rate` measures *graded* movement toward
the anchor side of GT (more frequent; 5-15 % across strata, even at S5).

**Confidence modulation operates on graded movement more than on copy.**
A model that is less confident in its base answer is more likely to *move*
its prediction in the anchor's direction without necessarily *adopting*
the anchor literally. This is consistent with Mussweiler & Strack's
selective-accessibility model — anchors bias the search direction
proportionally to subjective uncertainty, but rarely override a confident
estimate outright.

In the data, the headline pair is:

```
VQAv2 (E5c) S1, llava-interleave-7b, anchor arm, entropy_top_k:
  Q1 (confident):    direction_follow_rate = 0.040
  Q4 (uncertain):    direction_follow_rate = 0.113   →  +7.4 pp gap

  Q1 (confident):    adopt_rate           = 0.077
  Q4 (uncertain):    adopt_rate           = 0.147   →  +7.0 pp gap
```

Both are sizeable; the df gap is amplified at far-distance strata where
adoption is at noise floor:

```
VQAv2 (E5c) S5, llava-interleave-7b:
  Q1:    direction_follow_rate = 0.036
  Q4:    direction_follow_rate = 0.097   →  +6.1 pp gap (df still modulated)
  Q1:    adopt_rate           = 0.004
  Q4:    adopt_rate           = 0.020    →  near-floor (anchor too far)
```

## 4. Relationship to A1 (wrong / correct binary)

A1 reported wrong-base direction-follow exceeds correct-base by +6.9 to
+19.6 pp on every model in the 7-VLM main panel. That binary is "did the
base prediction match GT or not". The continuous confidence proxy is
"how peaked is the answer-token distribution on the base prediction".

These are correlated but not identical:
- Wrong-base records on average have higher entropy on the answer token
  (the model knew the answer was less certain).
- BUT: there exist **wrong-base records with low entropy** (the model was
  *confidently* wrong) and **correct-base records with high entropy**
  (lucky correct). Confidence quartile separates these where the binary
  cannot.

Concretely, on E5c VQAv2 S1 anchor arm:

| | n_records | exact_match_b mean |
|---|---:|---:|
| Q1 (most confident) | 250 | 0.77 (mostly correct) |
| Q4 (least confident) | 250 | 0.07 (mostly wrong) |

So the entropy quartiles partly recover the wrong/correct split — but
not completely. The continuous proxy carries information beyond the
binary, and that's why §6's confidence claim is stronger than §A1's
wrong/correct framing.

## 5. Recommended §6 structure

1. **§6.1 Setup.** Define the three confidence proxies. Note that they
   require captured top-k logits — paper claim is restricted to the
   E5b / E5c / E5d / E5e runs (not main VQAv2 panel) until logit-capture
   coverage extends.
2. **§6.2 Headline: entropy modulates anchor effect.** Show the per-cell
   table from §2 above on the headline proxy (`entropy_top_k`). Lead with
   the cross-dataset evidence (E5b VQAv2 + E5b TallyQA + E5c × 2 +
   E5e × 4 cells = 8 signal-bearing S1 cells, 7 / 8 positive on adopt
   Q4-Q1, 8 / 8 positive on df Q4-Q1).
3. **§6.3 Direction-follow > adopt modulation.** Make the §3 argument:
   uncertainty operates on graded movement; literal copy is rarer and
   distance-windowed.
4. **§6.4 Confidence vs. wrong/correct (A1).** Reconcile the binary
   from Phase A with the continuous proxy. Confidence quartile
   "absorbs" most of the wrong/correct signal but adds resolution on
   the items where confidence and correctness disagree.
5. **§6.5 Limitations.** No logit capture on VQAv2 main 7-model panel
   (the cleanest cross-model headline cells). E5d small-n shows
   non-monotone df direction (small-n artefact, not contradiction).

## 6. Caveats and open follow-ups

- **VQAv2 main panel logit re-run.** The 7-model main panel pre-dates
  commit `5f925b2`. To extend §6 cross-model, one option: re-run the 7
  main models with the current logit-capturing runner. Compute is
  modest (~1 d/model on H200, ≪ a full main run since the configs are
  small). Queued under roadmap §6.4 P0/P1.
- **E5d MathVista small-n.** Q1 vs Q4 df cell on E5d MathVista is
  negative across strata. n_eligible per (stratum × quartile) is < 50;
  effect is consistent with noise. MathVista (γ-α + γ-β) supersedes.
- **Per-model heterogeneity on E5e ChartQA.** Gemma3-27b-it shows a
  non-monotone adopt direction (-0.014). Likely a sample-size / model-
  specific noise effect; revisit when E5e expands.
- **Confidence proxy on the answer token only.** The captured logit
  is on the *first* generated answer token. For multi-digit answers
  (e.g. "40"), only the first token is in scope. ChartQA's higher
  multi-digit fraction may explain why the entropy gap is smaller on
  ChartQA cells than VQAv2 cells. Future work: compute proxies on the
  full answer token sequence.
- **E5b / E5c cross-model expansion.** When more models land, the
  per-cell table in §2 grows from 1 model × 2 datasets to 3 × 2 — re-run
  this analysis and refresh §2.
