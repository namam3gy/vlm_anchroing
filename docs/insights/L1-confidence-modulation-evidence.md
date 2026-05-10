# L1 — Confidence-modulated anchoring (§6 of the paper)

> **Historical record (post-2026-05-04).** InternVL3-8b was removed from
> the active model panel on 2026-05-10. The chart-stack reversal observed
> below in §3.3 / §B2 (wrong/correct decomposition) / §B3 (max-tokens 16
> follow-up) is preserved as a historical anomaly; the model is no longer
> in the live panel and the residual mechanism hypotheses queued for §8
> (confident-confabulation, InternViT-300M encoder family interaction)
> are closed. See `references/roadmap.md` §10 (2026-05-10 changelog) for
> the full removal rationale.

> **2026-05-10 update.** Paper switched headline binning from 4-quartile
> (Q1-Q4) to **6-bin (B1-B6)**. New 6-bin aggregate on the same
> 85-cell panel: **mean df B6−B1 gap = +0.182 (`cross_entropy`) /
> +0.231 (`log_prob_sum`)** (vs 4-bin +0.156 / +0.191 — 6-bin gap
> +17-21 % larger because extreme bins capture confidence endpoints
> more accurately). Strict monotonicity criterion changed: 6-bin
> ≥ 4 of 5 bin-pair strict ↑ on **52-60 / 85 cells** (relaxed,
> 1 dip allowed) replaces 4-bin "fully strict 3/3 on 51/85" as the
> paper headline; *fully strict 5/5* falls to 21-24 / 85 (5-pair
> strictness is structurally harder than 3-pair). InternVL3 chart-stack
> reversal sign-preservation 5/5 verified (paper §C.4 robustness
> table). Worked example switched to **PlotQA × LLaVA-OneVision-7b
> *(Main)*** (B1=B2=0 broad floor → B3-B5 sigmoid rise → B6 saturation;
> df 0.000 → 0.000 → 0.028 → 0.128 → 0.238 → 0.289). New canonical
> CSVs: `_data/L1_confidence_quartile_long_6bin.csv`,
> `_data/L1_proxy_monotonicity_6bin.csv`,
> `_data/L1_proxy_comparison_6bin.csv`. Original 4-bin CSVs preserved
> at `_data/L1_confidence_quartile_long.csv` etc. (no suffix). All
> §0–§3 below uses 4-bin Q labels — read as historical / dual-form
> reference; current paper headline is 6-bin.

> **2026-05-04 update.** Re-aggregated on the **5-dataset × 7-model**
> Phase 1 P0 v3 main matrix (was 4-dataset × 1-3 model). Coverage now:
> 85 anchor cells across {VQAv2, TallyQA, ChartQA, MathVista, PlotQA,
> InfoVQA} × {llava-interleave-7b, llava-onevision-7b, gemma3-4b-it,
> gemma3-27b-it, internvl3-8b, qwen2.5-vl-7b, qwen2.5-vl-32b}. Best
> proxy is now **`log_prob_sum`** (sequence-level confidence,
> length-aware): mean `direction_follow_rate` Q4 − Q1 = **+0.191**,
> **51/85 anchor cells fully monotone** (60 %). The shorter
> `cross_entropy` proxy is the paper-clean default (length-invariant,
> Q4 − Q1 = +0.156, 43/85 monotone). The qualitative
> "uncertainty-modulated graded pull, with wrong/correct as coarse
> binary projection" finding survives the expansion. **New finding
> §3.3: InternVL3-8b shows H7 *reversal* on chart-stack datasets** —
> least-confident records (Q4) anchor *less*, not more, on PlotQA / ChartQA.
> Same model has near-zero wrong − correct gap on those datasets
> (`E7-plotqa-infovqa-evidence.md` §4). Per-cell CSV refreshed at
> `_data/L1_confidence_quartile_long.csv`; per-proxy summary at
> `_data/L1_proxy_monotonicity.csv`.

> **2026-04-28 update.** Re-run on C-form re-aggregated data: the
> headline claim is *strengthened*. With `entropy_top_k`, mean
> `direction_follow_rate` Q4 − Q1 = **+0.152** (was +0.128 under
> anchor·gt form), with **23/35 anchor cells fully monotone**
> Q1 < Q2 < Q3 < Q4 (was 18/34). `adopt_rate` Q4 − Q1 = **+0.044**
> (was +0.044, unchanged), 10/35 monotone (was 10/34). Inline numbers
> below have been refreshed; the qualitative finding —
> "uncertainty-modulated graded pull, with wrong/correct as a coarse
> binary projection" — survives unchanged. Figure
> `paper_L1_confidence_quartile.png` regenerated against the new CSV.
> *(2026-05-04 supersession: see top.)*

> *(2026-05-10 supersession: paper headline switched to 6-bin. Tables
> below remain in 4-bin Q form as historical / cross-validation
> evidence; current 6-bin canonical numbers in
> `_data/L1_confidence_quartile_long_6bin.csv` and paper §4.4 / §6.)*

## 2.E. 5-dataset × 7-model expansion (2026-05-04, 4-bin Q, `cross_entropy` proxy — historical reference)

The §6 main panel covers 7 models on 5 datasets (the same matrix used
in §3.3 / §5; see `phase1-p0-v3-summary.md`). Per-cell `direction_follow`
Q4 − Q1 on the S1 anchor arm under the paper-default `cross_entropy`
proxy (4-bin partition; current paper headline is 6-bin):

| dataset | model | df Q1 | df Q4 | Δ |
|---|---|---:|---:|---:|
| TallyQA | gemma3-27b-it | 0.012 | 0.212 | **+0.200** |
| TallyQA | gemma3-4b-it | 0.036 | 0.205 | +0.169 |
| TallyQA | internvl3-8b | 0.003 | 0.151 | +0.149 |
| TallyQA | llava-onevision-7b-ov | 0.008 | 0.087 | +0.080 |
| TallyQA | qwen2.5-vl-7b-instruct | 0.002 | 0.101 | +0.099 |
| TallyQA | qwen2.5-vl-32b-instruct | 0.006 | 0.108 | +0.102 |
| ChartQA | gemma3-27b-it | 0.000 | 0.323 | **+0.323** |
| ChartQA | gemma3-4b-it | 0.045 | 0.305 | +0.260 |
| **ChartQA** | **internvl3-8b** | **0.089** | **0.000** | **−0.089** |
| ChartQA | llava-onevision-7b-ov | 0.000 | 0.194 | +0.194 |
| ChartQA | qwen2.5-vl-7b-instruct | 0.000 | 0.217 | +0.217 |
| ChartQA | qwen2.5-vl-32b-instruct | 0.000 | 0.163 | +0.163 |
| MathVista | gemma3-4b-it | 0.174 | 0.584 | **+0.410** |
| MathVista | gemma3-27b-it | 0.033 | 0.462 | +0.429 |
| MathVista | internvl3-8b | 0.078 | 0.200 | +0.122 |
| MathVista | qwen2.5-vl-32b-instruct | 0.000 | 0.236 | +0.236 |
| PlotQA | gemma3-4b-it | 0.092 | 0.479 | **+0.387** |
| PlotQA | gemma3-27b-it | 0.003 | 0.338 | +0.336 |
| **PlotQA** | **internvl3-8b** | **0.134** | **0.000** | **−0.134** |
| PlotQA | qwen2.5-vl-32b-instruct | 0.000 | 0.176 | +0.176 |
| InfoVQA | gemma3-27b-it | 0.019 | 0.454 | **+0.435** |
| InfoVQA | gemma3-4b-it | 0.068 | 0.406 | +0.338 |
| **InfoVQA** | **internvl3-8b** | **0.186** | **0.030** | **−0.156** |

(Full 85-cell table in `_data/L1_confidence_quartile_long.csv`.)

**InternVL3-8b shows H7 reversal on 3 of 5 datasets (PlotQA / ChartQA /
InfoVQA).** Least-confident records anchor *less*, not more — the most
confident records (Q1) are pulled hardest. This is the panel-side
analogue of the same model's H2 collapse (near-zero wrong − correct
gap on the same datasets, see `E7-plotqa-infovqa-evidence.md` §4).

Two readings consistent with the rest of the panel:

1. **Robustness via format-locking, not graded-uncertainty modulation.**
   InternVL3 emits "Based on..." prose despite the JSON-strict prompt
   (~30 % parse-failure on E4 Phase 1, see roadmap §9). When confidence
   is high, the model emits whatever its prose preamble says; when
   confidence is low, the prose more often contains the *gt* string
   rather than the anchor — so the anchor "doesn't stick" on Q4 the
   way it does for other models. Tests: re-run InternVL3 with
   `--max-new-tokens 16` to capture more of the prose tail.
2. **Selection effect from baseline em.** InternVL3's `em(b)` is very
   low on chart-stack datasets (PlotQA 0.019, ChartQA ~0.05). The
   high-confidence records (Q1) are over-represented among the few
   correct-base sids; on those, the model trusts its parse more,
   making Q1 *more* anchor-vulnerable. Tests: stratify InternVL3 cells
   by `base_correct` first, then by quartile within each.

The reversal **does not propagate** into the panel mean (51/85 still
monotone); it's an InternVL3-specific anomaly. But the §6 prose should
acknowledge it as a cell where the "graded confidence pull" reading
breaks down, similar to the E5e γ-β reasoning-mode cell
(`E5e-mathvista-reasoning-evidence.md` §3.1) — different mechanism,
same surface symptom.

> **Closed (2026-05-10): InternVL3-8b removed from active panel; no
> further investigation.** The two hypothesis-tests proposed above
> (max-new-tokens 16 rerun; base-correct stratification) were both
> executed in §B2 / §B3 below and falsified. Model removal closes the
> follow-up loop.

#### B2 follow-up (2026-05-04): wrong-base / correct-base × quartile decomposition

To distinguish hypotheses (a) and (b), we stratified InternVL3-8b's
a-arm pair records by `exact_match_b` (correct vs wrong base) before
computing Q1 / Q4 within each subset. Hypothesis (b) (selection
effect from very low em(b)) predicts the reversal should *vanish* on
wrong-base records — that's where the population is large and not
selected on confidence. Instead:

| dataset | base | n | Q1 df | Q4 df | Δ Q4 − Q1 |
|---|---|---:|---:|---:|---:|
| ChartQA | wrong | 514 | 0.117 | 0.000 | **−0.117** |
| ChartQA | correct | 115 | 0.036 | 0.000 | −0.036 |
| InfographicVQA | wrong | 861 | 0.158 | 0.005 | **−0.154** |
| InfographicVQA | correct | 211 | 0.077 | 0.055 | −0.022 |
| MathVista | wrong | 218 | 0.074 | 0.018 | −0.056 |
| MathVista | correct | 143 | 0.029 | 0.053 | +0.024 |
| PlotQA | wrong | **4,610** | 0.046 | 0.000 | −0.046 |
| PlotQA | correct | 97 | 0.042 | 0.040 | −0.002 |
| TallyQA | wrong | 6,934 | 0.015 | 0.082 | **+0.067** |
| TallyQA | correct | 31,311 | 0.002 | 0.054 | +0.052 |

The reversal **persists on wrong-base records** for 4 of 5 datasets
(ChartQA, InfographicVQA, MathVista, PlotQA). It's actually
*strongest* on wrong-base for InfoVQA (Δ −0.154 at n=861), strong on
ChartQA (Δ −0.117 at n=514), and unambiguous on the n=4,610 PlotQA
wrong-base cell. Hypothesis (b) is **falsified**: the reversal isn't
a small-sample artefact of the correct-base subset.

TallyQA is the only dataset that retains the normal Q4 > Q1
direction on InternVL3 (Δ +0.067 wrong, +0.052 correct). TallyQA's
answer space is 0-9 single digits — the prose preamble that
dominates other datasets ("Based on...") doesn't materialise around
single-digit answers, so the format-locking pathway is shut off.
This corroborates hypothesis (a): the reversal correlates with
datasets where the model is most likely to emit a prose wrapper.

**Provisional takeaway for §6.6 paper prose**: hypothesis (b)
selection effect is falsified; format-locking-via-truncation is the
remaining candidate. **Updated 2026-05-04** — the B3 follow-up below
also falsifies the truncation form of hypothesis (a).

#### B3 follow-up (2026-05-04): max-new-tokens 16 does NOT restore monotonicity

Smoke-test of the format-locking hypothesis: re-ran InternVL3-8b on
InfoVQA with `--max-new-tokens 16` (vs panel default 8) on n=250
sids; results vs the existing max=8 full run on n=1147:

| condition | max=8 (n=865 wrong) | max=16 (n=193 wrong) |
|---|---:|---:|
| Q1 df (most confident) | 0.111 | 0.146 |
| Q4 df (least confident) | 0.014 | 0.000 |
| **Δ Q4 − Q1 wrong-base** | **−0.097** | **−0.146** |

| condition | max=8 (n=211 correct) | max=16 (n=38 correct) |
|---|---:|---:|
| Q1 df | 0.135 | 0.222 |
| Q4 df | 0.018 | 0.000 |
| **Δ Q4 − Q1 correct-base** | **−0.116** | **−0.222** |

The reversal **persists, in fact slightly stronger** at max=16. The
all-base panel-level df is very close between the two runs
(max=8: df(a) = 0.155, df(m) = 0.138; max=16: df(a) = 0.180,
df(m) = 0.185), so max-tokens isn't gating overall anchor behaviour —
just truncation is not the cause of the Q1 > Q4 inversion.

Both candidate mechanisms (selection effect, format-locking via
truncation) are now falsified. The InternVL3 chart-stack reversal is
a genuine "high-confidence records anchor more" pattern. Possible
remaining mechanisms (left for §8 future work):

- **Confident-confabulation hypothesis.** InternVL3's high-confidence
  records on chart-stack data are the ones where the gt is plausibly
  visible in the chart; the model commits to a parse, and the anchor
  digit gets emitted as the parsed token *because* it matches a
  visual digit pattern the model was already locked onto. Low
  confidence → genuine uncertainty → the parsed first digit is
  whichever number bubbled up from the chart context, often gt-like.
- **Per-architecture interaction with InternViT-300M.** The other
  panel models use SigLIP / CLIP-ViT / Qwen-ViT / ConvNeXt; InternVL3
  is the only InternViT model in our panel. The reversal coincides
  with this encoder family — a single-model finding that needs
  cross-encoder testing to be falsifiable.

Paper-side action: cite the reversal as a documented but not yet
mechanistically explained boundary case. §6.6 paper prose updated to
match.

> **Closed (2026-05-10): model removed from active panel.** The
> confident-confabulation and InternViT-300M-encoder hypotheses above
> are no longer queued for §8 future work; they remain on record as
> historical mechanism candidates for the chart-stack reversal anomaly.

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
