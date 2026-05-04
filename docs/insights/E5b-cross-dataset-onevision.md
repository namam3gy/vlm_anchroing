# E5b cross-dataset extension — OneVision Main, 4 chart/figure/math datasets

> **2026-05-04 expansion.** Phase 2 OneVision-only 5-stratum E5b on the
> four cross-datasets (MathVista, ChartQA, InfoVQA, PlotQA) that
> previously had only S1 in the main matrix. This fills the explicit
> "GT range 0..8 only" gap called out in `E5b-anchor-distance-evidence.md`
> — those original numbers were on llava-interleave-7b × {VQAv2, TallyQA}
> with GT capped at 8, leaving open whether the plausibility-window claim
> survives on full GT range and on a different architecture.

**Status:** Sub-experiment of E5b, OneVision Main extension. Source:
`outputs/experiment_e5b_5strat_<ds>_onevision/llava-onevision-qwen2-7b-ov/<ts>/predictions.jsonl`
× 4. Per-cell aggregation: `docs/insights/_data/experiment_e5b_5strat_<ds>_onevision_per_cell.csv`
(via `scripts/analyze_e5e_wrong_correct.py`). Decay summary: `docs/insights/_data/e5b_5strat_decay_per_dataset.{csv,md}`
(via `scripts/build_e5b_5strat_decay_summary.py`). Run script: `scripts/_phase1_e5b_5strat_onevision_queue.sh`.

## What we tested

For each dataset, the full 5-stratum E5b stimulus matrix (b + a/m × {S1..S5} + d
= 12-cond) under `anchor_distance_scheme: relative`. Same dataset slices as
the existing 4-cond main-matrix runs (`experiment_e5e_<ds>_full`):

| Dataset | n_samples | GT range | n_records (12-cond) |
|---|---:|---|---:|
| MathVista | 385 | integer-only, ≤1000 | 4,578 |
| ChartQA | ~705 | numeric ≤1000 | 8,260 |
| InfoVQA | ~1,147 | numeric ≤1000 | 13,546 |
| PlotQA | ~5,000 | numeric ≤1000 | 58,874 |

Total: 85,258 records on 2 H200 with DataLoader prefetch + 2-shard
sharding, ~3.5 h wall-clock end-to-end.

OneVision was chosen over the full 6-model panel for cost: the cross-dataset
distance-decay claim is a shape claim about the headline model, not a
ranking claim. If the shape replicates on the Main, that is the §5 result;
the panel is not load-bearing here.

## Headline — adopt decays sharply, df decays gently, em is stable

Wrong-base slice (`base_correct=False` — the audience for anchor pull),
S1 → S5 deltas:

| Dataset | n(S1) | Δadopt(a) | Δdf(a) | Δem(a) |
|---|---:|---:|---:|---:|
| MathVista | 171 | **-0.089** | +0.003 | -0.018 |
| ChartQA | 226 | **-0.024** | -0.040 | +0.011 |
| InfoVQA | 443 | **-0.043** | -0.049 | -0.014 |
| PlotQA | 2,316 | **-0.079** | -0.062 | -0.008 |

Three findings:

1. **adopt_rate decays monotonically S1 → S5 on every dataset.** The
   plausibility-window prediction (S&M 1997: anchor admission requires
   the value to be in the plausible range for the target judgment)
   replicates on OneVision-7B-OV at full GT range. PlotQA, with the
   widest GT support, shows the largest decay (8.7 % → 0.7 %).
2. **df decays gently** (-0.04 to -0.06) on three of four datasets.
   MathVista is the exception (+0.003 — flat). MathVista has the
   smallest wrong-base support (n=171 at S1) and the most clipped GT
   distribution (integer ≤ 1000 with mode shape skewed toward small
   values), which compresses the across-stratum anchor-distance contrast.
3. **em is stable across strata** (Δ ranging -0.018 to +0.016). Anchor
   distance does not damage em beyond the initial drop from d-arm to
   a-arm. Once an anchor is rejected, the model returns roughly to its
   d-arm em rate; once an anchor is admitted, the redirect cost is paid
   regardless of distance.

Full per-dataset × per-arm × per-metric × per-stratum table (incl. m-arm):
`docs/insights/_data/e5b_5strat_decay_per_dataset.md`.

## Cross-dataset shape — adopt vs df

The key finding is that **adopt and df decay at different rates**, and
this is consistent across the four cross-datasets:

- adopt collapses by a factor of 5–13× from S1 to S5 (PlotQA: 8.7 % → 0.7 %,
  MathVista: 10.5 % → 1.6 %, InfoVQA: 4.5 % → 0.2 %, ChartQA: 2.8 % → 0.4 %).
- df shrinks by 22–30 % relative (PlotQA: 21 % → 15 %, InfoVQA: 19 % → 15 %,
  ChartQA: 18 % → 14 %), or stays flat (MathVista).

This is consistent with the M2 finding that adopt and df measure
different things: adopt is a binary "did the anchor win as the answer"
test that requires the anchor to be in the plausible range; df is a
sign-based "did the prediction move toward the anchor" test that
captures partial bias even when the anchor is not literally adopted.
Far anchors are rejected as candidates (low adopt) but their direction
still leaks into the prediction (non-zero df at S5 = sub-threshold pull).

The original E5b on llava-interleave-7b VQAv2/TallyQA showed
adopt collapsing from ~13 % at S1 to ~0 at S5; the OneVision cross-dataset
result reproduces this collapse at a different absolute level (lower
adopt at S1 because OneVision is more anchor-resistant overall, but the
same monotonic shape).

## m-arm — pure visual co-presence still produces direction-follow

The m-arm (anchor digit pixel-inpainted) is the control for "is the
effect digit-driven, or just visual co-presence". On all four datasets,
m-arm adopt is much lower than a-arm adopt (e.g., PlotQA S1: a=0.087,
m=0.025 — a 3.5× ratio) and decays to 0 by S4 in three of four datasets.
But m-arm df is non-trivial (~0.13–0.20 at S1 across datasets), and it
also decays with stratum.

Reading: a fraction of df on a-arm is from visual co-presence (matched
by m-arm); the rest is digit-driven (a > m gap). Distance gates both —
both arms decay with stratum.

## Why this matters

For §5, this is the cross-dataset replication of the plausibility-window
claim. The original E5b finding ("anchor adoption decays sharply with
anchor-to-GT distance, gated by uncertainty") is now backed by:

- **2 architectures** (llava-interleave-7b in the original; OneVision-7B-OV here)
- **6 datasets total** (VQAv2 + TallyQA in the original; ChartQA + InfoVQA +
  MathVista + PlotQA here)
- **GT ranges from {0..8} to ≤1000** (PlotQA spans the full range)

The shape — adopt monotonic decay, df gentle decay, em stable — replicates.
The plausibility-window mechanism is not specific to small-GT datasets
or a single architecture.

## What this doesn't say

- **Single model.** All numbers are OneVision-7B-OV. Cross-model
  generalization on 5-stratum is in `E5c-anchor-mask-evidence.md`
  (qwen2.5-vl-7b on VQAv2 + TallyQA at GT ≤ 8); 5-stratum on the full
  6-model panel × 4 cross-datasets is not run.
- **Stratum boundaries are dataset-relative.** `anchor_distance_scheme:
  relative` defines S1–S5 by quantile of `|a − GT|` within the dataset's
  anchor inventory, not by absolute distance. Cross-dataset comparison
  of "S1" therefore means "the closest quintile of available anchors for
  this dataset", not "anchors at distance ≤ k". This is the correct
  comparison for the plausibility-window claim (the question is whether
  the model treats relatively-close anchors differently from
  relatively-far anchors), but it should not be read as "S1 has the same
  absolute distance across datasets".
- **Single prompt.** Identical system prompt across all four datasets;
  paraphrase robustness untested.
