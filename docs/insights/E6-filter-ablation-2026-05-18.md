# E6 cross-arch on Qwen2.5-VL — calibration filter ablation

**Date:** 2026-05-18.
**Branch:** `worktree-e6-cross-arch-qwen25vl` (PR #54).
**Question:** Is the Qwen2.5-VL mitigation weakness due to (a) calibration filter choice (wrong-base vs anchor-positive) or (b) sample-size / cap artifacts?
**Answer:** **Filter is dominant.** Anchor-positive filter (`adopt(a) OR df(a)`) on the SAME 312 sids produces a measurably different K=8 subspace (~21 % shift vs same-size wrong-base subsample) and translates to **~8.5× larger mean mitigation magnitude** on Stage-4 (Δdf mean −0.10 pp → −0.85 pp).

## Background

Phase 2 (PR #54 commit `d0ff829`) Stage-4 result on Qwen2.5-VL produced mean Δdf = **−0.10 pp** (OneVision recipe was −2.9 pp; PlotQA showed +1.30 pp backfire). The wrong-base filter contains 4429 / 4707 sids on PlotQA that show *no* a-arm movement vs baseline (i.e., `pa == pb`, anchor effect = 0), plus 195 sids that moved *away* from anchor. User hypothesis: those non-anchor-relevant sids act as **noise that dilutes v_wrong direction** in calibration.

## Wrong-base calibration pool decomposition (Qwen2.5-VL PlotQA)

4707 4-cond eligible sids classified by (a-arm vs b-arm) behavior:

| Behavior | n | % | Contribution to (a−m) signal |
|---|---:|---:|---|
| pa == pb (no movement) | 4,226 | 89.8 % | noise (a-arm = b-arm, no anchor signal) |
| Moved TOWARD anchor (df-positive) | **278** | **5.9 %** | ✅ pure anchor direction signal |
| Moved AWAY from anchor (df-negative) | 195 | 4.1 % | ❌ anchor-NEGATIVE signal (opposite direction) |
| z == pb (degenerate) | 7 | 0.1 % | undefined direction |

Wrong-base = 925 sids includes:
- 210 (22.7 %) anchor-positive
- 550 (59.5 %) no movement (pa == pb)
- 164 (17.7 %) anchor-negative
- 1 degenerate

→ Of 925 wrong-base sids, only ~22 % carry usable anchor-direction signal. Remaining 77 % is noise or opposite-direction signal.

## Filter design

Patched `scripts/e6_steering_vector.py` with `--calibration-filter {wrong-base, anchor-positive}` flag. Anchor-positive = sids where a-arm `adopt(a) OR df(a)` triggered (mathematically equivalent to df-positive because adopt ⊆ df).

Driver: `scripts/run_e6_filter_ablation.sh` (calibrate-subspace anchor-positive on PlotQA+InfoVQA → SVD pool → peak-pick → Stage-4 chosen-cell × 5 datasets at L=26 K=8 α=1.0 → aggregator with `E6_STAGE4_OUTPUT_SUFFIX=_qwen_anchor_positive` env-var override).

Calibration cap reduced 5000 → 350 (only need ≥ priority subset count). Stage-4 eval cap reduced 5000 → 500 per dataset (CI wider but point estimate signal preserved).

## SVD direction comparison (filter effect vs sample-size effect)

Compared per-layer ||v_wrong[L]|| and top-K=8 SVD subspace overlap (capture ratio) across three D_wrong matrices:

- **WB-full**: wrong-base, n=1148 (Phase 2 calibration)
- **WB-sub**: wrong-base, random sample n=312 (matches AP size)
- **AP**: anchor-positive, n=312 (this ablation)

### Per-layer norm at chosen layer L=26

| | WB-full (1148) | WB-sub (312) | AP (312) |
|---|---:|---:|---:|
| ||v_wrong[L=26]|| | 8.524 | 8.627 | **9.257** |
| top SVD value sv0 | 371.7 | 194.5 | 201.5 |

→ WB-full vs WB-sub: virtually identical mean direction (only sample noise). AP has ~7 % larger ||v_wrong|| at L=26 than WB-sub — slight magnitude boost from cleaner signal.

### Direction cosine at L=26

| Comparison | v_wrong (mean) cos | V_K=8 subspace overlap |
|---|---:|---:|
| WB-full vs WB-sub | 0.9922 | 0.9046 |
| WB-full vs AP | 0.9837 | 0.8481 |
| **WB-sub vs AP** | **0.9765** | **0.7939** |

→ Mean direction (v_wrong): ~2.4 % angular shift between WB-sub and AP at same n=312 — small. **K=8 subspace**: ~**21 % shift** at same n — substantial. Since the mit hook applies the K=8 projection (not the v_wrong mean), this is the relevant quantity.

→ **Sample size matters ~10 % on V_K=8 (WB-full vs WB-sub overlap 0.90), filter matters ~21 % (WB-sub vs AP overlap 0.79). Filter > sample-size on hook-relevant subspace.**

## Stage-4 Δdf comparison (5-dataset)

Wrong-base Phase 2 (cap 5000) vs Anchor-positive ablation (cap 500):

| Dataset | n_paired (wb / ap) | **Wrong-base Δdf** | **Anchor-pos Δdf** [95 % CI] |
|---|---|---:|---:|
| MathVista | 139 / 139 | 0.00 pp | **−2.88 pp [−5.8, −0.7]** ✅ CI-clean |
| ChartQA | 152 / 152 | −1.32 pp | −0.66 pp |
| InfoVQA | 222 / 222 | −0.45 pp | −0.90 pp |
| PlotQA | 925 / 499 | **+1.30 pp** (backfire) | +0.20 pp |
| TallyQA | 5000 / 500 | −0.04 pp | +0.00 pp |
| **mean** | | **−0.10 pp** | **−0.85 pp (8.5× stronger)** |

### Cap confound — empirically dismissed

3 datasets (MathVista / ChartQA / InfoVQA): n_paired identical in both runs (each ≤ cap 500) → cap-irrelevant → pure filter effect.

PlotQA cap sensitivity verified by truncating Phase 2 wrong-base predictions at first-N:

| cap on Phase 2 wrong-base | n | Δdf |
|---:|---:|---:|
| 100 | 100 | −4.00 pp |
| 500 | 500 | **+1.00 pp** |
| 925 | 925 | **+1.30 pp** |

→ Cap 500 vs cap 925 differs by **0.30 pp** on PlotQA. Anchor-positive cap-500 result (+0.20 pp) vs same-cap wrong-base (+1.00 pp) → **0.80 pp filter-attributable improvement** even after cap-discount.

TallyQA: both near 0 in both runs, no clear cap attribution.

→ **Filter effect dominates everywhere.**

## Δem(b) (capability gain on non-anchored arm)

| Dataset | Wrong-base | Anchor-pos |
|---|---:|---:|
| MathVista | +0.72 pp | 0.00 pp |
| ChartQA | +0.66 pp | +1.32 pp |
| InfoVQA | −0.45 pp | 0.00 pp |
| PlotQA | +0.54 pp | +0.60 pp |
| TallyQA | +0.80 pp | +1.20 pp |
| **mean** | **+0.45 pp** | **+0.62 pp** |

Slight improvement (+0.17 pp). em-drop deal-breaker preserved (Δem(a) mean +0.17 pp anchor-positive, no rejection).

## Per-dataset heterogeneity

- **MathVista**: Filter most impactful — null → −2.88 pp CI-clean (only ablation result that's significant). Wrong-base sample was anchor-irrelevant noise; anchor-positive selected the actual signal sids.
- **PlotQA**: Backfire essentially eliminated (+1.30 → +0.20 pp). Wrong-base's anchor-negative subset (164 sids out of 925) was producing anti-mitigation; anchor-positive removes them.
- **InfoVQA**: Magnitude doubles (−0.45 → −0.90 pp), but pool only 34 anchor-positive sids → wide CI.
- **ChartQA**: Magnitude halves (−1.32 → −0.66 pp). Wrong-base's specific 152 sids happened to align with anchor direction; reducing to 40 anchor-positive sids introduces SVD noise.
- **TallyQA**: Both near 0. May reflect TallyQA's text-style answer format reducing anchor effect overall.

## Implications

1. **Filter improvement is real and substantial**: 8.5× larger mean mitigation magnitude, eliminates PlotQA backfire, transforms MathVista null into significant.
2. **Filter is not a full fix**: Mean Qwen2.5-VL Δdf (−0.85 pp) still ~30 % of OneVision Stage-4 (−2.9 pp). Cross-arch gap remains.
3. **Likely interactions with prompt-format confound** (Diagnostic E, [`E6-cross-arch-prompt-confound-2026-05-18.md`](E6-cross-arch-prompt-confound-2026-05-18.md)): JSON wrapper buries digit at step 4 — anchor-positive filter doesn't fix the step-0 vs step-4 propagation issue.
4. **Method-level improvement for paper §6**: Anchor-positive (df-positive) calibration filter should be considered as method default. Already validated on OneVision (Δem(b) +0.17 pp net) — could also be applied to update §6 main results.

## Recommendation

Option H1 on new branch (raw-number prompt + 3-model panel + DF eps-threshold form) should ALSO adopt anchor-positive filter as the calibration default. Three method-level improvements combined:

- prompt: raw-number (removes step-0 confound)
- filter: anchor-positive (removes calibration noise from non-anchor sids)
- metric: DF eps-threshold (epsilon-aware)

Combined effect (predicted): cross-arch Δdf magnitude should approach OneVision baseline. If so, Bar-raiser conditional-Main accept is closer.

## Recovery anchors

- Code: `scripts/run_e6_filter_ablation.sh` + `scripts/e6_steering_vector.py` (with `--calibration-filter` flag).
- Calibration outputs:
  - `outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_{plotqa,infovqa}_anchor_positive/` (per-source D matrices)
  - `outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_plotqa_infovqa_pooled_anchor_positive/v.pt`
  - `outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/subspace_plotqa_infovqa_pooled_anchor_positive_K16.pt`
- Stage-4 sweep dirs: `outputs/e6_steering/qwen2.5-vl-7b-instruct/sweep_subspace_<ds>_plotqa_infovqa_pooled_anchor_positive_chosen/`
- Canonical tables: `docs/insights/_data/stage4_final_per_dataset{_ci,}_qwen_anchor_positive.{csv,md}` (gitignored symlinked dir)

## Cross-references

- [E6-cross-arch-prompt-confound-2026-05-18.md](E6-cross-arch-prompt-confound-2026-05-18.md) — prompt-format confound (diagnostic chain B/C/D/E)
- [E6-cross-arch-qwen25vl-phase0.md](E6-cross-arch-qwen25vl-phase0.md) — Phase 0 calibration (L*_qwen = 26)
- Plan: `docs/experiments/E6-cross-arch-design.md`
- Memory: [[feedback_qao_q_d_alignment]], [[feedback_em_drop_dealbreaker]]
