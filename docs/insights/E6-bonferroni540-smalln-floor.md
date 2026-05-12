# Why we do not report Bonferroni-540 strict CIs on §6.2.3

**Last updated.** 2026-05-11.
**Owner doc.** `docs/paper/emnlp_draft_ko.md` §6.2.3 multiplicity-correction
honest note + §8.4 item 8.
**Status.** Diagnostic only. Reproducible from existing `_data/` artifacts.

## TL;DR

A naive "Bonferroni-540 free recompute" — read the 0.00463 % / 99.99537 %
tail quantiles of the existing B-paired-bootstrap draws to widen the
99.75 % Bonferroni-20 CIs into 99.99 % Bonferroni-540 CIs — looks like
the right answer to Round-4 MAJ-1 (the 27-cell pilot grid argmax is a
separate multiplicity layer beyond the 20-family). It is **not** the
right answer for our two small-n datasets. The bootstrap lower bound
on those two cells lands on the 1/n empirical discretization floor,
not on a statistically informative bound, and parametric normal-
approximation 99.99 % CIs disagree with the bootstrap on the same two
cells (parametric includes zero, bootstrap excludes zero by sitting on
the floor). We retain Bonferroni-20 as the headline correction and
disclose 27-cell selection as a separate multiplicity layer in §6.2.3
prose only.

## Setup

§6.2.3 reports paired-bootstrap CIs on Δadopt / Δdf / Δem(a) / Δem(b)
across five evaluation datasets for cell #17 (L = 26, K = 8, α = 1.0).
Resampling unit = sample-instance id paired between baseline and
mitigation arms; per-arm `(num, den)` recomputed each bootstrap so the
denominators of adopt and df shift correctly per arm. B = 10,000 draws
are stored at `docs/insights/_data/stage4_final_bootstrap_draws.npz`;
99.75 % Bonferroni-20 CIs (5 datasets × 4 metrics = 20 family) are in
`stage4_final_per_dataset_ci.csv`.

Round-4 MAJ-1 (round4_aggressive.md:218–240) asked us either to (i) re-
run with a Bonferroni-540 correction that includes the 27-cell pilot
selection layer, or (ii) disclose the 27-cell selection as a separate
multiplicity layer in §6.2.3 prose. The paper Round-4 response
(round4_response.md:115) chose option (ii) and noted that a "Bonferroni-
540 free recompute" would only widen the existing CIs at the 0.0046 %
tail — no new inference. Round-5 bar-raiser (round5_bar_raiser.md:34)
treated the matter as settled pending pre-camera-ready cleanup.

When pre-camera-ready cleanup actually executed the free recompute,
two empirical findings made strict 540-family CIs noninformative for
small-n cells. Both findings are reproducible from the existing draws.

## Finding 1 — Bonferroni-540 LO sits on the 1/n floor for small n

Running the bootstrap at B = 100,000 (see "Reproduce" below) gives
the following Bonferroni-540 (99.99 %) lower bounds on Δem(b) — the
multiplicity-robust headline metric — and on Δdf(a) for the chosen
cell.

| Dataset       | n      | metric | Δ point | BS Bonf-540 LO | floor?            |
|---------------|-------:|--------|--------:|---------------:|-------------------|
| PlotQA        |  2,306 | em_b   | +4.73   | +2.91          | —                 |
| TallyQA       |  4,978 | em_b   | +13.82  | +11.97         | —                 |
| InfoVQA       |    443 | em_b   | +9.03   | +4.06          | —                 |
| **ChartQA**   |  **224** | **em_b** | **+7.13**  | **+0.89**          | **= 2/n (= 0.893 %)** |
| **MathVista** |  **170** | **em_b** | **+9.40**  | **+0.59**          | **= 1/n (= 0.588 %)** |
| PlotQA        |  2,306 | df     | −5.15   | −8.89          | —                 |
| TallyQA       |  4,978 | df     | −0.34   | −2.13          | —                 |
| InfoVQA       |    443 | df     | −0.68   | −8.80          | —                 |
| ChartQA       |    224 | df     | −4.02   | −15.18         | —                 |
| MathVista     |    170 | df     | −4.12   | −19.41         | —                 |

For ChartQA Δem(b) the bootstrap lower bound is **exactly** 2 ⁄ 224 =
0.0089 ≈ +0.89 %; for MathVista Δem(b) it is **exactly** 1 ⁄ 170 ≈
+0.588 %. Both are the *smallest positive Δ value the paired-bootstrap
on that n can produce*, not a statistical confidence threshold. With
n = 170 only `33` distinct Δem(b) values appear across `100,000` draws
because Δ = (k_a − k_b) ⁄ 170 is discrete; the 0.00463 % tail position
unavoidably lands at the discrete grid edge once fewer than ≈ 4.6 of
100,000 draws fall on the wrong side of zero.

For the larger datasets (n ≥ 443) the Bonferroni-540 lower bound is
well inside the support of the bootstrap distribution and the
discretization grid is fine enough that the tail quantile is
statistically meaningful.

## Finding 2 — Bootstrap and parametric 99.99 % CIs disagree on the same two cells

If we instead apply the normal approximation `Δ ± z_{Bonf-540} · SE`
(`z_{Bonf-540} = Φ^{−1}(1 − 0.05 ⁄ 540 ⁄ 2) ≈ 3.909`) using the
bootstrap-estimated SE, the same two cells include zero.

| Dataset       | n     | metric | BS Bonf-540 LO | Param Bonf-540 LO | bound agreement |
|---------------|------:|--------|---------------:|------------------:|-----------------|
| PlotQA        | 2,306 | em_b   | +2.91          | +2.82             | ✓ both > 0       |
| TallyQA       | 4,978 | em_b   | +11.97         | +11.91            | ✓                |
| InfoVQA       |   443 | em_b   |  +4.06         |  +3.71            | ✓                |
| **ChartQA**   |   **224** | **em_b** |  **+0.89**          |  **−0.05**             | **✗ diverge**        |
| **MathVista** |   **170** | **em_b** |  **+0.59**          |  **−0.50**             | **✗ diverge**        |
| PlotQA        | 2,306 | df     |  −8.89         |  −8.76            | ✓ both < 0       |
| TallyQA       | 4,978 | df     |  −2.13         |  −2.26            | ✓ (both contain 0) |
| InfoVQA       |   443 | df     |  −8.80         |  −8.62            | ✓ (both contain 0) |
| ChartQA       |   224 | df     | −15.18         | −15.45            | ✓ (both contain 0) |
| MathVista     |   170 | df     | −19.41         | −19.49            | ✓ (both contain 0) |

The two diverging cells are exactly the two cells where the bootstrap
LO is on the empirical 1/n floor. Neither method is "the" right
answer at n ≤ 224 — the bootstrap is anti-conservative because it
cannot resample values it has never observed, and the parametric
normal extrapolates beyond the support of a bounded discrete metric —
but the divergence is itself the operative signal: a *statistical*
99.99 % CI at n = 170 cannot land on +0.588 % unless the underlying
data discretization is doing the work. At those two cells, n is the
binding constraint, not B.

## Finding 3 — Increasing B does not relax the floor

The 1/n floor is sample-size limited, not Monte-Carlo limited.
Repeating the diagnostic at B = 100,000 narrows the MC standard error
on every reported bound to ≤ 0.20 pp (split-half check) but does not
move the MathVista Δem(b) lower bound off +0.5882 % nor the ChartQA
Δem(b) lower bound off +0.893 % — those are the smallest positive
values the paired-bootstrap can produce at those n's. The bound is
"stable at the floor", not "stable at a statistical confidence
threshold." The only way to relax it is to add more data
(deferred to §8.4 item 8(b)).

## Decision

1. **Headline correction stays at Bonferroni-20** (5 datasets × 4
   metrics = 20 paired-test family).
   - Δem(b) 5/5 cells excludes zero at Bonferroni-20.
   - Δdf(a) PlotQA n = 2,306 single cell excludes zero at Bonferroni-20.
2. **27-cell selection layer disclosed in §6.2.3 prose only**, with
   pointer to this document for the empirical reason strict 540-family
   CIs are not reported.
3. **§8.4 item 8 split into 8(a) and 8(b).** 8(a) — pre-registered
   single-cell hypothesis test on §4.6 — remains free (re-analysis
   only). 8(b) — strict 540-family correction on §6.2.3 — requires
   ~5 H100-hour additional inference on one or two n ≥ 1,000 datasets
   beyond PlotQA / TallyQA so that the empirical 1/n floor on
   ChartQA / MathVista is no longer the binding constraint.

## Reproduce

```bash
# Existing B = 10,000 draws, stored
ls docs/insights/_data/stage4_final_bootstrap_draws.npz

# Recompute at B = 100,000 (≈ 1 min on 1 CPU)
uv run python scripts/build_e6_stage4_bootstrap_ci.py \
    --bootstrap 100000 \
    --out-dir /tmp/e6_b100k

# Compare BS Bonf-540 LO vs parametric Bonf-540 LO
uv run python - <<'PY'
import numpy as np
from scipy import stats
f = np.load("/tmp/e6_b100k/stage4_final_bootstrap_draws.npz")
z = stats.norm.ppf(1 - 0.05 / 540 / 2)  # ≈ 3.909
n_map = {"plotqa": 2306, "tallyqa": 4978, "infographicvqa": 443,
         "chartqa": 224, "mathvista": 170}
for ds, n in n_map.items():
    for m in ("em_b", "df"):
        d = f[f"{ds}__{m}"]
        bs_lo = float(np.percentile(d, 0.05 / 540 / 2 * 100))
        par_lo = float(d.mean() - z * d.std(ddof=1))
        floor = ""
        if abs(bs_lo - 1 / n) < 1e-4: floor = " (= 1/n)"
        elif abs(bs_lo - 2 / n) < 1e-4: floor = " (= 2/n)"
        print(f"  {ds:<15} n={n:>5} {m:<5}  BS LO {bs_lo*100:+6.2f}pp{floor:<8}"
              f"   Param LO {par_lo*100:+6.2f}pp")
PY
```
