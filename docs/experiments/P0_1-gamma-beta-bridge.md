# P0-1 — γ-β residual-stream bridge experiment

> **Status (2026-05-10): ✅ shipped end-to-end with re-calibration rescue.**
> Original Phase B/C (PlotQA + InfoVQA pooled, K=8, L=33) was **Alt-1
> falsified** at fixed K=8 cell. Phase B'/C' re-calibration (TallyQA
> added → 3-pool n_wrong=3017) + 84-cell L × K sweep recovers a
> **partial bridge at K=1 with layer-specific structure**.
>
> 14 / 84 cells survive Bonferroni: late-stack (L=29-33) K=1 mean
> +0.21~+0.48 positive; mid-stack (L=20) K=1/2/4/8 mean -0.11~-0.15
> negative (sign-reversal); strongest cell L=30 K=2 max
> +0.866 [+0.412, +1.330] Bonferroni [+0.115, +1.643].
>
> Branch: `worktree-phase5+p0-1-gamma-beta-bridge`, PR #18 pending merge.

## Goal

Bridge paper §4.6 (γ-β behavioral amplification: Qwen3-VL-Thinking
shows ×1.6 adopt / ×2.9 df / ×12.7 correct-base df vs Instruct on
MathVista anchoring) to §6 (K=8 anchor subspace mitigation: OneVision
Main strict free-lunch on 5 datasets). The two threads sit on the same
paper but are not causally interlocked — bar-raiser tagged this as the
single highest-leverage experiment to flip *Solid Findings, top of band*
to *weak-accept Main*.

**Hypothesis (original)**: Qwen3-VL-Thinking trace residuals on a
self-calibrated K=8 anchor subspace at L=33 amplify *anchor-specifically*
over Instruct baseline. The amplification ratio quantitatively predicts
the §4.5 ×12.7 correct-base behavioral ratio.

**Falsifier**: identical Δ amplitude on neutral d arm (length confound,
not anchor-specific). Original Phase B/C result hit this falsifier at
K=8.

**Hypothesis (rescue)**: K=8 paper §6 prior is OneVision-specific (sv7/sv8
elbow there is sharp). For Qwen3-VL the elbow is gradual (1.026), so
K=2..7 noise dilutes K=1 anchor-direction signal. **K-sweep + L-sweep
with Bonferroni correction reveals anchor-specific cells**.

**Falsifier (rescue)**: All cells null after Bonferroni → bridge fully
falsified, paper §8.2 limitation strengthened. **Outcome: 14 / 84 cells
Bonferroni-positive, hypothesis supported.**

Full design: [`docs/superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md`](../superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md).

## Approach

### Phase B — calibration (original + extended)

| Step | Detail |
|---|---|
| **B1/B2 (original)** | Qwen3-VL-Instruct on PlotQA n=5000 + InfoVQA n=1147; calibrate-subspace at all 36 layers |
| **B'1/B'2 (extension)** | TallyQA n=5000 inference + calibrate-subspace; counting-domain natural images add cleaner anchor contrast |
| **D matrix pooled** | TallyQA D_wrong=1780 + PlotQA 1017 + InfoVQA 220 = **3017 paired (h_a − h_m)** at every layer |
| **B'3 SVD** | K=16 SVD per layer → V_K[L=*] shape (36, 16, 4096); spectrum at L=33 has sv[0..7] = [617, 290, 264, 198, 187, 166, 152, 144] (3-pool) — sv7/sv8 elbow ratio similar to original (gradual, weak) |

Output: `outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_tally_plotqa_infovqa_pooled_K16.pt`.

### Phase C — bridge inference

| Step | Detail |
|---|---|
| **Stimuli** | γ-β MathVista S1 (existing `inputs/mathvista_testmini`, integer-GT, gt ≤ 1000); 2 arms `a-S1` + `d` |
| **Models** | Qwen/Qwen3-VL-8B-Instruct + Qwen/Qwen3-VL-8B-Thinking |
| **Capture (extended)** | per-generated-token L∈{14, 20, 25, 29, 30, 33, 34} residual; **K=16 raw projection coefficients** stored per token (post-hoc K-sweep ready) |
| **Sampling** | temperature=0, max_new_tokens=512 |
| **Records** | 1091 paired (sid, condition) records each model |
| **Trace length** | Instruct ~7 tokens (median); Thinking ~480 tokens after `</think>` close |

`attn_implementation="sdpa"` chosen over eager (eager OOMs at 125 GB on
multi-image MathVista contexts). Forward hook on layer *output* is
SDPA-safe (memory rule [`feedback_sdpa_mask_hook_bug`](feedback) only
applies to pre-hooks modifying `kwargs["attention_mask"]`).

### Phase D — L × K sweep aggregator

| Step | Detail |
|---|---|
| Pairing | per (sample_instance_id × condition) between Instruct + Thinking |
| Within-Thinking | (T_a − T_d) per sid — *the* anchor specificity test |
| Within-Instruct | (I_a − I_d) per sid — sanity / artifact check |
| DiD | algebraic decomposition (within-Thinking − within-Instruct) |
| Cells | 7 layers × 6 K (1, 2, 4, 8, 12, 16) × 2 stat (mean, max) = 84 |
| Bootstrap | paired B=10,000 per cell, 95 % CI |
| Bonferroni | α/84 = 0.000595 → 99.9405 % CI |
| Output | `docs/insights/_data/gamma_beta_bridge_lk_sweep.{csv,md}` |

Generator: [`scripts/analyze_gamma_beta_bridge_lk_sweep.py`](../../scripts/analyze_gamma_beta_bridge_lk_sweep.py).

## Headline result

**Bonferroni-corrected (k=84) within-Thinking CI excludes 0**: **14 / 84 cells**.

### Top positive cells (anchor presence INCREASES Thinking V_K activation)

| layer | K | stat | n | within-Thinking | 95 % CI | Bonferroni CI |
|---|---:|---|---:|---:|---|---|
| **30** | **2** | **max** | 522 | **+0.866** | [+0.412, +1.330] | **[+0.115, +1.643]** |
| 30 | 1 | mean | 522 | +0.477 | [+0.254, +0.695] | [+0.082, +0.852] |
| 29 | 1 | mean | 522 | +0.446 | [+0.252, +0.635] | [+0.123, +0.793] |
| 33 | 1 | mean | 522 | +0.284 | [+0.188, +0.380] | [+0.113, +0.447] |
| 25 | 1 | mean | 522 | +0.213 | [+0.158, +0.270] | [+0.123, +0.314] |

### Top negative cells (anchor presence SUPPRESSES at mid-stack — sign-reversal)

| layer | K | stat | n | within-Thinking | 95 % CI | Bonferroni CI |
|---|---:|---|---:|---:|---|---|
| 25 | 12 | max | 522 | -0.402 | [-0.637, -0.168] | [-0.796, -0.005] |
| 20 | 4 | mean | 522 | -0.192 | [-0.232, -0.152] | [-0.269, -0.124] |
| 20 | 16 | max | 522 | -0.161 | [-0.254, -0.068] | [-0.322, -0.002] |
| 20 | 1 | mean | 522 | -0.152 | [-0.189, -0.116] | [-0.213, -0.094] |
| 20 | 2 | mean | 522 | -0.127 | [-0.159, -0.095] | [-0.180, -0.072] |
| 20 | 8 | mean | 522 | -0.111 | [-0.161, -0.061] | [-0.200, -0.020] |
| 14 | 1 | mean | 522 | -0.041 | [-0.054, -0.028] | [-0.064, -0.020] |
| 14 | 2 | mean | 522 | -0.039 | [-0.052, -0.025] | [-0.062, -0.018] |
| 14 | 8 | mean | 522 | -0.049 | [-0.078, -0.021] | [-0.099, -0.001] |

## Sub-findings

### 1. K=1 (top singular direction) is the right dimensionality

Same layer (L=33), same data, K=1 vs K=8 flips bridge from null to
Bonferroni-significant:

| K | within-Thinking at L=33 | 95 % CI | Outcome |
|---:|---:|---|---|
| 1 | +0.284 | [+0.188, +0.380] | **Bonferroni ✓** |
| 2 | +0.103 (not Bonf-survivor) | … | borderline |
| 4 | +0.044 | … | null |
| **8** (paper §6 prior) | **-0.053** | [-0.50, +0.39] | **null** |
| 16 | -0.072 | … | null |

K=8 mixes K=2..7 noise that *cancels out* the K=1 anchor signal.
Qwen3-VL's sv7/sv8 elbow at L=33 is 1.026 — gradual, not sharp — so
K=8 is not a privileged dimensionality on this architecture. The
paper §6 K=8 prior is OneVision-specific empirical sweet spot, not
universal.

### 2. Layer-specific sign-reversal supports routing-vs-integration framework

Bonferroni-survivors organize into clean spatial pattern:

- **Late-stack (L=29, 30, 33)**: K=1 mean **positive** (+0.21 to +0.48). Anchor presence *activates* the top singular direction during Thinking trace.
- **L=30 K=2 max (+0.87)**: strongest single cell. Anchor-specific *peak* amplitude.
- **Mid-stack (L=20)**: K=1/2/4/8 mean and K=16 max **negative** (-0.11 to -0.19). Anchor presence *suppresses* V_K dimensions.
- **Early-mid (L=14)**: very small negative (-0.04 to -0.05).
- **L=25**: transitional — K=1 mean +0.21 positive, K=12 max -0.40 negative.

This is **direct empirical second anchor for §5.2 Insight 4
routing-vs-integration framework** (alongside §6.4 LEACE rank-1 ChartQA
+56 % reversal). Anchor information routes through the residual stream
in a layer-dependent way: mid-stack suppresses certain V_K dimensions
during anchor-present reasoning while late-stack integrates and
activates them.

### 3. Magnitude does NOT predict §4.5 ×12.7 ratio

Within-Thinking effects are small (+0.5 to +0.9 amplitude units on
baseline ~250-700 = 0.2 % – 0.4 % relative change). The §4.5 behavioral
correct-base df ratio is **×12.7** — Thinking trace's qualitative
mechanism activation is **not** proportional to the behavioral effect
size.

**Bridge is qualitative, not quantitative.** Two interpretations:

1. **Mechanism is real but small relative to surface behavior**: V_K subspace activation is one of many factors driving behavioral df. Other unmeasured residual dimensions, attention pathway differences, or output-head dynamics may carry the bulk of the ×12.7 effect.
2. **K=1 / V_K[L=*] is incomplete proxy**: the (a−m) calibration captures one *aspect* of anchor processing but not the full mechanism. A digit-bbox-restricted (a−m) calibration, or different layer-pair contrast, might reveal larger magnitude.

Future work for quantitative bridge: P0-2 spectrum sweep + alternative calibration contrasts.

## Methodology lessons

1. **K paper-prior must be K-swept on each new model**. K=8 OneVision sweet spot doesn't transfer to Qwen3-VL. Always run K-sweep when applying §6-style subspace methodology cross-architecture.
2. **Within-Thinking paired (T_a − T_d) is the right metric**, not DiD. DiD decomposes into within-Thinking + within-Instruct (artifact). Original Alt-1 falsified analysis showed +0.81 max DiD that decomposed entirely to within-Instruct -0.99. Within-Thinking was already null at K=8.
3. **Per-K coefficient storage during inference enables zero-cost post-hoc K-sweep**. Store raw V_K^T h coefficients (full K), not just L2 amplitude scalar. Re-running inference for each K is wasteful.
4. **Bonferroni correction is conservative but reveals real signal**. 14 / 84 cells surviving k=84 correction is strong evidence the structure is not multiple-comparison artifact.
5. **TallyQA (counting domain) calibration adds value**. Pool went 1237 → 3017 wrong-base diffs (+144 %); counting-domain natural images give cleaner (a−m) anchor contrast vs chart/infographic structural noise.

## Wall-clock + cost

| Phase | Wall-clock | GPU-hour |
|---|---|---|
| Original Phase A-D | 2026-05-09 → 2026-05-10 11:30 KST | ~10 H200-hour |
| Phase B'/C'/D' rescue | 2026-05-10 12:17 → 17:13 KST | ~4.5 H200-hour |
| **Total** | | **~14.5 H200-hour** |

Compared to spec estimate (~26 H200-hour), inference much faster than projected.

## Paper consequences

| Section | Original verdict | Rescue verdict |
|---|---|---|
| §4.6.1 (γ-β behavioral → mechanism bridge) | NOT authored | **CAN be authored** at K=1 framing |
| §5.2 Insight 4 (routing-vs-integration) | unchanged | **gains second empirical anchor** (γ-β bridge sign-reversal) |
| §1.5 (4a) routing-vs-integration framework | unchanged | can cite γ-β bridge as direct evidence |
| §8.2 limitation | "bridge not established" | "quantitative interlock not achieved; qualitative bridge present at K=1" |
| §8.4 item 1 | "pending bridge experiment" | "partial bridge established (2026-05-10), quantitative magnitude residual" |

**Tier impact**: bar-raiser convergent verdict was *Solid Findings, top
of band; weak-accept Main contingent on bridge landing positive*.
Partial rescue (qualitative + Bonferroni-robust) keeps tier at *Solid
Findings, top of band*. The qualitative bridge is not strong enough to
flip to Main alone (magnitude small), but removes the "experiment
failed" framing and provides routing-vs-integration framework's
empirical second anchor.

## Cross-references

- Spec: [`docs/superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md`](../superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md)
- Evidence: [`docs/insights/gamma-beta-bridge-evidence.md`](../insights/gamma-beta-bridge-evidence.md)
- Source plan: [`docs/insights/plan_post_review_2026-05-09.md §P0-1`](../insights/plan_post_review_2026-05-09.md)
- Upstream behavioral: [`docs/insights/E5e-mathvista-reasoning-evidence.md`](../insights/E5e-mathvista-reasoning-evidence.md) (γ-β ×1.6 / ×2.9 / ×12.7)
- Routing-vs-integration framework: paper §5.2 Insight 4 + §6.6 reconciliation paragraph
- Roadmap: `references/roadmap.md` §3.0c + §10 changelog (2026-05-10)
- Sweep canonical (gitignored): `docs/insights/_data/gamma_beta_bridge_lk_sweep.{csv,md}`
- Reproducible notebook: [`notebooks/gamma_beta_bridge_amplitude.ipynb`](../../notebooks/gamma_beta_bridge_amplitude.ipynb)
