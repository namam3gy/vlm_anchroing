# Design — Mitigation General-Capability Regression Test (E8)

**Date**: 2026-05-08
**Branch**: target `phase2/insight-mining-batch1` (or new sibling)
**Owner**: thyun.park
**Status**: Design approved through §1–§6, awaiting user review of this written spec
**Cross-links**:
- `references/roadmap.md` §7 Phase 4 (planned new entry)
- `docs/paper/sections/07_mechanism_mitigation.md` §7.4.5 (paper-side target, local-only / gitignored)
- `docs/insights/_data/stage4_final_per_dataset.csv` (existing strict-free-lunch evidence within anchoring family)
- `scripts/e6_steering_vector.py:827` (current subspace projection hook source)
- `outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/subspace_*.pt` (precomputed subspace tensor for chosen cell)

---

## §1 Goal & contribution

**Question.** When the §7.4.5 chosen subspace mitigation cell (L=26, K=8, α=1.0) is applied to LLaVA-OneVision-7b at inference time, is general (non-anchoring) VLM capability preserved?

**Why now.** The strict free-lunch claim in §7.4.5 (Δdf -2.9pp + Δem(a) +3.9pp + Δem(b) +8.8pp on paired-sids) is verified **only inside the anchoring task family** (5 anchoring datasets, anchoring metrics). A reviewer's natural follow-up question — "do the K=8 subspace directions carry representations beyond anchor-image bias?" — is currently unanswered.

**Hypothesis.** Because the projection is surgical (prefill last token only, single layer L=26, decode steps untouched at the residual level), general capability is statistically equivalent to baseline (per-benchmark Δ ≥ -1.0pp, macro Δ ≥ -0.5pp).

**Falsifier.** ≥1 benchmark with Δ < -1.0pp, or macro Δ < -0.5pp. In that case the §7.4.5 strict free-lunch claim is downgraded to "favorable trade-off" with explicit capability cost reported.

**Contribution to paper.** New sub-paragraph + table in §7.4.5 ("Capability preservation"). Required to defend the "deployable mitigation" framing per memory `feedback_mitigation_deployability`.

**Out of scope.**
- 5-model panel (other than OneVision-7b Main) — the chosen cell L=26 K=8 α=1.0 was calibrated on OneVision specifically.
- α sweep — fixed α=1.0 only.
- LLM-judge benchmarks (MMMU) — deferred until paper frame is locked (see "Deferred" below).
- Anchoring-axis benchmarks already covered by §3.3 main panel.

---

## §2 Approach (chosen)

**A — In-repo VLMEvalKit subclass with pinned dependency** (chosen over fork-and-submodule and custom-inference alternatives):

- Pin VLMEvalKit at a specific commit in `pyproject.toml` (`vlmeval @ git+https://github.com/open-compass/VLMEvalKit@<commit>`).
- Add `src/vlm_anchor/capability_eval.py` containing a thin `LLaVAOneVisionMitigated` subclass of VLMEvalKit's `LLaVA_OneVision` model wrapper. The subclass installs a forward hook on `self.model.language_model.model.layers[L]` in `__init__`, removes it in `__del__`.
- Hoist `_make_subspace_projection_hook` from `scripts/e6_steering_vector.py` into a new shared module `src/vlm_anchor/hooks.py`. Both anchoring (`e6_steering_vector.py`) and capability eval (`capability_eval.py`) import from this single source of truth, guaranteeing the same projection math is exercised in both contexts.
- Add `scripts/run_capability_eval.py` that registers the subclass with VLMEvalKit's `supported_VLM` dict and orchestrates per-benchmark interleaving (see §6).
- Add `scripts/aggregate_capability_eval.py` for partial + final summary emission.

**Rejected alternatives.** Fork-and-submodule (B) — submodule overhead, upstream-sync burden, weaker reviewer story. Custom inference loop (C) — ~600 LOC reimplementation, scoring-conformity risk, "why not standard?" reviewer question.

---

## §3 Architecture

### Files (new / modified)

```
src/vlm_anchor/
  hooks.py                                 NEW  — _make_subspace_projection_hook lifted from e6
  capability_eval.py                       NEW  — LLaVAOneVisionMitigated subclass
scripts/
  run_capability_eval.py                   NEW  — orchestration driver (interleaved per-bench)
  aggregate_capability_eval.py             NEW  — partial + final CSV/MD emitter + verdict
  e6_steering_vector.py                    MOD  — import hook factory from src/vlm_anchor/hooks
configs/
  capability_eval.yaml                     NEW  — subspace_path / cell / benchmark list / output_root
pyproject.toml                             MOD  — add vlmeval (git-pinned)
tests/
  test_capability_eval.py                  NEW  — hook factory parity test (e6 vs capability_eval)
docs/insights/
  E8-capability-preservation-evidence.md   NEW  — insight doc + reproducer
docs/insights/_data/
  capability_eval_per_benchmark.{csv,md}   NEW  — paper-ready per-benchmark Δ + 95% CI (gitignored)
  capability_eval_partial.{csv,md}         NEW  — running partial summary, updated after each benchmark (gitignored)
docs/paper/sections/
  07_mechanism_mitigation.md               MOD  — §7.4.5 sub-table + 1 paragraph (local-only, gitignored)
outputs/
  capability_eval/                         NEW  — gitignored
    <run_id>/                              VLMEvalKit raw outputs + progress.log
references/
  roadmap.md                               MOD  — Phase 4 new P0 entry, §10 changelog
```

### Hook factory single source of truth

`src/vlm_anchor/hooks.py:make_subspace_projection_hook(V_K, alpha)` — verbatim port of the body of `scripts/e6_steering_vector.py:_make_subspace_projection_hook` (lines 827-849 as of current commit). Behavior:

- At prefill (`hidden.shape[1] > 1`): subtract `α · V_K^T V_K · h` from the last-token residual; decode steps (`shape[1] == 1`) are no-ops, propagation occurs via KV cache.
- Returns `None` if `alpha == 0` (caller installs no hook).

`scripts/e6_steering_vector.py` is modified to `from vlm_anchor.hooks import make_subspace_projection_hook` and remove its local copy. `tests/test_capability_eval.py::test_hook_factory_parity` constructs identical (V_K, α, hidden) inputs and asserts byte-identical output between the old (saved as test fixture) and the new shared implementation.

### LLaVAOneVisionMitigated

```python
class LLaVAOneVisionMitigated(LLaVA_OneVision):
    def __init__(self, *, subspace_path: str, layer: int = 26, K: int = 8,
                 alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        full = torch.load(subspace_path, weights_only=True)  # (n_layers, K_max, d_model)
        V_K = full[layer, :K, :]
        layers = self.model.language_model.model.layers
        assert isinstance(layers[layer], Qwen2DecoderLayer), \
            f"Expected Qwen2DecoderLayer at layer {layer}, got {type(layers[layer])}"
        hook = make_subspace_projection_hook(V_K, alpha)
        self._mit_handle = layers[layer].register_forward_hook(hook)
        self._mit_call_count = 0  # incremented by a wrapper hook for sanity check

    def __del__(self):
        if hasattr(self, "_mit_handle"):
            self._mit_handle.remove()
```

`_mit_call_count` increments via a counter-wrapper added at install time; surfaced at the end of each benchmark for activation sanity check.

### Data flow

```
configs/capability_eval.yaml
  subspace_path: outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/subspace_<scope>.pt
  cell: {layer: 26, K: 8, alpha: 1.0}
  benchmarks: [RealWorldQA, OCRBench, HallusionBench, MMStar, MMBench_DEV_EN]
  output_root: outputs/capability_eval
  partial_csv: docs/insights/_data/capability_eval_partial.csv

scripts/run_capability_eval.py
  for bench in benchmarks:                      # fast-first ordering, see §6
    run_one(variant="baseline", bench)          # vanilla LLaVA_OneVision
    run_one(variant="mit",      bench)          # LLaVAOneVisionMitigated
    aggregate_partial.update(bench)             # writes _partial.csv + appends progress.log
  finalize_summary()                            # _partial → capability_eval_per_benchmark + verdict
```

`run_one(variant, bench)` invokes VLMEvalKit's evaluator programmatically (not via subprocess CLI); raw outputs land in `outputs/capability_eval/<run_id>/<variant>/<bench>/`.

---

## §4 Scoring & reporting

### Per-benchmark primary metric

| Benchmark | n | Split | Metric | Source |
|---|---|---|---|---|
| MMBench-DEV-EN | 4329 | dev (public eval split, NOT held-out test split) | overall accuracy (CircularEval) | VLMEvalKit default |
| OCRBench | 1000 | test | OCR score sum (max 1000), reported as % for table consistency | VLMEvalKit default |
| RealWorldQA | 765 | test | accuracy | VLMEvalKit default |
| MMStar | 1500 | full | accuracy | VLMEvalKit default |
| HallusionBench | 1129 | full | accuracy (combined yes/no + MCQ) | VLMEvalKit default |

**Total**: 8723 questions. **LLM-judge calls**: 0.

### Δ + 95% CI

VLMEvalKit greedy decoding → run-to-run noise effectively zero. Only randomness is question-level Bernoulli:

```
Δ = acc_mit − acc_baseline                       # per benchmark
SE(Δ) ≈ McNemar paired SE on shared question set # because both arms see identical questions
95% CI = Δ ± 1.96 · SE(Δ)
```

OCRBench (sum-style scoring) uses 1000-sample bootstrap on per-question score difference.

### Macro aggregation

```
macro_Δ = mean(Δ_i for i in 5 benchmarks)        # equal weight, OCRBench normalized to %
macro_CI ≈ mean(CI_i)                            # conservative
```

### Verdict (pre-registered)

```python
def verdict(per_bench_deltas: dict[str, float], macro_delta: float) -> str:
    failed = [k for k, v in per_bench_deltas.items() if v < -1.0]
    if failed:
        return f"DEGRADED:{','.join(failed)}"
    if macro_delta < -0.5:
        return "DEGRADED:macro"
    return "STRICT_FREE_LUNCH"
```

Three possible outcomes:
- **STRICT_FREE_LUNCH** — paper §7.4.5 retains the strict claim with this evidence as sub-table.
- **DEGRADED:<bench>** — §7.4.5 downgraded to "favorable trade-off"; cost is reported in prose; mitigation may be re-evaluated.
- **DEGRADED:macro** — same paper-side action as above.

### Final paper sub-table

```
Benchmark        n     baseline   +mit (L26, K8, α=1.0)   Δ        95% CI
─────────────────────────────────────────────────────────────────────────
MMBench-EN dev   4329    XX.X        XX.X                 ±X.X     [..]
OCRBench (%)     1000    XX.X        XX.X                 ±X.X     [..]
RealWorldQA      765     XX.X        XX.X                 ±X.X     [..]
MMStar           1500    XX.X        XX.X                 ±X.X     [..]
HallusionBench   1129    XX.X        XX.X                 ±X.X     [..]
─────────────────────────────────────────────────────────────────────────
Macro                                                      ΔX.X     [..]
Verdict                                                    STRICT_FREE_LUNCH | DEGRADED:...
```

### Edge cases

1. **VLMEvalKit parse failure** — VLMEvalKit's default LLM-judge fallback may trigger for free-form answers. Configured to **disabled** (`use_llm_judge=False`) for all 5 benchmarks; rely on exact-match scoring only. Both arms apply the same parser, so Δ remains fair.
2. **OCRBench sub-task disaggregation** — appendix only; main table reports total.
3. **GPU OOM** — VLMEvalKit batch size = 1 (default). Hook is batch-agnostic (`hidden[:, -1, :]` works across batch dim).
4. **Hook activation count = 0** — fail-fast: aggregator raises if `_mit_call_count == 0` for any benchmark in mit variant.

---

## §5 Paper placement, output policy, acceptance, schedule

### Paper placement

`docs/paper/sections/07_mechanism_mitigation.md` (local-only, gitignored), §7.4.5 end:

> ### Capability Preservation
>
> To verify the mitigation does not degrade general VLM capability beyond
> anchoring, we evaluate the chosen cell L=26, K=8, α=1.0 on five held-out
> benchmarks spanning broad MCQ (MMBench-EN dev), OCR (OCRBench),
> fine-grained perception (RealWorldQA), contamination-resistant general
> (MMStar), and hallucination (HallusionBench). LLaVA-OneVision-Data
> training composition (HF dataset card, snapshot 2026-05-08) confirms none
> of these five benchmarks appear in instruction tuning. We use VLMEvalKit
> greedy decoding identical to the baseline; the only difference is the
> forward hook on layer 26.
>
> [Table — 5 rows + macro]
>
> [Verdict prose — single paragraph, scaled to outcome]

`docs/insights/E8-capability-preservation-evidence.md` (NEW, committed) — full reproducer + raw evidence + per-question difference analysis.

### Output policy

- `outputs/capability_eval/` — gitignored (added to `.gitignore` if not already covered).
- `docs/insights/_data/capability_eval_per_benchmark.{csv,md}` — gitignored (existing `_data/*` convention per CLAUDE.md and memory `feedback_consolidation_writing`).
- `docs/insights/_data/capability_eval_partial.{csv,md}` — gitignored.
- `docs/insights/E8-capability-preservation-evidence.md` — committed.
- `docs/paper/sections/07_mechanism_mitigation.md` — local-only edits (already gitignored under existing convention).

### Acceptance criteria

`STRICT_FREE_LUNCH` declared only if all hold:

1. baseline + mit runs each complete 5/5 benchmarks without error and produce normal score JSONs.
2. Hook activation sanity: `_mit_call_count > 0` per benchmark in mit variant.
3. Per-benchmark Δ ≥ -1.0pp on all 5 benchmarks.
4. Macro Δ ≥ -0.5pp.
5. `tests/test_capability_eval.py` passes (hook parity test).
6. `aggregate_capability_eval.py` emits `capability_eval_per_benchmark.{csv,md}` and the verdict line; the paper sub-table is regenerated.

Failure of any criterion → verdict `DEGRADED:<reason>`; paper §7.4.5 framing downgraded to "favorable trade-off".

### Schedule + compute estimate

| Stage | Wall time | Cost |
|---|---|---|
| VLMEvalKit pin + LLaVAOneVisionMitigated impl + unit test | ~4-6h human | — |
| Sanity smoke (each bench × 100 questions × 2 variants) | ~1h GPU | — |
| Full interleaved run (5 benchmarks × 2 variants, 8723 q × 2) | ~12-14h H200 | — |
| Aggregator + paper §7.4.5 sub-table + insight doc | ~2-3h human | — |
| **Total** | **~13-15h GPU + 6-9h human** | **$0** (LLM-judge disabled) |

Single H200 GPU assumed, variants run sequentially within each benchmark (interleaved per §6).

### Dependencies & blockers

- **Subspace tensor existence**: verify `outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/subspace_*.pt` exists with shape `(n_layers, K_max ≥ 8, d_model)`; load `[26, :8, :]` slice cleanly. **Blocker for run_capability_eval.py step 1.**
- **VLMEvalKit OneVision wrapper attribute path**: the `assert isinstance(self.model.language_model.model.layers[26], Qwen2DecoderLayer)` is checked at construction time. **Blocker for impl validation.**
- Independent of the parked InternVL3 panel-drop decision — this work is OneVision-Main only.

### Roadmap entry

Add to `references/roadmap.md` §7 Phase 4 as new **P0**:

> **P0 — E8 Mitigation capability-preservation regression test on OneVision Main** (new 2026-05-08). VLMEvalKit-based eval of L=26 K=8 α=1.0 on 5 held-out benchmarks (MMBench-DEV-EN + OCRBench + RealWorldQA + MMStar + HallusionBench, n=8723, no LLM-judge). Strict free-lunch verdict required to retain §7.4.5 strict claim. Estimate ~14-16h GPU + 6-9h human.

§10 changelog gets a 2026-05-08 entry once spec is written.

---

## §6 Operational requirements (intermediate progress reporting)

User requirement: each benchmark's intermediate result reported as soon as it completes. Implementation:

### Per-benchmark interleaving

Driver iterates by ascending question count: `RealWorldQA (765) → OCRBench (1000) → HallusionBench (1129) → MMStar (1500) → MMBench-DEV-EN (4329)`. Smaller benchmarks first so the first Δ surfaces within ~2h rather than after the full ~13h.

```python
for bench in fast_first_order:
    run_one(variant="baseline", bench)          # ~30min – 1.5h depending on n
    run_one(variant="mit",      bench)          # ~30min – 1.5h
    delta_row = aggregate_partial.update(bench) # writes _partial.csv + appends progress.log
    log(progress.log, fmt(delta_row))           # one-line summary
finalize_summary()                              # rename _partial → final + verdict line
```

### Live artifacts (user-pollable)

- `outputs/capability_eval/<run_id>/progress.log` — one line per completed benchmark, e.g.:
  `[2026-05-08 14:32:11] OCRBench n=1000 baseline=58.3 mit=57.9 Δ=-0.4 ±0.7 [PASS]`
- `docs/insights/_data/capability_eval_partial.csv` — full table updated after each benchmark; user can `cat` or `column -t -s,` for live status.
- `docs/insights/_data/capability_eval_partial.md` — same data, human-readable.

### Final transition

After last benchmark: `_partial.{csv,md}` is renamed to `capability_eval_per_benchmark.{csv,md}`; verdict line appended at the bottom; aggregator exits 0 (success) or 1 (verdict `DEGRADED:*`).

### Failure modes

- A benchmark errors mid-run → `progress.log` records the error; `_partial.csv` keeps prior rows; driver continues to next benchmark; final verdict marks the bad row as `INCOMPLETE`.
- baseline succeeds but mit fails (or vice versa) for a single benchmark → that row contributes `INCOMPLETE`; macro Δ is computed over completed rows only with that fact noted.

---

## Deferred

- **MMMU-DEV-VAL** (LLM-judge benchmark, 900 q, multi-discipline reasoning). Re-run after paper §7.4.5 prose is locked, when ~$1-2 LLM-judge cost is justified by reviewer-pre-check value. Tracked as a separate todo.

---

## Risks

1. **VLMEvalKit attribute path drift**. If the pinned commit later proves incompatible with our hook installation, we revert by re-pinning. Sentinel: the assert at construction time.
2. **Hidden contamination on MMBench dev**. Theoretical; mitigated by including MMStar (explicitly contamination-resistant by design). If MMBench-DEV-EN shows large Δ but MMStar shows small Δ, that's a per-benchmark divergence to be investigated rather than concealed.
3. **Mit variant unexpectedly *helps* on some benchmark.** Possible (subspace projection might suppress confidence-overshoot artefacts). We report the Δ honestly; "+Δ" rows are marked but do not flip the verdict.
4. **Hook fires on a non-anchoring multi-image benchmark in unexpected way**. The hook gates on `hidden.shape[1] > 1` (prefill) and `hidden.dim() == 3`. For OneVision multi-image inputs the prefill is still a single sequence, last-token semantics still hold. No special-case logic needed.

---

## Approval status

- §1 Goal & contribution — approved 2026-05-07
- §2 Approach (chosen A) — approved 2026-05-07
- §3 Architecture — approved 2026-05-07
- §4 Scoring & reporting — approved 2026-05-08 (with MMMU deferred + AI2D/ScienceQA dropped + MMStar/HallusionBench added)
- §5 Paper placement, output policy, acceptance, schedule — approved 2026-05-08
- §6 Operational (intermediate reporting) — approved 2026-05-08
- **Written spec — awaiting user review.**
