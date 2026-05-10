# Remove InternVL3 from paper, insights, scripts, and references

**Status**: design approved 2026-05-10; ready for implementation plan
**Worktree**: `worktree-paper+remove-internvl3` (`.claude/worktrees/paper+remove-internvl3`)
**Branch**: `worktree-paper+remove-internvl3`

## Goal

Remove InternVL3-8b from the active paper architecture (paper drafts,
insight docs, references, scripts, tests, and `_data/` CSVs) while
preserving raw `outputs/` artifacts and configs for audit. Re-anchor the
H6 (anchor-pull vs multi-image distraction orthogonal axes) cluster
claim on the 6-model main panel that already exists in
`phase1_p0_v3_7model_5dataset_summary.csv`, where InternVL3 rows are
NaN and the gemma vs qwen separation is empirically clean.

## Decisions (frozen by user 2026-05-10)

| Axis | Decision | Rationale |
|---|---|---|
| Scope | Paper + insight + scripts + references + `_data/*` CSVs + InternVL3-only code paths | User: "internvl 만을 위한 코드는 삭제" |
| `outputs/` | **Preserve** `outputs/<exp>/internvl3-8b/` directories untouched | Audit trail; no active script reads from them |
| `configs/*.yaml` | Keep yaml files in git, **remove `internvl3-8b` model entry** from each yaml's `models:` list | User intent is to preserve historical artifacts, not invite accidental future runs |
| H6 cluster | **Retain**, narrative re-written on 6-model panel | InternVL3 was the only "pure distraction" corner in the original E2 pilot; main matrix shows gemma family (anchoring corner) vs qwen + onevision (distraction corner) gives a cleaner cluster |
| InternVL3 NaN in canonical CSV | Leave as is, do not reinvestigate | Already effectively excluded from main matrix |
| Untracked InternVL3 scripts on master | Delete (`scripts/_compute_internvl3_6bin_wrongbase.py`, `scripts/_internvl3_wrongbase_6bin.py`) | Part of cleanup |
| Paper drafts (gitignored) | Edit in original working tree at absolute paths, **outside** worktree PR | `docs/paper/*` is gitignored; PR-only workflow can't carry these files |
| Worktree commit/PR workflow | Branch + commits + `gh pr create`; user merges | Memory: paper-completion phase = PR-only, no direct master push |

## Phase 1 result (already computed)

5-dataset main matrix (PlotQA, InfoVQA, ChartQA, TallyQA, MathVista),
6-model panel after removing InternVL3 (NaN). Per-model means:

| Model | adopt(a) | acc_drop_d_vs_b | Cluster |
|---|---|---|---|
| gemma3-4b | 0.125 | −0.005 | **anchoring corner** |
| gemma3-27b | 0.081 | −0.003 | **anchoring corner** |
| llava-interleave-7b | 0.052 | 0.014 | mixed |
| llava-onevision-7b (Main) | 0.043 | 0.026 | **distraction corner** |
| qwen2.5-vl-32b | 0.030 | 0.015 | distraction-leaning |
| qwen2.5-vl-7b | 0.014 | 0.010 | low both |

**Key finding**: gemma family is anti-correlated with qwen + onevision
on the (anchor-pull, multi-image-distraction) plane. H6 orthogonal-axes
claim survives without InternVL3, with a cleaner cross-family
separation than the original E2 pilot framing.

**Sub-finding (PlotQA-specific, per user emphasis)**: chart family
(PlotQA / InfoVQA / ChartQA) shows weak distraction signal; the strong
distraction signal lives on TallyQA + MathVista. New §5 framing
opportunity.

## Architecture: data flow

```
H6 cluster figure
  ← phase1_p0_v3_7model_5dataset_summary.csv (filter NaN; rename to 6model)
     ← outputs/experiment_e7_*, outputs/experiment_e5e_* (raw, kept)

Paper §5 cluster prose
  ← H6 figure + per-dataset table
  ← references/roadmap.md §3.3, H6 row revised

Insight docs (~18)
  ← _data CSVs after InternVL3 row removal + builder script regen

scripts/ model lists
  → no InternVL3 in any model whitelist
  → InternVL3-only scripts deleted

tests/test_models.py
  → InternVL3 fixtures/asserts removed; pytest green
```

## Components to change

### A. `_data/` CSVs (gitignored, local-only)

Regenerate from raw `outputs/` via builder scripts (after model
whitelist update). For builder-less CSVs, drop InternVL3 rows in place
and re-render any `*.md` table that depends on them.

- `phase1_p0_v3_7model_5dataset_summary.csv` → rename to
  `phase1_p0_v3_6model_5dataset_summary.csv`; drop NaN row
- `main_panel_5dataset_per_cell.csv`, `main_panel_5dataset_summary.md` regen
- `experiment_e7_*_per_cell.csv`, `experiment_e5e_*_per_cell.csv` regen
- `cross_dataset_peaks.csv` regen (mechanism panel)
- `L1_proxy_monotonicity*.csv`, `L1_confidence_*.csv` regen
- `section41_swap_analysis.csv` audit + regen if InternVL3 row present
- `C_form_migration_*.csv` **leave as is** (frozen audit trail)
- New: `H6_2axis_per_model.csv` + `H6_2axis_per_dataset.csv`

### B. Builder scripts (model whitelist update)

- `scripts/build_e5e_e7_5dataset_summary.py` — drop `internvl3-8b`
- `scripts/build_e5b_5strat_decay_summary.py` — N/A (OneVision-only)
- `scripts/build_paper_figures.py`
- `scripts/_build_figure_4_1_plotqa.py`
- `scripts/_analyze_section41_swap.py`
- `scripts/analyze_cross_dataset_peaks.py`
- `scripts/analyze_attention_per_layer.py`
- `scripts/extract_attention_mass.py`
- `scripts/causal_anchor_ablation.py`
- `scripts/analyze_causal_ablation.py`
- `scripts/analyze_e4_mitigation.py`
- `scripts/e4_attention_reweighting.py`
- `scripts/_b3_max16_6bin.py` — audit
- `scripts/run_experiment.py` — audit

### C. InternVL3-only code (delete)

- `scripts/_compute_internvl3_6bin_wrongbase.py` (master untracked)
- `scripts/_internvl3_wrongbase_6bin.py` (master untracked)
- `src/vlm_anchor/models.py` — check for InternVL3-specific branches
- `tests/test_models.py` — remove InternVL3 fixtures
- Any other module-level InternVL3 special-case (grep verified)

### D. Insight docs (git-tracked, ~18)

Update line/row removals + narrative re-writes:
- `E7-plotqa-infovqa-evidence.md` — 7-model → 6-model wrong-base df
  ranking; "InternVL3 H2 collapse" sentence dropped or reframed
- `E4-mitigation-evidence.md` — "InternVL3 = H6 distraction-not-
  anchoring" replaced with cleaner main-matrix cluster
- `headline-numbers.md` — E4 mitigation panel update; H6 paragraph rewrite
- `paper_summary_slides.md` — H6 slide update
- `00-summary.md`, `phase1-p0-v3-summary.md`
- `E1-patch-evidence.md`, `E1b-per-layer-localisation.md`,
  `E1d-causal-evidence.md`, `E5b-anchor-distance-evidence.md`,
  `E5e-mathvista-evidence.md`, `L1-confidence-modulation-evidence.md`,
  `paper-section-3-problem-definition.md`,
  `paper-section-7-4-mitigation-free-lunch.md`,
  `paper-section-8-f1-future-work.md`, `_resume_2026-05-05.md`
- `C-form-migration-report.md` — leave (audit trail)

### E. References (git-tracked)

- `references/roadmap.md` — H6 row, §3.1/§3.2/§3.3 model-list rows,
  §10 changelog entry 2026-05-10
- `references/roadmap_ko.md` — same
- `references/project.md` — §0.4 mechanism panel: 5-model + InternVL3
  appendix → 5-model only
- `references/plan_phase1.md` — 6-model matrix → 5-model

### F. Tests

- `tests/test_models.py` — remove InternVL3 imports/fixtures
- `uv run python -m pytest` green

### G. Paper drafts (gitignored, edited outside PR)

Original working tree absolute paths:
- `/mnt/.../docs/paper/emnlp_draft_ko.md`
- `/mnt/.../docs/paper/sections/01_intro.md`
- `/mnt/.../docs/paper/sections/04_datasets.md`
- `/mnt/.../docs/paper/sections/05_distance_digitpixel.md`
- `/mnt/.../docs/paper/sections/06_confidence.md`
- `/mnt/.../docs/paper/sections/07_mechanism_mitigation.md`

Changes:
- Update panel cardinality wherever InternVL3 was counted: main
  panel was 7-model in canonical CSV (one row NaN) → present as
  6-model main panel post-removal; mechanism panel was already
  5-model perfect-square + InternVL3 appendix → drop appendix
  mention
- H6 narrative re-write on 6-model cluster (gemma vs qwen + onevision)
- Abstract / §1.3 / §5.2 panel cardinality counts updated
- Remove all explicit "InternVL3-8b" mentions and replace numeric
  references with re-aggregated values

## Implementation order

```
P1 ── data
   ├─ B1 builder scripts model whitelist
   ├─ A regen CSVs + new H6 CSV
   └─ D insight docs (depends on A)

P2 ── code cleanup
   ├─ C delete InternVL3-only code
   └─ B2 audit other scripts for InternVL3 references

P3 ── references
   └─ E roadmap / project / plan_phase1

P4 ── verification
   ├─ F pytest green
   └─ commit by logical unit (data, scripts, insights, references, tests)

P5 ── PR
   └─ gh pr create + user merges

P6 ── paper drafts (separate, gitignored, original tree)
   └─ G edits + recovery anchor in roadmap §10
```

## Risks

| Risk | Trigger | Mitigation |
|---|---|---|
| Builder script doesn't regenerate cleanly | Phase 1 Phase A | Manual pandas drop + write fallback |
| Cross-cutting scripts break (e.g., `analyze_causal_ablation.py`) | Phase 2 Phase C | Run each script smoke-only after edit |
| Insight narrative drift between docs | Phase 1 Phase D | Headline numbers come from regen CSVs, not memory; `paper-section-7-4-mitigation-free-lunch.md` audit trail is canonical |
| Tests break in non-InternVL3 fixtures | Phase 4 | Run `pytest` after every commit; revert if regression |
| Paper draft edits introduce inconsistency with PR'd insights | Phase 6 | Edit drafts only after PR merged; cross-link via roadmap §10 |

## Acceptance

- `git grep -i internvl3` in worktree returns 0 hits in
  `scripts/`, `src/`, `tests/`, `references/*.md` (live sections),
  `docs/insights/*.md` (current evidence). Explicit allowlist for
  historical/audit content:
  - `docs/insights/C-form-migration-report.md` (frozen audit trail)
  - `docs/insights/_resume_*.md` (point-in-time session snapshots)
  - `references/roadmap.md` §10 changelog (historical entries
    untouched; new 2026-05-10 entry documents the removal)
- `git grep -i internvl3` in original working tree's `docs/paper/`
  returns 0 hits
- `uv run python -m pytest` green
- Phase 1 H6 figure + cluster narrative landed in `docs/insights/`
- PR open with logical-unit commits

## Out of scope

- Re-running any inference (InternVL3 or otherwise)
- Touching `outputs/` raw artifacts (only stop reading from them)
- Touching `configs/*.yaml` (preserved per user; if a config has an
  InternVL3 entry, leave the entry but it will not be referenced by
  any active script)
- Re-investigating why InternVL3 summary.json is NaN
- Anything in `_local/`
