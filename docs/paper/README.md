# Paper draft — Cross-modal numerical anchoring in VLMs

**Target:** EMNLP 2026 Main, ARR May 25 deadline.

## Layout

- `draft.md` — single-file integrated paper draft (markdown). LaTeX
  conversion happens at submission time; markdown is the working
  format until then.
- `sections/` — individual section drafts (intro, related, method, etc.).
  Each can be edited independently and concatenated into `draft.md`.

## Source-of-truth pointers

- Paper outline: `references/project.md §0.1 / §0.2`
- C-form metric definitions: `src/vlm_anchor/metrics.py` +
  `docs/insights/M2-metric-definition-evidence.md`
- Headline numbers: `references/roadmap.md §3.3` (post-C-form refresh)
- Migration audit: `docs/insights/C-form-migration-report.md`

Every numeric claim in the draft must trace back to one of:
- `outputs/<experiment>/<model>/<timestamp>/summary.json` (live tree),
- `docs/insights/_data/*.csv` (refreshed CSVs), or
- `docs/insights/<E_id>-evidence.md` (which themselves cite the above).

**Never quote numbers from `outputs/before_C_form/`** — it's a
pre-refactor backup; see its README.
