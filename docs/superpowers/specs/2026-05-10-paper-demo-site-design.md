# Paper Demo Site — Design

**Date:** 2026-05-10
**Branch / worktree:** `worktree-paper+demo-site` at
`.claude/worktrees/paper+demo-site/`
**Goal:** Ship a static GitHub Pages companion site for the EMNLP submission
("Cross-modal numerical anchoring in VLMs"). Site combines a paper
landing page with one interactive teaser that lets readers feel the
phenomenon (`b/a/m/d` condition flipper across the 5-model main panel).

## 1. Scope and non-goals

**In scope**
- One-page scrollable landing site (Hero → TL;DR → Phenomenon → Demo →
  Headline results → Mitigation → Citation → Footer).
- Interactive condition flipper backed by pre-computed predictions for
  6 cherry-picked samples × 5 models × 4 conditions.
- Static asset pipeline: a build script extracts predictions and copies
  the four image variants per sample into `site/`.
- GitHub Actions workflow that mirrors `site/` to `gh-pages` on every
  master push.

**Out of scope**
- Live model inference. Predictions are frozen at build time.
- Dashboards / cross-dataset explorers / per-layer probes. Reserved for
  potential follow-up; not part of v1.
- Heavy SPA framework. Vanilla HTML + Tailwind CDN + plain JS only.
- Authentication, analytics beyond what GitHub provides by default.

## 2. Site map

Single-page scroll, eight sections in order:

| # | Section | Contents |
|---|---|---|
| 1 | Hero | Title, authors (placeholder until ready), affiliation, link buttons (Paper PDF, arXiv, Code, Data, BibTeX). |
| 2 | TL;DR | One-paragraph plain-language summary plus one teaser figure from `docs/figures/`. |
| 3 | The phenomenon | Worked example showing all four conditions side-by-side, 2–3 sentences explaining the `(a − d)` and `(a − m)` contrasts. |
| 4 | Interactive demo | The condition flipper. See §3. |
| 5 | Headline results | Three result cards (numbers + 1-line caption) plus two figures from `docs/figures/`. |
| 6 | Mitigation | One paragraph on the §7.4.5 cell (L=26, K=8, α=1.0) plus a before/after bar figure. |
| 7 | Citation | BibTeX block with copy-to-clipboard button. |
| 8 | Footer | License (paper repo's), contact (placeholder). |

## 3. Interactive demo (condition flipper)

### 3.1 UI

```
┌─────────────────────────────────────────────────────────────────┐
│  Sample picker (6 thumbnails, dataset label below each)          │
│  ◉ S1 [VQAv2]   ○ S2 [ChartQA]   ○ S3 [PlotQA]   ...             │
├─────────────────────────────────────────────────────────────────┤
│  Question + GT + anchor value                                    │
│  Target image       │   Second image (varies with condition)     │
├─────────────────────────────────────────────────────────────────┤
│  Condition: [ b ] [ a ] [ m ] [ d ]    ← toggle row              │
├─────────────────────────────────────────────────────────────────┤
│  Predictions table (5 rows × 4 columns; current condition cell  │
│  highlighted; ⚓ marker on `pred == anchor`; bold on `pred == gt`)│
├─────────────────────────────────────────────────────────────────┤
│  Caption (auto-generated): "On S1, 3/5 models pulled to anchor"  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Behavior

- Picking a sample swaps the question, GT, anchor value, target image,
  and the four condition images, and re-renders the prediction table.
- Toggling condition swaps only the second image and highlights the
  matching column.
- Cells where `pred == anchor` get an ⚓ glyph; cells where `pred == gt`
  render bold. Both can be true (rare but allowed).
- All data flows from one fetch of `site/data/demo.json` at page load.
  No further network calls after first paint.

### 3.3 Cherry-pick rules

The build script `scripts/build_demo_data.py` walks `outputs/<exp>/<model>/`
and selects 6 samples. A sample is **eligible** only if all 5 main-panel
models have predictions for all 4 conditions (`b/a/m/d`) on that sample.
The (model × dataset) coverage matrix is partial in practice, so eligible
samples concentrate in datasets with full panel coverage; criterion 4
adapts accordingly rather than promising a fixed dataset spread.

Among eligible samples, prefer those satisfying:

1. b-arm prediction matches GT (the model's baseline is correct).
2. ≥ 3 of the 5 main-panel models adopt anchor in a-arm
   (`pred_a == anchor`).
3. Same sample's m-arm largely returns to GT (digit-pixel control fires).
4. **Spread across multiple datasets** within the eligible pool — pull
   from as many of {VQAv2, ChartQA, PlotQA, MathVista, InfographicsVQA}
   as the panel-coverage matrix allows.

If criteria 1–3 yield fewer than 6 eligible samples, the script reports
the full candidate ranking and exits non-zero rather than relaxing
criteria silently. The spec author then either picks manually or
revisits the panel composition.

### 3.4 Main panel (5 models)

5-model subset of the paper's canonical 6-model main panel
(`references/project.md §0.4`, finalized 2026-05-04). Selection
prioritizes Main + size-and-family diversity; `qwen2.5-vl-32b-instruct`
is dropped because the 27B Gemma already covers the large-model band.

| Display label | HF id |
|---|---|
| LLaVA-OneVision 7B (main) | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Gemma-3 27B | `google/gemma-3-27b-it` |
| InternVL3 8B | `OpenGVLab/InternVL3-8B` |
| Gemma-3 4B | `google/gemma-3-4b-it` |

Per §3.3, only samples with full b/a/m/d predictions for all five models
are eligible; the build script fails loudly if zero such samples exist
in the supplied `outputs/` tree.

Note: `llava-next-interleaved-7b` was dropped from the canonical main
panel on 2026-05-04 (commit `0e7998e`) because its native resolution
is insufficient for chart/figure datasets, and is not included here.
`qwen3-vl-8b` is the §5 reasoning ablation, not a §3 main-panel
model, and is also excluded.

## 4. Data layer

### 4.1 Source-of-truth path

Input directories the build script reads from:
- `outputs/<experiment>/<model>/<timestamp>/predictions.csv` — per-sample
  predictions. Latest non-smoke run per `(model, dataset)` (apply the
  "largest run, not alphabetically-latest" rule from
  `feedback_smoke_run_pollution`).
- `inputs/<dataset>/images/...` — target images.
- `inputs/irrelevant_number/{N}.png` — anchor images.
- `inputs/irrelevant_number_masked/{N}.png` — masked anchor images.
- `inputs/irrelevant_neutral/{N}.png` — neutral distractor images.

The script must explicitly avoid `outputs/before_C_form/` (frozen,
pre-refactor backup).

### 4.2 Output shape — `site/data/demo.json`

```json
{
  "models": [
    {"id": "llava-interleave-7b", "label": "LLaVA-Interleave 7B (main)"},
    {"id": "llava-onevision-7b",  "label": "LLaVA-OneVision 7B"},
    {"id": "qwen2.5-vl-7b",       "label": "Qwen2.5-VL 7B"},
    {"id": "qwen3-vl-8b",         "label": "Qwen3-VL 8B"},
    {"id": "gemma-3-27b",         "label": "Gemma-3 27B"}
  ],
  "samples": [
    {
      "id": "S1",
      "dataset": "VQAv2",
      "question": "How many photos can you see?",
      "gt": 4,
      "anchor": 5,
      "images": {
        "target":  "assets/img/S1/target.jpg",
        "anchor":  "assets/img/S1/anchor.jpg",
        "masked":  "assets/img/S1/masked.jpg",
        "neutral": "assets/img/S1/neutral.jpg"
      },
      "predictions": {
        "llava-interleave-7b": {"b": 4, "a": 5, "m": 4, "d": 4},
        "llava-onevision-7b":  {"b": 4, "a": 5, "m": 4, "d": 4},
        "qwen2.5-vl-7b":       {"b": 4, "a": 4, "m": 4, "d": 4},
        "qwen3-vl-8b":         {"b": 4, "a": 4, "m": 4, "d": 4},
        "gemma-3-27b":         {"b": 4, "a": 5, "m": 4, "d": 4}
      }
    }
    // ... 5 more samples
  ]
}
```

### 4.3 Sizing

6 samples × 4 images × ~250 KB ≈ 6 MB images + ~5 KB JSON. Well within
GitHub Pages limits and a single page-load budget.

## 5. File layout

```
site/
├── index.html                    # single page, all sections
├── styles.css                    # custom CSS atop Tailwind CDN
├── main.js                       # data fetch, demo widget, copy-to-clipboard
├── data/
│   └── demo.json                 # built artifact, committed
└── assets/
    ├── img/<sample-id>/{target,anchor,masked,neutral}.jpg
    ├── figures/                  # copied from docs/figures/
    │   ├── teaser.png
    │   ├── headline_1.png
    │   ├── headline_2.png
    │   └── mitigation_bars.png
    └── favicon.svg

scripts/
└── build_demo_data.py            # walks outputs/, writes site/data + site/assets

.github/workflows/
└── deploy-pages.yml              # master push → site/ → gh-pages branch
```

`site/` is committed in full. The build script is the only thing that
edits `site/data/` and `site/assets/img/`; figures and source files are
edited manually.

`.gitignore` is left untouched — `inputs/` and `outputs/` stay ignored,
but `site/` lives outside both, so committed assets are tracked
naturally.

## 6. Build and deploy

### 6.1 Local build (one-shot, manual)

```bash
uv run python scripts/build_demo_data.py \
  --output-root outputs \
  --inputs-root inputs \
  --site-root site \
  --datasets vqav2,chartqa,plotqa,mathvista,infographicvqa \
  --models llava-interleave-7b,llava-onevision-7b,qwen2.5-vl-7b,qwen3-vl-8b,gemma-3-27b \
  --num-samples 6
```

Re-run only when:
- predictions change (re-pick samples), or
- a new model joins the main panel.

The script is idempotent: it overwrites `site/data/demo.json` and the
`site/assets/img/` tree it controls.

### 6.2 CI deploy

`.github/workflows/deploy-pages.yml`:

```yaml
name: Deploy demo site
on:
  push:
    branches: [master]
    paths: ["site/**"]
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
          publish_branch: gh-pages
```

GitHub Pages settings: source = `gh-pages` branch / root.

The CI does **not** run `build_demo_data.py`; predictions live outside
the repo (`outputs/` is gitignored), so CI cannot rebuild them. Local
build → commit → push is the flow.

## 7. Visual style

- Tailwind via CDN (`https://cdn.tailwindcss.com`). No build step.
- Typography: Inter (Google Fonts) for body, JetBrains Mono for code /
  predictions.
- Color palette: neutral grays for chrome; one accent color (deep blue)
  for links and highlighted condition column; warm amber for ⚓ markers.
- Mobile: stacks the demo's two-image row vertically below ~720 px.
- Dark mode: not in v1.

## 8. Open items (resolved at implementation time)

| # | Item | Owner | Resolution path |
|---|---|---|---|
| O1 | Author names + affiliations | user | Placeholder strings in `index.html`; user replaces before public launch. |
| O2 | arXiv URL, paper PDF URL | user | Placeholder `#` link; replaced post-arXiv submission. |
| O3 | Headline result card numbers (3 cards) | implementer | Pull from `references/roadmap.md` §3.3 and verify against `docs/insights/_data/*.csv`. Per `feedback_paper_table_audit`. |
| O4 | Which 2 figures for §5 headline | implementer | Pick from `docs/figures/`; favor `paper_M2_variant_comparison.png` and `paper_E5e_*.png`. |
| O5 | Which figure for §6 mitigation | implementer | Use the existing E6 / mitigation chart already used in `docs/paper/sections/07_mechanism_mitigation.md`. |
| O6 | BibTeX entry | implementer | Generate placeholder; user finalises post-submission. |

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cherry-pick criteria 1–3 yield < 6 eligible samples | Build script reports the candidate ranking and exits non-zero rather than silently relaxing rules; spec author resolves manually. |
| (model × dataset) panel coverage too thin (e.g. only VQAv2 has all 5 models) | §3.3 / §3.4 explicitly tolerate single-dataset outcomes. The script reports the coverage matrix so the user can decide whether to broaden it (run more inference) or accept the narrower demo. |
| Build script reads from `outputs/before_C_form/` accidentally | Hard-coded skip on that path; assertion fails the script if pointed there. |
| Image total size balloons (e.g. someone uses 1024² source images) | Build script downsizes to max 768 px on the long edge during copy. |
| Numbers in headline cards drift from canonical CSVs | Implementer follows `feedback_paper_table_audit`: every numeric claim cross-checked against `docs/insights/_data/*.csv` before commit. |
| GitHub Actions deploy fails on first run (permissions) | Use `peaceiris/actions-gh-pages@v4` (battle-tested); enable Pages source = gh-pages in settings as a manual one-time step. |

## 10. Acceptance criteria

The site is ready to merge when:

1. Local preview (`python -m http.server -d site`) renders all 8
   sections without console errors.
2. Sample picker, condition toggle, and prediction highlighting all
   work for every one of the 6 samples on a fresh page load.
3. `site/data/demo.json` validates against the schema in §4.2.
4. Every numeric claim in §5 (Headline results) and §6 (Mitigation)
   traces to a row in `docs/insights/_data/*.csv` or to a
   `summary.json` under the live `outputs/` tree (not
   `outputs/before_C_form/`).
5. CI workflow has run successfully at least once and the gh-pages
   branch contains the latest `site/` contents.
6. All `O1`–`O6` items are either filled in or carry a tracked
   placeholder with a `TODO(name): ...` marker for the user.
