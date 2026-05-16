# Overleaf bundle — EMNLP outline draft

Drop-in Overleaf project source for `docs/paper/emnlp_outline_ko.md`.
Korean + English mixed prose; compiles to ACL 2024 / EMNLP two-column layout
for page-count estimation.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Single-file LaTeX document (Korean prose preserved verbatim) |
| `acl.sty` | ACL 2024 style file (fetched from `acl-org/acl-style-files@master`) |
| `acl_natbib.bst` | ACL natbib bibliography style |
| `custom.bib` | Placeholder bib (real entries TBD) |
| `figures/` | All figures referenced in `main.tex` (10 PNGs, ~2 MB) |
| `_reference_acl_latex.tex` | Upstream `pdflatex` template (read-only, gitignored) |
| `_reference_acl_lualatex.tex` | Upstream XeLaTeX/LuaLaTeX template (read-only, gitignored) |

## Upload to Overleaf

Two equivalent paths.

### Option A — zip and upload as a new project
1. `cd docs/paper/overleaf && zip -r overleaf.zip main.tex acl.sty acl_natbib.bst custom.bib figures/`
2. Overleaf → **New Project** → **Upload Project** → drop `overleaf.zip`.
3. Open project → **Menu** (left sidebar) → **Compiler** → set to **XeLaTeX**.
4. **Recompile**.

### Option B — start from Overleaf's ACL template, then replace
1. Overleaf → **New Project** → **Templates** → search "ACL" → pick *Association for Computational Linguistics (ACL)*.
2. Delete the template's `acl_latex.tex` (or rename it out of the way).
3. Drag in `main.tex` from this directory.
4. Drag in the `figures/` folder (preserve directory structure).
5. **Menu** → **Compiler** → **XeLaTeX**. **Recompile**.

## Why XeLaTeX?

`main.tex` uses `\usepackage{xeCJK}` with **Noto Sans CJK KR** for the Korean
prose blocks (almost the entire body — the outline keeps Korean for narration
and English for technical terms). `pdflatex` cannot render Korean; LuaLaTeX
also works but compiles slower on Overleaf's free tier. XeLaTeX is the
recommended path.

The font (`Noto Sans CJK KR`) ships with the Overleaf TeX Live image; no font
upload is needed. If you ever see "Font not found", the closest fallbacks are
`NanumGothic` or `UnDotum`.

## Compile locally (optional)

Requires `xelatex` + the `kotex`/`xeCJK` package and Noto fonts:

```bash
cd docs/paper/overleaf
xelatex main.tex
bibtex  main           # only if you populate custom.bib
xelatex main.tex
xelatex main.tex
```

## What's in `main.tex`

A faithful conversion of `docs/paper/emnlp_outline_ko.md` — every section,
subsection, bullet, table, and figure-caption is preserved.

- Outline placeholders (the `{{...}}` blocks in the markdown) are rendered as
  grey-italic inline notes via the `\todoNote{...}` macro so they remain
  visible during page-count estimation. Replace them with real prose as the
  draft fills in.
- All 10 referenced figures are included locally — no external fetch at
  compile time.
- Tables converted to `booktabs` (`\toprule` / `\midrule` / `\bottomrule`).
- Metric definitions and notation (`p_i^c`, `gt_i`, `z_i`, `\epsilon`-threshold
  DF form) are typeset in math mode.

## What it does NOT include

- Real references in `custom.bib` (the outline only has placeholder citations
  like `[Tversky and Kahneman, 1974]`).
- Author block (set to "Anonymous Submission" for review mode).
- Final figures for §4.2 (digit-pixel causality 3-arm version) and §5.2
  (combined K-progression) — preview figures stand in; see `\todoNote{...}`
  markers in `main.tex`.

## Regenerate after outline edits

This bundle is a one-shot manual conversion of `emnlp_outline_ko.md` (as of
the parent commit). Edits to the outline will not propagate automatically —
edit `main.tex` directly, or re-run the conversion when the markdown changes
substantially.
