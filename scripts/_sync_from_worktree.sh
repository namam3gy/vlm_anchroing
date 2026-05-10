#!/usr/bin/env bash
# Sync gitignored research artefacts from a worktree branch back to the main
# checkout.
#
# Why this exists: docs/insights/_data/, outputs/, and (parts of) docs/paper/
# are gitignored. When work happens in a git worktree under
# .claude/worktrees/<branch>/ and a PR is merged into master, only the
# tracked files (prose, scripts, tracked figures) come along — the gitignored
# artefacts (CSVs, NPZ raw draws, generated PDFs / paper local-only sections,
# raw output predictions) stay stranded inside the worktree.
#
# Run after merging a worktree branch's PR to copy the artefacts into the
# main checkout. Idempotent. Uses rsync --update so newer files in the main
# checkout are NOT overwritten.
#
# Usage:
#   bash scripts/_sync_from_worktree.sh <worktree-branch-name>
#   bash scripts/_sync_from_worktree.sh <worktree-branch-name> --dry-run
#
# Example:
#   bash scripts/_sync_from_worktree.sh phase5+p0-1-gamma-beta-bridge
#
# Default trees synced (all gitignored):
#   docs/insights/_data/     (per-experiment CSVs, NPZ raw draws, MD summaries)
#   outputs/                 (raw predictions / amplitude jsonls / per-run dirs)
#   docs/paper/              (KOREAN draft + sections; reviews/ is tracked)
#
# Add --include-figures to also pull docs/figures/ untracked PNGs (rare —
# tracked figures already merge via git).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <worktree-branch-name> [--dry-run] [--include-figures]" >&2
  exit 1
fi

BRANCH="$1"; shift
DRY_RUN=""
INCLUDE_FIGURES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run" ;;
    --include-figures) INCLUDE_FIGURES=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_ROOT="$REPO_ROOT/.claude/worktrees/$BRANCH"

if [[ ! -d "$WORKTREE_ROOT" ]]; then
  echo "error: worktree not found at $WORKTREE_ROOT" >&2
  echo "available worktrees:" >&2
  ls -1 "$REPO_ROOT/.claude/worktrees/" 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

cd "$REPO_ROOT"

# rsync flags:
#   -a   archive (recursive + perms + times)
#   -v   verbose (one line per copied file)
#   -h   human-readable sizes
#   --update  skip files where destination is newer (do-no-harm semantics)
#   --info=stats2  print summary at the end
#
# --exclude'd subtrees (already git-tracked, no need to rsync — git handles them):
#   docs/paper/reviews/   (un-ignored via .gitignore rule !docs/paper/reviews/*.md)
RSYNC_FLAGS=(-avh --update --info=stats2 --exclude='reviews/' $DRY_RUN)

declare -a TREES=(
  "docs/insights/_data/"
  "outputs/"
  "docs/paper/"
)
if [[ "$INCLUDE_FIGURES" == 1 ]]; then
  TREES+=("docs/figures/")
fi

for tree in "${TREES[@]}"; do
  src="$WORKTREE_ROOT/$tree"
  dst="$REPO_ROOT/$tree"
  if [[ ! -d "$src" ]]; then
    echo "[skip] $tree — not present in worktree"
    continue
  fi
  mkdir -p "$dst"
  echo "[sync] $tree"
  rsync "${RSYNC_FLAGS[@]}" "$src" "$dst"
done

echo
if [[ -n "$DRY_RUN" ]]; then
  echo "[done] dry-run only — no files were modified."
else
  echo "[done] sync complete from $BRANCH → main checkout."
  echo "       run 'git status' to see any newly-tracked files that may need staging."
fi
