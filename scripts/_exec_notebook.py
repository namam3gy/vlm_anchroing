"""Execute a Jupyter notebook in-process via direct `exec` of each code cell.

Workaround for a pandas 3.0.2 + ipykernel display-layer TypeError that
hits `nbconvert --execute` (the side-effects we need — CSV writes,
figure renders — happen fine when cells are exec'd directly without
the ipykernel display path).

Usage:
    .venv/bin/python scripts/_exec_notebook.py notebooks/paper_cross_model_cross_dataset.ipynb
"""
from __future__ import annotations
import argparse
import sys
import traceback
from pathlib import Path

import nbformat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going after a cell fails (default = stop).")
    args = ap.parse_args()

    nb = nbformat.read(args.path, as_version=4)
    print(f"[exec-notebook] {args.path}  ({len([c for c in nb.cells if c.cell_type == 'code'])} code cells)")
    ns: dict = {"__name__": "__main__"}
    failed = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            exec(compile(cell.source, f"<cell-{i}>", "exec"), ns)
        except SystemExit:
            raise
        except Exception as e:
            failed += 1
            print(f"\n[exec-notebook] cell {i}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(limit=5, file=sys.stderr)
            if not args.continue_on_error:
                sys.exit(2)
    if failed:
        print(f"[exec-notebook] {failed} cell(s) failed.", file=sys.stderr)
        sys.exit(1)
    print("[exec-notebook] all cells succeeded.")


if __name__ == "__main__":
    main()
