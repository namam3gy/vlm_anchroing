"""Aggregate the 27-cell E6 pilot grid into a 4-metric heatmap (closes R4 CRIT-2).

Each pilot run (`pilot_grid_<calib>_n250/predictions.jsonl`) holds, for one
calibration dataset, 28 cells × 250 wrong-base sids × 4 conditions. The 28
cells are: 1 baseline (cell_label = "baseline") + 27 grid cells
(L ∈ {25,26,27} × K ∈ {2,4,8} × α ∈ {0.5,1.0,2.0}) labelled `L<LL>_K<KK>_a<α>`.

For each (calibration_dataset, cell), we compute Δ-versus-baseline on four
metrics on **the same 250-sid wrong-base population**:

  Δ adopt(a) = adopt_cell − adopt_base
  Δ df(a)    = df_cell    − df_base   (C-form: pa!=pb AND (pa-pb)*(anchor-pb) > 0)
  Δ em(a)    = em(a)_cell − em(a)_base
  Δ em(b)    = em(b)_cell − em(b)_base

Outputs:
- `docs/insights/_data/E6_pilot_grid_27cells.csv`  (54 rows = 27 cells × 2 calib datasets)
- `docs/figures/E6_pilot_grid_<calib>_heatmap.png` × 2 (per-calibration-dataset 4×3 panel)

The 4×3 panel: rows = 4 metrics (Δadopt / Δdf / Δem(a) / Δem(b)),
cols = 3 layers (L=25, 26, 27). Each cell = K (rows) × α (cols) heatmap.
The chosen cell (L=26, K=8, α=1.0) is marked with a star.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava-onevision-qwen2-7b-ov"

PILOTS = [
    ("plotqa", "PlotQA", "pilot_grid_plotqa_n250"),
    ("infographicvqa", "InfoVQA", "pilot_grid_infographicvqa_n250"),
]

LAYERS = (25, 26, 27)
KS = (2, 4, 8)
ALPHAS = (0.5, 1.0, 2.0)
CHOSEN = (26, 8, 1.0)

METRIC_LABELS = {
    "adopt": "Δ adopt(a)",
    "df": "Δ df(a)",
    "em_a": "Δ em(a)",
    "em_b": "Δ em(b)",
}
METRICS = tuple(METRIC_LABELS.keys())
# +negative direction = mitigation effect; mark direction explicitly per metric.
DIR = {"adopt": -1, "df": -1, "em_a": +1, "em_b": +1}


def _try_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _try_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_pilot(jsonl_path: Path) -> dict[str, dict[str, dict[str, dict]]]:
    """Group records as cell_label → sid → condition → record."""
    by_cell: dict = defaultdict(lambda: defaultdict(dict))
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_cell[r["cell_label"]][r["sample_instance_id"]][r["condition"]] = r
    return by_cell


def _eligible_paired_sids(cell_data: dict, baseline_data: dict) -> list[str]:
    """Sids parseable on both b+a in BOTH the cell and the baseline arm."""
    out = []
    for sid in set(cell_data) & set(baseline_data):
        ok = True
        for arm in (cell_data, baseline_data):
            b = arm[sid].get("target_only")
            a = arm[sid].get("target_plus_irrelevant_number_S1")
            if not (b and a):
                ok = False
                break
            try:
                float(b["parsed_number"])
                float(a["parsed_number"])
                float(a["anchor_value"])
            except (KeyError, TypeError, ValueError):
                ok = False
                break
        if ok:
            out.append(sid)
    return out


def _metrics_on_sids(arm: dict, sids: list[str]) -> dict[str, float]:
    em_a_n = em_a_d = em_b_n = em_b_d = 0
    df_n = df_d = ad_n = ad_d = 0
    for sid in sids:
        b = arm[sid].get("target_only")
        a = arm[sid].get("target_plus_irrelevant_number_S1")
        if not (b and a):
            continue
        pb = _try_float(b["parsed_number"])
        pa = _try_float(a["parsed_number"])
        anchor = _try_float(a.get("anchor_value"))
        if pb is None or pa is None or anchor is None:
            continue
        gt_b = _try_int(b.get("ground_truth"))
        gt_a = _try_int(a.get("ground_truth"))
        if gt_b is not None:
            em_b_d += 1
            if int(pb) == gt_b:
                em_b_n += 1
        if gt_a is not None:
            em_a_d += 1
            if int(pa) == gt_a:
                em_a_n += 1
        if pb != anchor:
            ad_d += 1
            if pa == anchor:
                ad_n += 1
        df_d += 1
        if pa != pb and (pa - pb) * (anchor - pb) > 0:
            df_n += 1
    return {
        "n": len(sids),
        "adopt": ad_n / ad_d if ad_d else float("nan"),
        "df": df_n / df_d if df_d else float("nan"),
        "em_a": em_a_n / em_a_d if em_a_d else float("nan"),
        "em_b": em_b_n / em_b_d if em_b_d else float("nan"),
    }


def _cell_grid_to_array(rows: list[dict], calib: str, metric: str) -> np.ndarray:
    """Return [3 layers, 3 K, 3 alpha] array of Δmetric values for one calib dataset."""
    arr = np.full((len(LAYERS), len(KS), len(ALPHAS)), np.nan, dtype=np.float64)
    for r in rows:
        if r["calib"] != calib:
            continue
        try:
            li = LAYERS.index(int(r["layer"]))
            ki = KS.index(int(r["K"]))
            ai = ALPHAS.index(float(r["alpha"]))
        except ValueError:
            continue
        arr[li, ki, ai] = r[f"delta_{metric}"]
    return arr


def _plot_heatmap(rows: list[dict], calib_tag: str, calib_label: str, out_path: Path) -> None:
    """4 metrics × 3 layers = 12 small heatmaps (K × α)."""
    fig, axes = plt.subplots(
        len(METRICS), len(LAYERS), figsize=(11, 12), constrained_layout=True
    )
    fig.suptitle(
        f"E6 27-cell pilot grid — {calib_label} (n=250 wrong-base) — Δ vs baseline (pp)",
        fontsize=12,
    )
    for mi, metric in enumerate(METRICS):
        arr = _cell_grid_to_array(rows, calib_tag, metric) * 100.0  # to pp
        # Per-metric vmin/vmax symmetric around 0
        absmax = float(np.nanmax(np.abs(arr))) if not np.all(np.isnan(arr)) else 1.0
        absmax = max(absmax, 1.0)
        for li, layer in enumerate(LAYERS):
            ax = axes[mi, li]
            data = arr[li]  # [K, α]
            # color: + direction (positive Δem) = blue, - direction (negative Δdf) = also blue.
            # Use a single signed cmap; flip direction sign for - metrics so blue=better.
            data_for_color = data * DIR[metric]
            im = ax.imshow(
                data_for_color,
                cmap="RdBu",
                vmin=-absmax,
                vmax=+absmax,
                aspect="equal",
                origin="upper",
            )
            ax.set_xticks(range(len(ALPHAS)))
            ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=8)
            ax.set_yticks(range(len(KS)))
            ax.set_yticklabels([f"K={k}" for k in KS], fontsize=8)
            for ki, kv in enumerate(KS):
                for ai, av in enumerate(ALPHAS):
                    val = data[ki, ai]
                    if np.isnan(val):
                        continue
                    txt = f"{val:+.1f}"
                    color = (
                        "white"
                        if abs(data_for_color[ki, ai]) > 0.55 * absmax
                        else "black"
                    )
                    ax.text(ai, ki, txt, ha="center", va="center", fontsize=8, color=color)
                    if (layer, kv, av) == CHOSEN:
                        ax.add_patch(
                            plt.Rectangle(
                                (ai - 0.45, ki - 0.45),
                                0.9,
                                0.9,
                                fill=False,
                                edgecolor="black",
                                linewidth=2.0,
                            )
                        )
                        ax.text(
                            ai,
                            ki - 0.32,
                            "★",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="black",
                            fontweight="bold",
                        )
            if mi == 0:
                ax.set_title(f"L = {layer}", fontsize=10)
            if li == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=7)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "docs" / "insights" / "_data" / "E6_pilot_grid_27cells.csv"),
    )
    ap.add_argument(
        "--fig-dir", default=str(PROJECT_ROOT / "docs" / "figures")
    )
    ap.add_argument(
        "--e6-root", type=str, default=None,
        help="Override `outputs/e6_steering/` (where the pilot-sweep "
             "`<model>/pilot_grid_*/predictions.jsonl` files live). "
             "Reproducer notebooks point this at an isolated tree.",
    )
    ap.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices used for the heatmap rows/columns "
             "and the selection-replay table. Defaults to the canonical "
             "27-cell grid (25,26,27). Cells with layers outside this set "
             "still land in the CSV.",
    )
    ap.add_argument(
        "--ks", type=str, default=None,
        help="Comma-separated K (subspace rank) values. Defaults to 2,4,8.",
    )
    ap.add_argument(
        "--alphas", type=str, default=None,
        help="Comma-separated steering magnitudes. Defaults to 0.5,1.0,2.0.",
    )
    ap.add_argument(
        "--chosen", type=str, default=None,
        help="L,K,alpha for the chosen-cell star marker (default 26,8,1.0).",
    )
    args = ap.parse_args()

    global LAYERS, KS, ALPHAS, CHOSEN
    if args.layers:
        LAYERS = tuple(int(x) for x in args.layers.split(",") if x.strip())
    if args.ks:
        KS = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if args.alphas:
        ALPHAS = tuple(float(x) for x in args.alphas.split(",") if x.strip())
    if args.chosen:
        parts = [x.strip() for x in args.chosen.split(",")]
        CHOSEN = (int(parts[0]), int(parts[1]), float(parts[2]))
    out_csv = Path(args.out_csv)
    fig_dir = Path(args.fig_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    e6_root = Path(args.e6_root).resolve() if args.e6_root else (
        PROJECT_ROOT / "outputs" / "e6_steering"
    )

    rows: list[dict] = []
    for calib_tag, calib_label, dirname in PILOTS:
        pilot_dir = e6_root / MODEL / dirname
        pred = pilot_dir / "predictions.jsonl"
        if not pred.exists():
            print(f"[skip] {calib_label}: missing {pred}")
            continue
        by_cell = _load_pilot(pred)
        if "baseline" not in by_cell:
            print(f"[skip] {calib_label}: no baseline cell")
            continue
        baseline = by_cell["baseline"]

        for cell_label, cell_data in sorted(by_cell.items()):
            if cell_label == "baseline":
                continue
            # Parse cell label "L26_K08_a1.0"
            try:
                parts = cell_label.split("_")
                layer = int(parts[0][1:])
                K = int(parts[1][1:])
                alpha = float(parts[2][1:])
            except (IndexError, ValueError):
                print(f"[warn] {calib_label}: cannot parse cell label {cell_label!r}")
                continue
            sids = _eligible_paired_sids(cell_data, baseline)
            base_metrics = _metrics_on_sids(baseline, sids)
            cell_metrics = _metrics_on_sids(cell_data, sids)
            row = {
                "calib": calib_tag,
                "calib_label": calib_label,
                "cell_label": cell_label,
                "layer": layer,
                "K": K,
                "alpha": alpha,
                "is_chosen": (layer, K, alpha) == CHOSEN,
                "n_paired": base_metrics["n"],
            }
            for m in METRICS:
                row[f"baseline_{m}"] = base_metrics[m]
                row[f"cell_{m}"] = cell_metrics[m]
                row[f"delta_{m}"] = cell_metrics[m] - base_metrics[m]
            rows.append(row)

    if not rows:
        raise SystemExit("No pilot data found.")

    # Write canonical CSV
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {out_csv}  ({len(rows)} rows)")

    # Render heatmaps per calibration dataset
    for calib_tag, calib_label, _ in PILOTS:
        fig_path = fig_dir / f"E6_pilot_grid_{calib_tag}_heatmap.png"
        _plot_heatmap(rows, calib_tag, calib_label, fig_path)
        print(f"[write] {fig_path}")

    # ---- Selection-rule replay summary ----
    # Em-deal-breaker rule: reject cell if Δem(a) ≤ −6 pp on EITHER calib dataset.
    em_drop_threshold = 0.06
    by_cell_grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell_grouped[r["cell_label"]].append(r)

    print()
    print("=== Selection rule replay (em-deal-breaker −6 pp on either calib) ===")
    summary_lines = []
    summary_lines.append("| cell # | L | K | α | min Δem(a) over calib (pp) | rejected | mean Δdf(a) (pp) | mean Δem(b) (pp) | chosen |")
    summary_lines.append("|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|")
    cell_id = 0
    for L in LAYERS:
        for K in KS:
            for a in ALPHAS:
                cell_id += 1
                lab = f"L{L:02d}_K{K:02d}_a{a:.1f}"
                cell_rows = by_cell_grouped.get(lab, [])
                if not cell_rows:
                    summary_lines.append(
                        f"| {cell_id} | {L} | {K} | {a} | — | — | — | — | — |"
                    )
                    continue
                min_dem_a = min(r["delta_em_a"] for r in cell_rows) * 100
                mean_ddf = sum(r["delta_df"] for r in cell_rows) / len(cell_rows) * 100
                mean_demb = sum(r["delta_em_b"] for r in cell_rows) / len(cell_rows) * 100
                rejected = min_dem_a <= -em_drop_threshold * 100
                chosen = (L, K, a) == CHOSEN
                summary_lines.append(
                    f"| {cell_id} | {L} | {K} | {a} | {min_dem_a:+.1f} | {'**REJ**' if rejected else '✓'} | {mean_ddf:+.1f} | {mean_demb:+.1f} | {'**★**' if chosen else '—'} |"
                )

    sel_md_path = out_csv.with_name("E6_pilot_grid_27cells_selection_replay.md")
    sel_md_path.write_text(
        "# 27-cell pilot grid — selection-rule replay\n"
        "\n"
        "Auto-generated by `scripts/aggregate_e6_pilot_grid.py`.\n"
        "Selection rule (ex ante, see §6.2.2): reject any cell with "
        "**Δem(a) ≤ −6 pp on either calibration dataset**; rank surviving by "
        "combined |Δdf(a)|. Per-cell metrics here = mean across PlotQA + InfoVQA "
        "calibration pilots (each n=250 wrong-base).\n"
        "\n"
        + "\n".join(summary_lines)
        + "\n"
    )
    print(f"[write] {sel_md_path}")

    # Final verdict line for chosen cell
    chosen_rows = by_cell_grouped[f"L{CHOSEN[0]:02d}_K{CHOSEN[1]:02d}_a{CHOSEN[2]:.1f}"]
    if chosen_rows:
        for r in chosen_rows:
            print(
                f"  CHOSEN {r['calib_label']:<8} Δadopt={r['delta_adopt']*100:+.2f}pp "
                f"Δdf={r['delta_df']*100:+.2f}pp Δem(a)={r['delta_em_a']*100:+.2f}pp "
                f"Δem(b)={r['delta_em_b']*100:+.2f}pp"
            )


if __name__ == "__main__":
    main()
