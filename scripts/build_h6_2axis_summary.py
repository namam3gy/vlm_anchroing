"""Build H6 (anchor-pull vs multi-image distraction) 2-axis cluster summary.

H6: cross-modal failures decouple into two orthogonal axes.
- anchor-pull = adopt(a) = #(pa==anchor AND pb!=anchor) / #(pb!=anchor)
- multi-image distraction = acc_drop_d_vs_b = exact_match(b) - exact_match(d)

Source: docs/insights/_data/phase1_p0_v3_7model_5dataset_summary.csv
Models: 6-model main panel (InternVL3 NaN dropped).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path("/mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing")
SRC = REPO / "docs/insights/_data/phase1_p0_v3_7model_5dataset_summary.csv"
OUT_DIR = REPO / "docs/insights/_data"
FIG = REPO / "docs/figures/H6_2axis_scatter_5dataset.png"

DATASET_SHORT = {
    "experiment_e7_plotqa_full": "PlotQA",
    "experiment_e7_infographicvqa_full": "InfoVQA",
    "experiment_e5e_chartqa_full": "ChartQA",
    "experiment_e5e_tallyqa_full": "TallyQA",
    "experiment_e5e_mathvista_full": "MathVista",
}
MODEL_SHORT = {
    "llava-onevision-qwen2-7b-ov": "onevision-7b (Main)",
    "llava-next-interleaved-7b": "interleave-7b",
    "qwen2.5-vl-7b-instruct": "qwen2.5-vl-7b",
    "qwen2.5-vl-32b-instruct": "qwen2.5-vl-32b",
    "gemma3-4b-it": "gemma3-4b",
    "gemma3-27b-it": "gemma3-27b",
    "gemma3-12b-it": "gemma3-12b",
}

def main() -> None:
    df = pd.read_csv(SRC)
    df = df.dropna(subset=["acc_b", "acc_d", "adopt_a"]).copy()
    df = df[df["model"] != "internvl3-8b"].copy()
    assert df["model"].nunique() == 6, df["model"].unique()
    df["acc_drop_d_vs_b"] = df["acc_b"] - df["acc_d"]
    df["dataset_short"] = df["dataset"].map(DATASET_SHORT)
    df["model_short"] = df["model"].map(MODEL_SHORT)

    per_dataset = df[["dataset_short", "model_short", "acc_b", "acc_d",
                       "acc_drop_d_vs_b", "adopt_a"]].sort_values(
        ["dataset_short", "model_short"]
    )
    per_dataset.to_csv(OUT_DIR / "H6_2axis_per_dataset.csv", index=False)

    per_model = df.groupby("model_short").agg(
        adopt_a_mean=("adopt_a", "mean"),
        acc_drop_mean=("acc_drop_d_vs_b", "mean"),
        n_datasets=("dataset_short", "nunique"),
    ).round(4).sort_values("adopt_a_mean", ascending=False)
    per_model.to_csv(OUT_DIR / "H6_2axis_per_model.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for model in per_model.index:
        x = per_model.loc[model, "adopt_a_mean"]
        y = per_model.loc[model, "acc_drop_mean"]
        ax.scatter(x, y, s=80)
        ax.annotate(model, (x, y), xytext=(6, 4),
                     textcoords="offset points", fontsize=9)
    ax.set_xlabel("adopt(a) — anchor pull (5-dataset mean)")
    ax.set_ylabel("acc_drop_d_vs_b — multi-image distraction")
    ax.set_title("H6: 2-axis decomposition (6-model panel, 5 datasets)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"Wrote {OUT_DIR / 'H6_2axis_per_model.csv'}")
    print(f"Wrote {OUT_DIR / 'H6_2axis_per_dataset.csv'}")
    print(f"Wrote {FIG}")

if __name__ == "__main__":
    main()
