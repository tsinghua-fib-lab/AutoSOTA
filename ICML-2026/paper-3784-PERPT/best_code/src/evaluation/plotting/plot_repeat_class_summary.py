#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric(
    df,
    metric,
    ylabel,
    out_path,
    sort_by="gamba",
):
    """
    metric: "ce" or "corr"
    """
    g_col = f"gamba_{metric}_mean"
    c_col = f"caduceus_{metric}_mean"

    plot_df = df[["repClass", g_col, c_col]].copy()

    # sort for readability
    if sort_by == "gamba":
        plot_df = plot_df.sort_values(g_col)
    elif sort_by == "caduceus":
        plot_df = plot_df.sort_values(c_col)

    plot_df = plot_df.melt(
        id_vars="repClass",
        value_vars=[g_col, c_col],
        var_name="model",
        value_name="value",
    )

    plot_df["model"] = plot_df["model"].map({
        g_col: "gamba",
        c_col: "caduceus",
    })

    plt.figure(figsize=(10, max(4, 0.4 * plot_df["repClass"].nunique())))
    sns.barplot(
        data=plot_df,
        x="value",
        y="repClass",
        hue="model",
        orient="h",
        errorbar=None,
    )

    plt.xlabel(ylabel)
    plt.ylabel("repeat class")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # sanity filter: drop rows with no valid data
    df = df[
        df[["gamba_ce_mean", "caduceus_ce_mean"]].notna().any(axis=1)
    ]

    plot_metric(
        df,
        metric="ce",
        ylabel="mean CE loss",
        out_path=out_dir / "repeatclass_ce_loss.png",
        sort_by="gamba",
    )

    plot_metric(
        df,
        metric="corr",
        ylabel="phyloP Pearson r",
        out_path=out_dir / "repeatclass_phylop_corr.png",
        sort_by="gamba",
    )


if __name__ == "__main__":
    main()

# python /home/mica/gamba/src/evaluation/plotting/plot_repeat_class_summary.py \
#   --csv /home/mica/gamba/data_processing/data/240-mammalian/repeat_loss_eval/repeatclass_ce_corr_summary.csv \
#   --out_dir /home/mica/gamba/data_processing/data/240-mammalian/repeat_loss_eval/plots
