#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CORE_ORDER = ["vista_enhancer", "UCNE", "exons", "coding_regions", "upstream_TSS", "promoters"]
OTHER_ORDER = ["repeats", "introns", "UTR5", "UTR3"]

RENAME_MAP = {"upstream_TSS": "2kb upstream of TSS"}

PURPLE = "#7b1fa2"
BLUE   = "#1976d2"

def load_violin_npz(path: str, label: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame({
        "category": z["category"].astype(str),
        "chrom": z["chrom"].astype(str),
        "data_split": z["data_split"].astype(str),
        "loss": z["loss"].astype(np.float32),
    })
    df["model"] = label
    return df

def rename_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["category"].replace(RENAME_MAP)
    return df

def subset_order(df: pd.DataFrame, wanted: list[str]) -> tuple[pd.DataFrame, list[str]]:
    present = [c for c in wanted if c in set(df["category"])]
    return df[df["category"].isin(present)].copy(), present

def violin_plot(df: pd.DataFrame, order: list[str], title: str, out_path: str, ylim: tuple[float,float] | None):
    if df.empty or not order:
        print(f"[WARN] No data for plot: {title}")
        return
    plt.figure(figsize=(1.6*max(6, len(order)), 6))
    ax = sns.violinplot(
        data=df, x="category", y="loss",
        order=order, hue="model",
        scale="width", inner="quartile", cut=0, dodge=True,
        palette=[PURPLE, BLUE]
    )
    ax.set_xlabel("Feature Category")
    ax.set_ylabel("CE loss")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(title=None)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")

def main():
    p = argparse.ArgumentParser(description="Compare CE-loss violins for two models.")
    p.add_argument("--npz_a", required=True, help="Path to model A NPZ")
    p.add_argument("--npz_b", required=True, help="Path to model B NPZ")
    p.add_argument("--label_a", required=True, help="Label for model A")
    p.add_argument("--label_b", required=True, help="Label for model B")
    p.add_argument("--outdir", required=True, help="Output directory for plots")
    p.add_argument("--ylim", type=float, nargs=2, default=None, help="y-limits, e.g., 1.0 1.38")
    args = p.parse_args()

    df_a = load_violin_npz(args.npz_a, args.label_a)
    df_b = load_violin_npz(args.npz_b, args.label_b)

    df = pd.concat([df_a, df_b], ignore_index=True)
    df = rename_categories(df)

    # Plot 1: core categories (with renamed TSS label)
    df_core, order_core = subset_order(df, [RENAME_MAP.get(c, c) for c in CORE_ORDER])
    violin_plot(
        df_core, order_core,
        title="CE loss by category (core set)",
        out_path=os.path.join(args.outdir, "violin_core_categories.png"),
        ylim=tuple(args.ylim) if args.ylim else None
    )

    # Plot 2: other categories
    df_other, order_other = subset_order(df, OTHER_ORDER)
    violin_plot(
        df_other, order_other,
        title="CE loss by category (other set)",
        out_path=os.path.join(args.outdir, "violin_other_categories.png"),
        ylim=tuple(args.ylim) if args.ylim else None
    )

if __name__ == "__main__":
    main()
