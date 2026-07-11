#!/usr/bin/env python3
import argparse, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# categories
RENAME_MAP = {"upstream_TSS": "2kb upstream of TSS"}
CORE_ORDER_RAW = [ "repeats", "introns", "UTR5", "UTR3", "vista_enhancer", "UCNE", "exons", "coding_regions", "upstream_TSS", "promoters"]
CORE_ORDER = [RENAME_MAP.get(c, c) for c in CORE_ORDER_RAW]
# OTHER_ORDER = ["repeats", "introns", "UTR5", "UTR3"]

# colors
GREEN = "#4287f5"
RED   = "#FF8C32"
GREY  = "#616161"
PALETTE = [GREEN, RED, GREY]

def load_df(path: str, label: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = ["category", "chrom", "data_split", "loss"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {path}")
    df = df[need].copy()
    df["category"] = df["category"].replace(RENAME_MAP)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["model"] = label
    return df.dropna(subset=["category", "loss"])

def subset_and_order(df: pd.DataFrame, wanted: list[str]) -> tuple[pd.DataFrame, list[str]]:
    present = [c for c in wanted if c in set(df["category"])]
    return df[df["category"].isin(present)].copy(), present

def violin_plot(df: pd.DataFrame, order: list[str], title: str, out_path: str, ylim=None):
    if df.empty or not order:
        print(f"[warn] no data for {title}")
        return
    plt.figure(figsize=(1.6*max(6, len(order)), 6))
    ax = sns.violinplot(
        data=df, x="category", y="loss",
        order=order, hue="model",
        scale="width", inner="quartile", cut=0, dodge=True,
        palette=PALETTE
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
    #save as svg
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()
    print(f"[ok] {out_path}")

def main():
    p = argparse.ArgumentParser(description="Compare CE-loss violins from three Parquet DFs.")
    p.add_argument("--parquet_a", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus_dual_step_44000/all_results.parquet")
    p.add_argument("--parquet_b", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus_seq_only_step_44000/all_results.parquet")
    p.add_argument("--parquet_c", default="/home/mica/gamba/data_processing/data/240-mammalian/phylop_corr_analysis/caduceus-theirs/all_results.parquet")
    p.add_argument("--label_a", default="Bi-Gamba MLM+MEM")
    p.add_argument("--label_b", default="Bi-Gamba MLM-only")
    p.add_argument("--label_c", default="Caduceus")
    p.add_argument("--outdir", default="/home/mica/gamba/data_processing/data/240-mammalian/figures")
    p.add_argument("--ylim", type=float, nargs=2, default=[0.0, 3.0])
    args = p.parse_args()

    dfa = load_df(args.parquet_a, args.label_a)
    dfb = load_df(args.parquet_b, args.label_b)
    dfc = load_df(args.parquet_c, args.label_c)
    df = pd.concat([dfa, dfb, dfc], ignore_index=True)

    # Plot 1: core set
    df_core, order_core = subset_and_order(df, CORE_ORDER)
    violin_plot(df_core, order_core, "CE loss by category (Biologically)",
                os.path.join(args.outdir, "violin_core_categories.png"),
                ylim=tuple(args.ylim))

    # # Plot 2: other set
    # df_other, order_other = subset_and_order(df, OTHER_ORDER)
    # violin_plot(df_other, order_other, "CE loss by category (Other set)",
    #             os.path.join(args.outdir, "violin_other_categories.png"),
    #             ylim=tuple(args.ylim))

if __name__ == "__main__":
    main()
