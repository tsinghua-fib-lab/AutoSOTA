#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "for each ROI category, compute the % of regions (upstream or random) whose "
            "category-based annotation matches the ROI category, and plot the "
            "distribution of region categories"
        )
    )
    parser.add_argument(
        "--input_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/region_info/upstream_region_annotations.tsv",
        help="input TSV produced by region annotation script",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/region_info/upstream_region_match_summary.tsv",
        help="optional path to write summary TSV; if empty, no file is written",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/region_info/upstream_info/",  # will be set based on region_type if empty
        help="directory to save per-category distribution plots",
    )
    parser.add_argument(
        "--region_type",
        type=str,
        choices=["upstream", "random"],
        default="upstream",
        help="which region_type to summarize (expects column 'region_type' if present)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_tsv)
    if not input_path.exists():
        raise FileNotFoundError(f"input TSV not found: {input_path}")

    logging.info(f"loading {input_path}")
    df = pd.read_csv(input_path, sep="\t")

    # if region_type column exists, filter to requested type
    if "region_type" in df.columns:
        before = len(df)
        df = df[df["region_type"] == args.region_type].copy()
        logging.info(
            f"filtered to region_type={args.region_type}: {len(df)} rows "
            f"(from {before})"
        )

    required_cols = [
        "category_its_upstream_of",
        "region_identified_by_category",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing required column in TSV: {col}")

    # normalize region_identified_by_category to string
    df["region_identified_by_category"] = df["region_identified_by_category"].fillna(
        "unknown"
    ).astype(str)

    # helper: does this region include the same category label as the ROI category?
    def has_same_category(row) -> bool:
        roi_cat = row["category_its_upstream_of"]
        region_cats = row["region_identified_by_category"]
        if region_cats == "unknown" or region_cats.strip() == "":
            return False
        cats = [c.strip() for c in region_cats.split(";") if c.strip()]
        return roi_cat in cats

    df["same_category_region"] = df.apply(has_same_category, axis=1)

    # for extra context, mark unknowns
    df["region_unknown"] = df["region_identified_by_category"].eq("unknown")

    # aggregate per ROI category
    grouped = df.groupby("category_its_upstream_of").agg(
        total_regions=("same_category_region", "size"),
        num_same_category=("same_category_region", "sum"),
        num_unknown=("region_unknown", "sum"),
    )
    grouped["pct_same_category"] = (
        grouped["num_same_category"] / grouped["total_regions"] * 100.0
    )
    grouped["pct_unknown"] = (
        grouped["num_unknown"] / grouped["total_regions"] * 100.0
    )

    grouped = grouped.reset_index().rename(
        columns={"category_its_upstream_of": "category"}
    )

    # print nicely
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print(f"\nsummary for region_type={args.region_type}\n")
    print(grouped.sort_values("category").to_string(index=False))

    # optionally write to file
    if args.output_tsv:
        out_path = Path(args.output_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"writing summary to {out_path}")
        grouped.to_csv(out_path, sep="\t", index=False)

    # ----------------------------------------------------
    # per-category plots of region category distribution
    # ----------------------------------------------------

    # figure out plot directory
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
    else:
        # default: .../region_info/{region_type}_info
        # base off the directory of the input_tsv
        base_dir = input_path.parent
        plot_dir = base_dir / "region_info" / f"{args.region_type}_info"

    plot_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"saving plots to {plot_dir}")

    # explode region categories so each category gets its own row
    def parse_region_categories(s: str):
        s = s.strip()
        if s == "" or s == "unknown":
            return ["unknown"]
        parts = [c.strip() for c in s.split(";") if c.strip()]
        return parts if parts else ["unknown"]

    df_exploded = df.copy()
    df_exploded["region_category_list"] = df_exploded[
        "region_identified_by_category"
    ].apply(parse_region_categories)
    df_exploded = df_exploded.explode("region_category_list")
    df_exploded = df_exploded.rename(
        columns={"region_category_list": "region_category"}
    )

    # plot for each roi category
    for roi_cat, sub in df_exploded.groupby("category_its_upstream_of"):
        counts = sub["region_category"].value_counts().sort_index()
        total = counts.sum()
        pct = counts / total * 100.0

        fig, ax = plt.subplots(figsize=(8, 4))
        pct.plot(kind="bar", ax=ax)

        ax.set_title(
            f"{args.region_type} region categories for ROI category: {roi_cat}"
        )
        ax.set_xlabel("region category")
        ax.set_ylabel(f"percentage of {args.region_type} regions (%)")
        ax.set_ylim(0, 100)

        # rotate and right-align x tick labels
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

        plt.tight_layout()

        plot_path = plot_dir / f"{args.region_type}_distribution_{roi_cat}.png"
        logging.info(f"saving plot for {roi_cat} to {plot_path}")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

# python /home/mica/gamba/src/evaluation/global_binary_summarize_upstream_matches.py --region_type upstream

# python /home/mica/gamba/src/evaluation/global_binary_summarize_upstream_matches.py\
#   --region_type random \
#   --input_tsv /home/mica/gamba/data_processing/data/240-mammalian/region_info/random_region_annotations.tsv \
#   --output_tsv /home/mica/gamba/data_processing/data/240-mammalian/region_info/random_region_match_summary.tsv \
#   --plot_dir /home/mica/gamba/data_processing/data/240-mammalian/region_info/random_info/
