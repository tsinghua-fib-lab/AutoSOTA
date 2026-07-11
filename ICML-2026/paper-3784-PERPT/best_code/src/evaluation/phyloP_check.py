#!/usr/bin/env python3
import argparse
import os
import json
import glob
import logging
import pathlib
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyBigWig
from pyfaidx import Fasta

# ----- project-style imports / path setup -----
import sys
sys.path.append("/home/mica/gamba/")
sys.path.append("/home/mica/gamba/src/")  # for src.evaluation.utils.*
from src.evaluation.utils.helpers import load_bed_file, extract_context

# ------------------------------- config --------------------------------
CATEGORY_ORDER = [
    "introns",
    "UCNE",
    "vista_enhancer",
    "coding_regions",   # falls back to 'exons' if no files
]

PLOT_PALETTE = "tab10"

# ----------------------------- logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------- helpers -----------------------------------
def _roi_span(info: Dict) -> Tuple[int, int]:
    fs = int(info["feature_start_in_window"])
    fe = int(info["feature_end_in_window"])
    if fe <= fs:
        return None
    return fs, fe

def _summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0 or np.isnan(s).all():
        return None
    valid = ~np.isnan(s)
    if not valid.any():
        return None
    s = s[valid]
    mean = float(np.mean(s))
    std  = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
    frac_pos = float(np.mean(s > 0))
    frac_neg = float(np.mean(s < 0))
    return {
        "mean": mean,
        "std": std,
        "frac_pos": frac_pos,
        "frac_neg": frac_neg,
        "n": int(s.size),
    }

def _find_beds_for_category(category: str, root="/home/mica/gamba/data_processing/data/regions") -> List[str]:
    p = os.path.join(root, category, "*.bed")
    files = glob.glob(p)
    # fallback: if coding_regions not found, try exons
    if category == "coding_regions" and len(files) == 0:
        files = glob.glob(os.path.join(root, "exons", "*.bed"))
    return files

def _load_regions(category: str, genome: Fasta, bw: pyBigWig.pyBigWig) -> List[Dict]:
    beds = _find_beds_for_category(category)
    print("Found {} bed files for category {}".format(len(beds), category))
    out = []
    for bf in beds:
        out.extend(load_bed_file(bf, category, genome, bw))
    return out

def _sample_regions(regions: List[Dict], k: int, seed: int) -> List[Dict]:
    if len(regions) <= k:
        return regions
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(regions), size=k, replace=False)
    return [regions[i] for i in idx]

def compute_actual_phylop_summary(
    bigwig_file: str,
    genome_fasta: str,
    categories: List[str],
    per_category_n: int,
    seed: int = 1337,
    model_type_for_context: str = "baseline",  # ensures extract_context aligns with your existing code
) -> pd.DataFrame:
    """
    For each category, sample up to N regions, extract per-base phyloP within ROI,
    and compute region-level summaries.
    """
    logging.info(f"Opening genome: {genome_fasta}")
    genome = Fasta(genome_fasta)

    logging.info(f"Opening phyloP bigWig: {bigwig_file}")
    bw = pyBigWig.open(bigwig_file)

    rows = []
    for cat in categories:
        logging.info(f"[{cat}] loading regions…")
        candidate_regions = _load_regions(cat, genome, bw)
        if len(candidate_regions) == 0:
            logging.warning(f"[{cat}] no regions found, skipping")
            continue

        sampled = _sample_regions(candidate_regions, per_category_n, seed)
        logging.info(f"[{cat}] sampled {len(sampled)} regions")

        # Build a context per region and summarize ROI phyloP
        kept = 0
        for r in sampled:
            ctx = extract_context(bigwig_file, r, genome, model_type=model_type_for_context)
            if not ctx or "scores" not in ctx:
                continue
            span = _roi_span(ctx)
            if span is None:
                continue
            fs, fe = span
            ss = np.asarray(ctx["scores"], dtype=np.float32)
            if ss.size == 0 or np.isnan(ss).all():
                continue
            ss_roi = ss[fs:fe]
            stats = _summarize_scores(ss_roi)
            if stats is None:
                continue
            kept += 1
            rows.append({
                "category": cat,
                "chrom": ctx.get("chrom", r.get("chrom")),
                "start": int(ctx.get("start", r.get("start", -1))),
                "end": int(ctx.get("end", r.get("end", -1))),
                "feature_start_in_window": int(fs),
                "feature_end_in_window": int(fe),
                "roi_len": int(fe - fs),
                "mean_phyloP": stats["mean"],
                "std_phyloP": stats["std"],
                "frac_pos": stats["frac_pos"],
                "frac_neg": stats["frac_neg"],
                "n_sites": stats["n"],
            })
        logging.info(f"[{cat}] kept {kept} regions with valid ROI phyloP")

    bw.close()

    if len(rows) == 0:
        logging.warning("No valid regions across all categories.")
        return pd.DataFrame(columns=[
            "category","chrom","start","end",
            "feature_start_in_window","feature_end_in_window","roi_len",
            "mean_phyloP","std_phyloP","frac_pos","frac_neg","n_sites"
        ])

    df = pd.DataFrame(rows)
    # enforce categorical order for plotting
    df["category"] = pd.Categorical(df["category"], categories=categories, ordered=True)
    return df

def _save_tables(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "actual_phyloP_region_stats.parquet", index=False)
    df.to_csv(outdir / "actual_phyloP_region_stats.csv", index=False)
    # brief category summary
    agg = (
        df.groupby("category")
          .agg(
              n_regions=("mean_phyloP", "size"),
              mean_of_means=("mean_phyloP", "mean"),
              std_of_means=("mean_phyloP", "std"),
              mean_of_stds=("std_phyloP", "mean"),
          )
          .reset_index()
    )
    agg.to_csv(outdir / "actual_phyloP_category_summary.csv", index=False)

def _plot_distributions(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Violin of per-region mean phyloP, one violin per category
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="category", y="mean_phyloP",
        inner="quartile", palette=PLOT_PALETTE, cut=0
    )
    plt.title("Distribution of region-level mean phyloP")
    plt.xlabel("")
    plt.ylabel("Mean phyloP per region")
    plt.tight_layout()
    plt.savefig(outdir / "actual_phyloP_mean_violin.png", dpi=300)
    plt.close()

    # Boxplot of per-region std of phyloP
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="category", y="std_phyloP",
        palette=PLOT_PALETTE
    )
    plt.title("Distribution of region-level phyloP variability (std)")
    plt.xlabel("")
    plt.ylabel("Std of phyloP per region")
    plt.tight_layout()
    plt.savefig(outdir / "actual_phyloP_std_box.png", dpi=300)
    plt.close()

    # Bar of category mean of means with error bars (SEM)
    g = (
        df.groupby("category")
          .agg(mu=("mean_phyloP", "mean"),
               sd=("mean_phyloP", "std"),
               n=("mean_phyloP", "size"))
          .reset_index()
    )
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    plt.figure(figsize=(8, 5))
    plt.bar(g["category"].astype(str), g["mu"], yerr=1.96*g["sem"], capsize=3)
    plt.title("Average region mean phyloP ± 95% CI")
    plt.xlabel("")
    plt.ylabel("Mean phyloP")
    plt.tight_layout()
    plt.savefig(outdir / "actual_phyloP_mean_bar_ci.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Compute actual phyloP rates per region for selected categories."
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to phyloP bigWig.",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to genome FASTA (used by helpers).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_representations/actual_phyloP_rates",
        help="Output directory.",
    )
    parser.add_argument(
        "--per_category_n", type=int, default=1000,
        help="Sample size per category."
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed for sampling."
    )
    parser.add_argument(
        "--categories", type=str, nargs="+",
        default=CATEGORY_ORDER,
        help="Categories to include."
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    logging.info(f"Writing outputs to {outdir}")

    df = compute_actual_phylop_summary(
        bigwig_file=args.bigwig_file,
        genome_fasta=args.genome_fasta,
        categories=args.categories,
        per_category_n=args.per_category_n,
        seed=args.seed,
        model_type_for_context="baseline",
    )

    if df.empty:
        logging.warning("No data produced. Exiting.")
        return

    _save_tables(df, outdir)
    _plot_distributions(df, outdir)

    # Minimal console summary
    cats = ", ".join([c for c in args.categories if c in set(df["category"].astype(str))])
    logging.info(f"Done. Categories summarized: {cats}")
    logging.info(f"Rows: {len(df)}  | saved: CSV, Parquet, and 3 figures.")

if __name__ == "__main__":
    main()
