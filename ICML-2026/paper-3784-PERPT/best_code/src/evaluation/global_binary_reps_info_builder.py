#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pyBigWig
import pandas as pd
from pyfaidx import Fasta

# project paths (same as original)
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
sys.path.append("/home/mica/gamba/src/")

from src.evaluation.utils.helpers import load_bed_file  # type: ignore

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- categories we sample ROIs from ----------------
CATEGORY_ORDER = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

# --------- GTF handling ----------

def parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
    """
    parse the 9th GTF column into a dict.
    very simple parser that looks for key "value"; pairs.
    """
    out: Dict[str, str] = {}
    if not isinstance(attr_str, str):
        return out
    for field in attr_str.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        parts = field.split(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        val = parts[1].strip().strip('"')
        out[key] = val
    return out


def load_gtf_for_chrom(gtf_dir: str, chrom: str) -> Optional[pd.DataFrame]:
    """
    load /{gtf_dir}/{chrom}.gtf into a DataFrame, if it exists.
    """
    gtf_path = os.path.join(gtf_dir, f"{chrom}.gtf")
    if not os.path.exists(gtf_path):
        logging.warning(f"GTF file not found for {chrom}: {gtf_path}")
        return None

    df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "chrom",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )
    return df


def annotate_region_with_gtf(
    chrom: str,
    start: int,
    end: int,
    gtf_df: Optional[pd.DataFrame],
) -> str:
    """
    find what GTF feature this region falls into.
    returns a string label or 'unknown'.
    overlap = any GTF row where intervals intersect.
    """
    if gtf_df is None or gtf_df.empty:
        return "unknown"

    overlaps = gtf_df[
        (gtf_df["end"] >= start) & (gtf_df["start"] <= end)
    ]
    if overlaps.empty:
        return "unknown"

    row = overlaps.iloc[0]
    attrs = parse_gtf_attributes(row["attribute"])
    gene_name = attrs.get("gene_name", attrs.get("gene_id", "unknown_gene"))
    feature = row["feature"]
    return f"{feature}:{gene_name}"


# --------- regions/CATEGORY labelling ----------

def discover_region_categories(regions_dir: str) -> List[str]:
    """
    find all *base* subdirectories under regions_dir to treat as annotation categories.
    ignore *_upstream and *_random dirs so we only annotate with real categories
    (promoters, UCNE, etc.), not synthetic sets.
    """
    cats = []
    for name in os.listdir(regions_dir):
        full = os.path.join(regions_dir, name)
        if not os.path.isdir(full):
            continue
        if name.endswith("_upstream") or name.endswith("_random"):
            continue
        cats.append(name)
    cats.sort()
    return cats


def load_region_beds(
    regions_dir: str,
    categories: List[str],
    chromosomes: List[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    load /regions_dir/{category}/{chrom}.bed (if exists) for each category+chrom.
    returns: category -> chrom -> df(chrom,start,end,...)
    """
    region_beds: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cat in categories:
        per_chrom: Dict[str, pd.DataFrame] = {}
        for chrom in chromosomes:
            bed_path = os.path.join(regions_dir, cat, f"{chrom}.bed")
            if not os.path.exists(bed_path):
                continue
            df = pd.read_csv(
                bed_path,
                sep="\t",
                header=None,
                comment="#",
            )
            if df.shape[1] >= 3:
                df = df.rename(columns={0: "chrom", 1: "start", 2: "end"})
            per_chrom[chrom] = df
        if per_chrom:
            region_beds[cat] = per_chrom
    return region_beds


def annotate_region_with_categories(
    chrom: str,
    start: int,
    end: int,
    region_beds: Dict[str, Dict[str, pd.DataFrame]],
) -> str:
    """
    check overlap of [start,end) with each category's BED regions for this chrom.
    returns ';'-separated list of categories, or 'unknown' if none.
    """
    matches = []
    for cat, per_chrom in region_beds.items():
        df = per_chrom.get(chrom)
        if df is None or df.empty:
            continue
        overlaps = df[
            (df["end"] > start) & (df["start"] < end)
        ]
        if not overlaps.empty:
            matches.append(cat)

    if not matches:
        return "unknown"
    return ";".join(matches)


# --------- paired base / alt loading ----------

def load_paired_regions_for_category(
    regions_dir: str,
    category: str,
    chromosomes: List[str],
    max_pairs: int,
    suffix: str,
) -> List[Dict[str, Any]]:
    """
    load paired base + alt regions from:
      regions_dir/category/chr*.bed
      regions_dir/category_{suffix}/chr*.bed

    assume both have a 'pair_id' column (7th column).
    returns list of dicts with base + alt coords and pair_id.

    suffix should be 'upstream' or 'random'.
    """
    base_dir = os.path.join(regions_dir, category)
    alt_dir = os.path.join(regions_dir, f"{category}_{suffix}")

    if not os.path.isdir(base_dir):
        logging.warning(f"[{category}/{suffix}] base dir not found: {base_dir}")
        return []
    if not os.path.isdir(alt_dir):
        logging.warning(f"[{category}/{suffix}] alt dir not found: {alt_dir}")
        return []

    pairs: List[Dict[str, Any]] = []

    for chrom in chromosomes:
        base_bed_path = os.path.join(base_dir, f"{chrom}.bed")
        alt_bed_path = os.path.join(alt_dir, f"{chrom}.bed")

        if not os.path.exists(base_bed_path) or not os.path.exists(alt_bed_path):
            continue

        base_df = pd.read_csv(
            base_bed_path,
            sep="\t",
            header=None,
            comment="#",
        )
        alt_df = pd.read_csv(
            alt_bed_path,
            sep="\t",
            header=None,
            comment="#",
        )

        # expect: chrom, start, end, name, score, strand, pair_id
        cols_full = ["chrom", "start", "end", "name", "score", "strand", "pair_id"]

        if base_df.shape[1] < 7 or alt_df.shape[1] < 7:
            raise ValueError(
                f"[{category}/{suffix}] expected at least 7 columns (including pair_id) in "
                f"{base_bed_path} and {alt_bed_path}, got {base_df.shape[1]}, {alt_df.shape[1]}"
            )

        base_df = base_df.iloc[:, :7]
        alt_df = alt_df.iloc[:, :7]
        base_df.columns = cols_full
        alt_df.columns = cols_full

        # merge on pair_id
        merged = pd.merge(
            base_df,
            alt_df,
            on="pair_id",
            suffixes=("_base", "_alt"),
        )

        logging.info(
            f"[{category}/{suffix}] {chrom}: {len(merged)} paired regions "
            f"(base: {len(base_df)}, alt: {len(alt_df)})"
        )

        for _, row in merged.iterrows():
            pairs.append(
                {
                    "chrom": row["chrom_base"],
                    "base_start": int(row["start_base"]),
                    "base_end": int(row["end_base"]),
                    "alt_start": int(row["start_alt"]),
                    "alt_end": int(row["end_alt"]),
                    "pair_id": row["pair_id"],
                }
            )

    if not pairs:
        logging.warning(f"[{category}/{suffix}] no paired regions found on requested chromosomes")

    if max_pairs is not None and max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    logging.info(f"[{category}/{suffix}] using {len(pairs)} paired base/{suffix} regions total")
    return pairs


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "annotate pre-computed ROI + 2kb-upstream OR random pairs "
            "using per-chromosome GTF + regions/CATEGORY BEDs"
        )
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="path to phyloP bigwig file (kept for compatibility; not used heavily here)",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="path to genome fasta",
    )
    parser.add_argument(
        "--gtf_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/gtfs",
        help="directory containing per-chromosome GTFs, e.g. {gtf_dir}/chr1.gtf",
    )
    parser.add_argument(
        "--regions_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/regions",
        help="root directory containing CATEGORY, CATEGORY_upstream, CATEGORY_random subdirs",
    )
    parser.add_argument(
        "--region_type",
        type=str,
        choices=["upstream", "random"],
        default="upstream",
        help="which paired regions to annotate; expects CATEGORY_<region_type> subdirs",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data_processing/data/240-mammalian/region_info/",
        help=(
            "output TSV path. if empty, auto-named as "
            "'{region_type}_region_annotations.tsv' in the same dir as genome_fasta"
        ),
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="max number of paired regions per category",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
            "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
            "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
            "chr19", "chr20", "chr21", "chr22", "chrX",
        ],
        help="chromosomes to include (and expect corresponding GTF/BED files)",
    )
    parser.add_argument(
        "--upstream_offset",
        type=int,
        default=2000,
        help="kept for bookkeeping; upstreams were already created with this offset",
    )

    args = parser.parse_args()


    # derive default output path if not provided
    if args.output_folder:
        out_path =  Path(f"{args.output_folder}/{args.region_type}_region_annotations.tsv")
    else:
        out_dir = Path(args.genome_fasta).parent
        out_path = out_dir / f"{args.region_type}_region_annotations.tsv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"region_type: {args.region_type}")
    logging.info(f"genome: {args.genome_fasta}")
    logging.info(f"gtf dir: {args.gtf_dir}")
    logging.info(f"regions dir: {args.regions_dir}")
    logging.info(f"output tsv: {out_path}")

    genome = Fasta(args.genome_fasta)  # not used directly right now, but cheap to keep

    # load GTF per chromosome once
    gtf_by_chrom: Dict[str, Optional[pd.DataFrame]] = {}
    for chrom in args.chromosomes:
        gtf_by_chrom[chrom] = load_gtf_for_chrom(args.gtf_dir, chrom)

    # discover all *base* region categories and load their BEDs for annotation
    all_region_categories = discover_region_categories(args.regions_dir)
    logging.info(f"annotation categories (regions/): {all_region_categories}")
    region_beds = load_region_beds(args.regions_dir, all_region_categories, args.chromosomes)

    rows = []

    # bigwig not strictly needed here, but keeping open/close to mirror original api
    bw = pyBigWig.open(args.bigwig_file)

    for category in CATEGORY_ORDER:
        logging.info(f"processing category={category} ({args.region_type})")

        pairs = load_paired_regions_for_category(
            regions_dir=args.regions_dir,
            category=category,
            chromosomes=args.chromosomes,
            max_pairs=args.num_regions,
            suffix=args.region_type,
        )
        if not pairs:
            continue

        cat_rows_before = len(rows)

        for pair in pairs:
            chrom = pair["chrom"]
            base_start = pair["base_start"]
            base_end = pair["base_end"]
            alt_start = pair["alt_start"]
            alt_end = pair["alt_end"]
            pair_id = pair["pair_id"]

            gtf_df = gtf_by_chrom.get(chrom)
            gtf_annotation = annotate_region_with_gtf(chrom, alt_start, alt_end, gtf_df)
            category_annotation = annotate_region_with_categories(
                chrom, alt_start, alt_end, region_beds
            )

            rows.append(
                {
                    "chrom": chrom,
                    "pair_id": pair_id,
                    "category_its_upstream_of": category,
                    "category_start_pos": base_start,
                    "category_end_pos": base_end,
                    "start_pos": alt_start,
                    "end_pos": alt_end,
                    "region_type": args.region_type,  # 'upstream' or 'random'
                    "region_identified_by_gtf": gtf_annotation,
                    "region_identified_by_category": category_annotation,
                    "upstream_offset": args.upstream_offset,
                }
            )

        logging.info(
            f"[{category}/{args.region_type}] collected {len(rows) - cat_rows_before} rows "
            f"(total so far: {len(rows)})"
        )

    bw.close()

    if not rows:
        logging.warning("no regions collected; nothing to write")
        return

    df = pd.DataFrame(rows)
    logging.info(f"writing {len(df)} rows to {out_path}")
    df.to_csv(out_path, sep="\t", index=False)
    logging.info("done.")


if __name__ == "__main__":
    main()
