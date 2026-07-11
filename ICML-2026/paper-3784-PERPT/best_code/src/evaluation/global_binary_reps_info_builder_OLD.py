#!/usr/bin/env python3
import argparse
import os
import sys
import glob
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

# ---------------- helpers ----------------
def make_upstream_region(
    region: Dict[str, Any],
    genome: Fasta,
    offset: int = 2000,
) -> Optional[Dict[str, Any]]:
    """
    shift region upstream by `offset` bp, preserving length.
    returns new region dict or None if out-of-bounds.
    """
    chrom = region["chrom"]
    start = int(region["start"])
    end = int(region["end"])
    length = end - start
    if length <= 0:
        return None

    new_start = start - offset
    new_end = new_start + length
    if new_start < 0:
        return None

    chrom_len = len(genome[chrom])
    if new_end > chrom_len:
        return None

    new_region = dict(region)
    new_region["start"] = new_start
    new_region["end"] = new_end
    return new_region


def collect_base_regions_for_category(
    category: str,
    genome: Fasta,
    bigwig_file: Any,
    chromosomes: List[str],
    max_regions: int,
) -> List[Dict[str, Any]]:
    """
    load regions for a category from
    /home/mica/gamba/data_processing/data/OLD_regions/{category}/*.bed,
    filter by chrom, truncate to max_regions.
    """
    bed_pattern = f"/home/mica/gamba/data_processing/data/OLD_regions/{category}/*.bed"
    bed_files = glob.glob(bed_pattern)
    if not bed_files:
        logging.warning(f"[{category}] no BED files matching {bed_pattern}")
        return []

    base_regions: List[Dict[str, Any]] = []
    for bed_file in bed_files:
        loaded = load_bed_file(bed_file, category, genome, bigwig_file)
        base_regions.extend([r for r in loaded if r["chrom"] in chromosomes])

    if not base_regions:
        logging.warning(f"[{category}] no regions on requested chromosomes {chromosomes}")
        return []

    base_regions = base_regions[:max_regions]
    logging.info(f"[{category}] using {len(base_regions)} base regions")
    return base_regions


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
    find all subdirectories under regions_dir to treat as annotation categories.
    e.g., coding_regions, exons, introns, phyloP_negative, ...
    """
    cats = []
    for name in os.listdir(regions_dir):
        full = os.path.join(regions_dir, name)
        if os.path.isdir(full):
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
            # assume standard BED: chrom, start, end are first 3 columns
            # keep all cols but name first three for clarity
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
        # using half-open-ish overlap check; good enough for this purpose
        overlaps = df[
            (df["end"] > start) & (df["start"] < end)
        ]
        if not overlaps.empty:
            matches.append(cat)

    if not matches:
        return "unknown"
    return ";".join(matches)


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "sample ROI + 2kb-upstream regions and annotate upstream regions "
            "using per-chromosome GTF + regions/CATEGORY BEDs"
        )
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="path to phyloP bigwig file (needed by load_bed_file)",
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
        help="root directory containing CATEGORY subdirs with chr*.bed",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/upstream_region_annotations.tsv",
        help="output TSV with upstream region annotations",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="max number of base regions per category (for the upstream-of side)",
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
        help="distance upstream to shift each base region",
    )

    args = parser.parse_args()

    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"genome: {args.genome_fasta}")
    logging.info(f"bigwig: {args.bigwig_file}")
    logging.info(f"gtf dir: {args.gtf_dir}")
    logging.info(f"regions dir: {args.regions_dir}")
    logging.info(f"output tsv: {out_path}")

    genome = Fasta(args.genome_fasta)

    # load GTF per chromosome once
    gtf_by_chrom: Dict[str, Optional[pd.DataFrame]] = {}
    for chrom in args.chromosomes:
        gtf_by_chrom[chrom] = load_gtf_for_chrom(args.gtf_dir, chrom)

    # discover all region categories and load their BEDs for annotation
    all_region_categories = discover_region_categories(args.regions_dir)
    logging.info(f"annotation categories (regions/): {all_region_categories}")
    region_beds = load_region_beds(args.regions_dir, all_region_categories, args.chromosomes)

    # main collection
    rows = []

    # open bigwig once (load_bed_file uses it)
    bw = pyBigWig.open(args.bigwig_file)

    for category in CATEGORY_ORDER:
        logging.info(f"processing category={category}")

        base_regions = collect_base_regions_for_category(
            category=category,
            genome=genome,
            bigwig_file=bw,
            chromosomes=args.chromosomes,
            max_regions=args.num_regions,
        )
        if not base_regions:
            continue

        for region_idx, region in enumerate(base_regions):
            chrom = region["chrom"]
            category_start = int(region["start"])
            category_end = int(region["end"])

            upstream = make_upstream_region(
                region,
                genome,
                offset=args.upstream_offset,
            )
            if upstream is None:
                continue

            up_start = int(upstream["start"])
            up_end = int(upstream["end"])

            gtf_df = gtf_by_chrom.get(chrom)
            gtf_annotation = annotate_region_with_gtf(chrom, up_start, up_end, gtf_df)
            category_annotation = annotate_region_with_categories(
                chrom, up_start, up_end, region_beds
            )

            rows.append(
                {
                    "chrom": chrom,
                    "category_its_upstream_of": category,
                    "category_start_pos": category_start,
                    "category_end_pos": category_end,
                    "start_pos": up_start,
                    "end_pos": up_end,
                    "region_identified_by_gtf": gtf_annotation,
                    "region_identified_by_category": category_annotation,
                    "region_idx_in_bed": region_idx,
                    "upstream_offset": args.upstream_offset,
                }
            )

        logging.info(f"[{category}] collected {len(rows)} total upstream rows so far")

    bw.close()

    if not rows:
        logging.warning("no upstream regions collected; nothing to write")
        return

    df = pd.DataFrame(rows)
    logging.info(f"writing {len(df)} rows to {out_path}")
    df.to_csv(out_path, sep="\t", index=False)
    logging.info("done.")


if __name__ == "__main__":
    main()
