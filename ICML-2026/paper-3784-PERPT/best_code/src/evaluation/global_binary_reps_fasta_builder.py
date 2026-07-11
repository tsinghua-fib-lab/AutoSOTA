#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any
import pyBigWig
import pandas as pd
from pyfaidx import Fasta

# project paths (same as original)
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
sys.path.append("/home/mica/gamba/src/")

from src.evaluation.utils.helpers import load_bed_file, extract_context  # type: ignore

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------- categories ----------------
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
def make_upstream_region(region: Dict[str, Any], genome: Fasta, offset: int = 2000):
    """
    shift region upstream by `offset` bp, preserving length.
    returns new region dict or None if out-of-bounds.

    identical logic to the original code.
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
    bigwig_file: str,
    chromosomes: List[str],
    max_regions: int,
) -> List[Dict[str, Any]]:
    """
    load regions for a category from
    /home/mica/gamba/data_processing/data/regions/{category}/*.bed,
    filter by chrom, truncate to max_regions.
    """
    bed_pattern = f"/home/mica/gamba/data_processing/data/regions/{category}/*.bed"
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


def main():
    parser = argparse.ArgumentParser(
        description="sample ROI + 2kb-upstream contexts into FASTA with metadata "
        "using gamba-style asymmetric 2048bp windows"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="path to phyloP bigwig file (same as original)",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="path to genome fasta",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default='/home/mica/gamba/data_processing/data/240-mammalian/global_representations_upstream_pairs/',
        help="output prefix (e.g., /path/to/upstream_pairs); "
             "will write prefix.fa, prefix_meta.tsv, prefix_category_codes.json",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="max number of base regions per category",
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
        help="chromosomes to include",
    )
    parser.add_argument(
        "--max_window",
        type=int,
        default=2048,
        help="context window length passed to extract_context (default 2048, gamba-style asym)",
    )

    args = parser.parse_args()

    out_prefix = Path(args.output_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    fasta_path = out_prefix / "sequences.fa"
    meta_path = out_prefix / "meta.tsv"
    codes_path = out_prefix / "category_codes.json"


    logging.info(f"genome: {args.genome_fasta}")
    logging.info(f"bigwig: {args.bigwig_file}")
    logging.info(f"output fasta: {fasta_path}")
    logging.info(f"output meta:  {meta_path}")
    logging.info(f"output codes: {codes_path}")

    genome = Fasta(args.genome_fasta)

    # category -> integer code (0-based) using CATEGORY_ORDER
    category_codes = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}

    # write code mapping json
    import json
    with open(codes_path, "w") as f:
        json.dump(category_codes, f, indent=2)

    sequences = []  # list of (seq_id, sequence)
    meta_rows = []

    for category in CATEGORY_ORDER:
        logging.info(f"processing category={category}")
        code = category_codes[category]

        #open bigwig
        bw = pyBigWig.open(args.bigwig_file)

        base_regions = collect_base_regions_for_category(
            category=category,
            genome=genome,
            bigwig_file=bw,
            chromosomes=args.chromosomes,
            max_regions=args.num_regions,
        )
        bw.close()
        if not base_regions:
            continue

        pair_idx = 0  # index of roi/upstream pair within this category

        for region_idx, region in enumerate(base_regions):
            upstream = make_upstream_region(region, genome, offset=2000)
            if upstream is None:
                continue

            # apply gamba-style asymmetric 2048bp window:
            # [context_before] + [region], total length <= max_window
            pos_ctx = extract_context(
                args.bigwig_file,
                region,
                genome,
                model_type="gamba",
                context_window=args.max_window,
            )
            neg_ctx = extract_context(
                args.bigwig_file,
                upstream,
                genome,
                model_type="gamba",
                context_window=args.max_window,
            )

            if pos_ctx is None or neg_ctx is None:
                continue

            # attach metadata and write two sequences:
            # >{code}.{pair_idx}.1  -> ROI
            # >{code}.{pair_idx}.0  -> upstream (neg)
            for is_roi, ctx in ((1, pos_ctx), (0, neg_ctx)):
                seq = ctx["sequence"]
                chrom = ctx["chrom"]
                start = int(ctx["start"])
                end = int(ctx["end"])
                strand = ctx.get("strand", "+")

                fs = int(ctx.get("feature_start_in_window", 0))
                fe = int(ctx.get("feature_end_in_window", 0))

                seq_id = f"{code}.{pair_idx}.{is_roi}"

                sequences.append((seq_id, seq))

                meta_rows.append(
                    {
                        "seq_id": seq_id,
                        "category": category,
                        "code": code,
                        "pair_idx": pair_idx,      # within category
                        "is_roi": int(is_roi),     # 1=ROI,0=upstream
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "feature_start_in_window": fs,
                        "feature_end_in_window": fe,
                        "region_idx_in_bed": region_idx,
                    }
                )

            pair_idx += 1

        logging.info(f"[{category}] wrote {pair_idx} ROI/upstream pairs "
                     f"({2 * pair_idx} sequences)")

    if not sequences:
        logging.warning("no sequences collected; nothing to write")
        return

    # write FASTA
    logging.info(f"writing FASTA with {len(sequences)} sequences to {fasta_path}")
    with open(fasta_path, "w") as f_out:
        for seq_id, seq in sequences:
            f_out.write(f">{seq_id}\n")
            # plain single-line sequences (no wrapping)
            f_out.write(f"{seq}\n")

    # write metadata
    logging.info(f"writing metadata for {len(meta_rows)} sequences to {meta_path}")
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(meta_path, sep="\t", index=False)

    logging.info("done.")


if __name__ == "__main__":
    main()
