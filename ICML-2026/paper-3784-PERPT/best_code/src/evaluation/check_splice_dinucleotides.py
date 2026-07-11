#!/usr/bin/env python3
"""
check_splice_dinucleotides.py

Checks that label1 (true) + label2-5 (cryptic/annotated near/far) splice sites
all have the same 2bp dinucleotide (strand-aware) for donor/acceptor.

Typical expectations (on transcript strand):
  donor   -> GT
  acceptor-> AG

Example:
  python check_splice_dinucleotides.py \
    --splice_tsv_dir /home/mica/gamba/data_processing/data/splice_sites \
    --genome_fasta   /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
    --site_type donor \
    --chromosomes chr21 chr22 \
    --max_rows_per_chrom 20000

Notes:
- Assumes TSV columns like in your pipeline:
  label1_pos, label2_pos_cryptic_near, label3_pos_cryptic_far,
  label4_pos_annotated_near, label5_pos_annotated_far, strand
- Handles '.' missing entries.
- If strand is '-', it reverse-complements the fetched 2bp so you always see
  dinucs in transcript orientation.
"""

import argparse
import os
import glob
from collections import Counter, defaultdict

import pandas as pd
from pyfaidx import Fasta

RC = str.maketrans("acgtACGT", "tgcaTGCA")

LABEL_COLS = {
    1: "label1_pos",
    2: "label2_pos_cryptic_near",
    3: "label3_pos_cryptic_far",
    4: "label4_pos_annotated_near",
    5: "label5_pos_annotated_far",
}

LABEL_NAMES = {
    1: "true(label1)",
    2: "cryptic_near(label2)",
    3: "cryptic_far(label3)",
    4: "annot_near(label4)",
    5: "annot_far(label5)",
}

def revcomp(seq: str) -> str:
    return seq.translate(RC)[::-1]

def fetch_2bp(genome: Fasta, chrom: str, pos0: int, strand: str) -> str:
    # pyfaidx uses 0-based, end-exclusive slices.
    s = str(genome[chrom][pos0:pos0+2]).upper()
    if len(s) != 2:
        return "??"
    if any(b not in "ACGT" for b in s):
        return "NN"
    if strand == "-":
        s = revcomp(s)
    return s

def fmt_top(counter: Counter, k=10) -> str:
    total = sum(counter.values())
    items = counter.most_common(k)
    parts = [f"{dinuc}:{n} ({n/total:.1%})" for dinuc, n in items]
    if len(counter) > k:
        parts.append(f"... +{len(counter)-k} more")
    return ", ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice_tsv_dir", required=True)
    ap.add_argument("--genome_fasta", required=True)
    ap.add_argument("--site_type", choices=["donor", "acceptor"], required=True)
    ap.add_argument("--chromosomes", nargs="+", default=["chr22"])
    ap.add_argument("--max_rows_per_chrom", type=int, default=20000,
                   help="cap rows read per chromosome file (after filtering missing labels)")
    ap.add_argument("--require_all_labels", action="store_true",
                   help="if set, keep only rows where labels 2-5 are all present (like your embed script)")
    ap.add_argument("--tolerance", type=float, default=0.01,
                   help="allowed fraction off-target from expected dinuc per label before failing")
    args = ap.parse_args()

    genome = Fasta(args.genome_fasta, as_raw=True, sequence_always_upper=True)

    expected = "GT" if args.site_type == "donor" else "AG"

    # counters[label_id] -> Counter of dinucs
    counters = {lid: Counter() for lid in LABEL_COLS}
    bad_rows = defaultdict(int)

    total_sites = 0

    for chrom in args.chromosomes:
        pattern = os.path.join(args.splice_tsv_dir, f"{chrom}_{args.site_type}_labels.tsv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[warn] missing: {pattern}")
            continue

        tsv = matches[0]
        df = pd.read_csv(tsv, sep="\t")

        # drop obvious missing label1
        df = df[df["label1_pos"] != "."].copy()

        if args.require_all_labels:
            mask_all = True
            for lid in (2,3,4,5):
                mask_all = mask_all & (df[LABEL_COLS[lid]] != ".")
            df = df[mask_all].copy()

        if args.max_rows_per_chrom is not None and len(df) > args.max_rows_per_chrom:
            df = df.sample(args.max_rows_per_chrom, random_state=0)

        for _, row in df.iterrows():
            strand = row.get("strand", "+")
            if strand not in ["+", "-"]:
                strand = "+"

            for lid, col in LABEL_COLS.items():
                v = row.get(col, ".")
                if v == "." or pd.isna(v):
                    bad_rows[f"missing_L{lid}"] += 1
                    continue
                try:
                    pos = int(v)
                except Exception:
                    bad_rows[f"nonint_L{lid}"] += 1
                    continue

                dinuc = fetch_2bp(genome, row["chrom"], pos, strand)
                counters[lid][dinuc] += 1

            total_sites += 1

    print(f"\nchecked site_type={args.site_type} expected={expected}")
    print(f"rows processed (per-chrom sampled): {total_sites}\n")

    # report per label
    any_fail = False
    for lid in sorted(counters):
        c = counters[lid]
        n = sum(c.values())
        if n == 0:
            print(f"L{lid} {LABEL_NAMES[lid]}: n=0 (no data)")
            any_fail = True
            continue

        off = n - c.get(expected, 0)
        off_frac = off / n

        print(f"L{lid} {LABEL_NAMES[lid]}: n={n}")
        print(f"  top: {fmt_top(c, k=8)}")
        print(f"  expected {expected}: {c.get(expected, 0)} ({c.get(expected, 0)/n:.1%}) | off-target: {off} ({off_frac:.1%})")

        if off_frac > args.tolerance:
            print(f"  [FAIL] off-target fraction {off_frac:.3%} > tolerance {args.tolerance:.3%}")
            any_fail = True
        print("")

    if bad_rows:
        print("row issues:")
        for k, v in sorted(bad_rows.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")
        print("")

    # optional: check “same nucleotide distribution across labels” more strictly
    # (i.e., not only matching expected, but also that label2-5 match label1 distribution).
    base = counters[1]
    base_total = sum(base.values()) or 1
    base_freq = {k: v/base_total for k, v in base.items()}

    def l1_distance(freq_a, freq_b):
        keys = set(freq_a) | set(freq_b)
        return sum(abs(freq_a.get(k,0.0) - freq_b.get(k,0.0)) for k in keys)

    print("distribution distance vs label1 (L1 distance; 0 means identical):")
    for lid in sorted(counters):
        c = counters[lid]
        tot = sum(c.values()) or 1
        freq = {k: v/tot for k, v in c.items()}
        d = l1_distance(base_freq, freq)
        print(f"  L{lid} vs L1: {d:.4f}")
    print("")

    if any_fail:
        raise SystemExit(2)

if __name__ == "__main__":
    main()

# python src/evaluation/check_splice_dinucleotides.py \
#   --splice_tsv_dir /home/mica/gamba/data_processing/data/splice_sites \
#   --genome_fasta   /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
#   --site_type acceptor \
#   --chromosomes chr21 chr22 \
#   --require_all_labels \
#   --max_rows_per_chrom 50000
