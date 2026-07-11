#!/usr/bin/env python3
"""
validate_splice_labels.py

One-stop validation for chr*_{donor,acceptor}_labels.tsv produced by find_splice_labels.py.

Checks:
1) Strand-aware coordinate sanity for label1_pos vs intron boundaries.
2) Strand-normalized dinucleotide purity for label1 (expected AG for acceptor, GT for donor).
3) Strand-normalized dinucleotide purity for labels 1..5 (and top-k summaries).
4) Distribution distance (L1) of each label vs label1.

Exit code:
  0 = all checks pass
  2 = any check fails

Example:
  python validate_splice_labels.py \
    --splice_tsv_dir /home/mica/gamba/data_processing/data/splice_sites \
    --genome_fasta   /home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa \
    --site_type acceptor \
    --chromosomes chr1 chr2 chr22 \
    --sample_per_chrom 50000
"""

import argparse
import os
import glob
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pyfaidx import Fasta


RC = str.maketrans("ACGT", "TGCA")

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


def revcomp(s: str) -> str:
    return s.translate(RC)[::-1]


def fetch_2bp(genome: Fasta, chrom: str, pos0: int) -> str:
    s = str(genome[chrom][pos0:pos0 + 2]).upper()
    if len(s) != 2:
        return "??"
    if any(b not in "ACGT" for b in s):
        return "NN"
    return s


def fmt_top(counter: Counter, k=8) -> str:
    tot = sum(counter.values()) or 1
    items = counter.most_common(k)
    parts = [f"{dinuc}:{n} ({n/tot:.1%})" for dinuc, n in items]
    if len(counter) > k:
        parts.append(f"... +{len(counter)-k} more")
    return ", ".join(parts)


def l1_distance(freq_a: dict, freq_b: dict) -> float:
    keys = set(freq_a) | set(freq_b)
    return float(sum(abs(freq_a.get(k, 0.0) - freq_b.get(k, 0.0)) for k in keys))


def main():
    ap = argparse.ArgumentParser(description="Validate splice label TSVs (strand-aware).")
    ap.add_argument("--splice_tsv_dir", default="/home/mica/gamba/data_processing/data/splice_sites/")
    ap.add_argument("--genome_fasta", default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa")
    ap.add_argument("--site_type", choices=["donor", "acceptor"], required=True)
    ap.add_argument("--chromosomes", nargs="+", default=["chr22"])
    ap.add_argument("--sample_per_chrom", type=int, default=50000,
                    help="sample at most this many rows per chromosome TSV (after filtering)")
    ap.add_argument("--require_all_labels", action="store_true",
                    help="if set, keep only rows where labels 2-5 are all present")
    ap.add_argument("--motif_min_frac", type=float, default=0.95,
                    help="minimum fraction of expected dinuc required for each label to pass")
    ap.add_argument("--max_l1_distance", type=float, default=0.05,
                    help="maximum allowed L1 distance vs label1 distribution for labels 2-5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    expected = "GT" if args.site_type == "donor" else "AG"

    genome = Fasta(args.genome_fasta, as_raw=True, sequence_always_upper=True)

    # aggregate across chromosomes
    coord_bad_by_strand = Counter()
    coord_total_by_strand = Counter()
    missing_counts = Counter()

    dinuc_counts = {lid: Counter() for lid in LABEL_COLS}  # strand-normalized
    total_rows = 0

    rng = np.random.default_rng(args.seed)

    # load + sample
    for chrom in args.chromosomes:
        pattern = os.path.join(args.splice_tsv_dir, f"{chrom}_{args.site_type}_labels.tsv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[warn] missing TSV: {pattern}")
            continue

        tsv = matches[0]
        df = pd.read_csv(tsv, sep="\t")

        if args.require_all_labels:
            mask = df[LABEL_COLS[2]] != "."
            for lid in (3, 4, 5):
                mask = mask & (df[LABEL_COLS[lid]] != ".")
            df = df[mask].copy()

        if args.sample_per_chrom and len(df) > args.sample_per_chrom:
            # deterministic-ish sample
            df = df.sample(args.sample_per_chrom, random_state=args.seed)

        for _, r in df.iterrows():
            strand = r.get("strand", "+")
            if strand not in ["+", "-"]:
                strand = "+"

            # ----- check 1: strand-aware coordinate sanity (label1 vs intron boundaries)
            try:
                pos = int(r["label1_pos"])
                is0 = int(r["intron_start"])
                ie0 = int(r["intron_end"])
            except Exception:
                missing_counts["bad_int_core_fields"] += 1
                continue

            if args.site_type == "acceptor":
                ok = (strand == "+" and pos == ie0 - 2) or (strand == "-" and pos == is0)
            else:  # donor
                ok = (strand == "+" and pos == is0) or (strand == "-" and pos == ie0 - 2)

            coord_total_by_strand[strand] += 1
            if not ok:
                coord_bad_by_strand[strand] += 1

            # ----- checks 2/3: dinucleotide purity for labels 1..5 (strand-normalized)
            for lid, col in LABEL_COLS.items():
                v = r.get(col, ".")
                if v == "." or pd.isna(v):
                    missing_counts[f"missing_L{lid}"] += 1
                    continue
                try:
                    p = int(v)
                except Exception:
                    missing_counts[f"nonint_L{lid}"] += 1
                    continue

                d = fetch_2bp(genome, r["chrom"], p)
                if strand == "-":
                    d = revcomp(d)
                dinuc_counts[lid][d] += 1

            total_rows += 1

    # ---- report + pass/fail logic
    any_fail = False

    print(f"\nvalidated site_type={args.site_type} expected={expected}")
    print(f"rows processed: {total_rows}\n")

    # check 1 report
    print("check 1) coordinate sanity (label1 vs intron boundary):")
    for s in ["+", "-"]:
        tot = coord_total_by_strand.get(s, 0)
        bad = coord_bad_by_strand.get(s, 0)
        frac = (bad / tot) if tot else 0.0
        print(f"  strand {s}: total={tot} bad={bad} bad_frac={frac:.3f}")
        if tot > 0 and bad > 0:
            any_fail = True
    print("")

    # checks 2/3 report
    print("check 2/3) dinucleotide purity (strand-normalized):")
    for lid in sorted(dinuc_counts):
        c = dinuc_counts[lid]
        tot = sum(c.values()) or 1
        frac = c.get(expected, 0) / tot
        print(f"  L{lid} {LABEL_NAMES[lid]}: n={tot}")
        print(f"    top: {fmt_top(c, k=8)}")
        print(f"    expected {expected}: {c.get(expected, 0)} ({frac:.1%})")

        if frac < args.motif_min_frac:
            print(f"    [FAIL] expected fraction {frac:.3f} < motif_min_frac {args.motif_min_frac:.3f}")
            any_fail = True
        print("")

    # check 4 report: L1 distance vs label1
    print("check 4) distribution match vs label1 (L1 distance):")
    base = dinuc_counts[1]
    base_tot = sum(base.values()) or 1
    base_freq = {k: v / base_tot for k, v in base.items()}

    for lid in sorted(dinuc_counts):
        c = dinuc_counts[lid]
        tot = sum(c.values()) or 1
        freq = {k: v / tot for k, v in c.items()}
        d = l1_distance(base_freq, freq)
        print(f"  L{lid} vs L1: {d:.4f}")
        if lid != 1 and d > args.max_l1_distance:
            print(f"    [FAIL] L1 distance {d:.4f} > max_l1_distance {args.max_l1_distance:.4f}")
            any_fail = True
    print("")

    if missing_counts:
        print("missing/malformed fields summary:")
        for k, v in sorted(missing_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")
        print("")

    if any_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
# python src/evaluation/check_splice_labels.py \