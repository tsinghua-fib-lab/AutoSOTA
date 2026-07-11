#!/usr/bin/env python3
# make_common_regions.py
import argparse
from pathlib import Path

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters",
]

ROLE_FOLDERS = {
    "roi": lambda c: c,
    "upstream": lambda c: f"{c}_upstream",
    "random": lambda c: f"{c}_random",
    "random-noannot": lambda c: f"{c}_random-noannot",
}

def iter_beds(d: Path):
    return sorted(d.glob("chr*.bed")) if d.exists() else []

def read_bed(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            chrom, start, end, name, score, strand, pair_id = parts[:7]
            rows.append((chrom, start, end, name, score, strand, str(pair_id), line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions_root", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--categories", nargs="+", default=CATEGORY_ORDER)
    ap.add_argument("--chromosomes", nargs="+", default=None, help="optional list like chr1 chr2 ... to filter")
    args = ap.parse_args()

    regions_root = Path(args.regions_root)
    out_root = Path(args.out_root)
    chrom_set = set(args.chromosomes) if args.chromosomes else None

    for cat in args.categories:
        # load all rows per role, grouped by chrom
        per_role = {}
        per_role_ids = {}

        for role, fn in ROLE_FOLDERS.items():
            d = regions_root / fn(cat)
            role_rows = {}  # chrom -> list[(pair_id, line)]
            ids = set()
            for bf in iter_beds(d):
                chrom = bf.stem
                if chrom_set and chrom not in chrom_set:
                    continue
                rows = read_bed(bf)
                if not rows:
                    continue
                role_rows.setdefault(chrom, [])
                for (chrom, start, end, name, score, strand, pid, raw) in rows:
                    if chrom_set and chrom not in chrom_set:
                        continue
                    ids.add(pid)
                    role_rows[chrom].append((pid, raw))
            per_role[role] = role_rows
            per_role_ids[role] = ids

        # intersection across all 4 roles
        if any(len(per_role_ids[r]) == 0 for r in ROLE_FOLDERS.keys()):
            print(f"[skip] {cat}: missing at least one role")
            continue

        common = set.intersection(*(per_role_ids[r] for r in ROLE_FOLDERS.keys()))
        print(f"[{cat}] common pair_ids across 4 roles: {len(common)}")

        # write filtered beds per role, per chrom
        for role, fn in ROLE_FOLDERS.items():
            out_dir = out_root / fn(cat)
            out_dir.mkdir(parents=True, exist_ok=True)
            for chrom, lst in per_role[role].items():
                out_path = out_dir / f"{chrom}.bed"
                with out_path.open("w") as w:
                    for pid, raw in lst:
                        if pid in common:
                            w.write(raw if raw.endswith("\n") else raw + "\n")

if __name__ == "__main__":
    main()


# python /home/mica/gamba/src/evaluation/generate_common_eval_set.py \
#   --regions_root /home/mica/gamba/data_processing/data/regions \
#   --out_root     /home/mica/gamba/data_processing/data/regions_common \
#   --chromosomes  chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX
