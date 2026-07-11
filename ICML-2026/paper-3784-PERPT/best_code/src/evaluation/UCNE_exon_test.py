import pyBigWig
import pandas as pd
import numpy as np
import argparse

def extract_mean_conservation(chrom, start, end, bw):
    try:
        length = end - start
        vals = np.zeros(length, dtype=np.float32)

        intervals = bw.intervals(chrom, start, end)
        if intervals:
            for istart, iend, val in intervals:
                # Clip intervals to fit within the array bounds
                rel_start = max(istart - start, 0)
                rel_end = min(iend - start, length)
                vals[rel_start:rel_end] = val  # fill known values
        return np.mean(vals)
    except Exception as e:
        print(f"Error with region {chrom}:{start}-{end} — {e}")
        return np.nan


def compute_avg_conservation(bed_path, bw):
    df = pd.read_csv(bed_path, sep='\t', header=None, names=['chrom', 'start', 'end'] + [f'col{i}' for i in range(4, 10)])
    df['mean_cons'] = df.apply(lambda row: extract_mean_conservation(row['chrom'], int(row['start']), int(row['end']), bw), axis=1)
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare UCNE vs Exon conservation")
    parser.add_argument('--ucne_bed', type=str, default ='/home/mica/gamba/data_processing/data/conserved_elements/filteredunseen_hg38UCNE_coordinates.bed', help='First BED file')
    parser.add_argument('--exon_bed', type=str, default='/home/mica/gamba/data_processing/data/UCSC coordinates/unseen_exons_chr2_chr22_chr16_chr3.bed', help='Second BED file (optional)')
    
    parser.add_argument('--big_wig', type=str, default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig', help='Path to the bigWig file')
    

    args = parser.parse_args()

    bw = pyBigWig.open(args.big_wig)

    print("Processing UCNEs...")
    ucne_df = compute_avg_conservation(args.ucne_bed, bw)
    print("Processing Exons...")
    exon_df = compute_avg_conservation(args.exon_bed, bw)

    ucne_cons = ucne_df['mean_cons'].dropna()
    exon_cons = exon_df['mean_cons'].dropna()

    print(f"\nUCNEs:   Mean = {ucne_cons.mean():.3f}, Std = {ucne_cons.std():.3f}, N = {len(ucne_cons)}")
    print(f"Exons:   Mean = {exon_cons.mean():.3f}, Std = {exon_cons.std():.3f}, N = {len(exon_cons)}")


if __name__ == "__main__":
    main()
