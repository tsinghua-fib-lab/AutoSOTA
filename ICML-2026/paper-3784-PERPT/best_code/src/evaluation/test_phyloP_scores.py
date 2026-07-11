import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyBigWig
from typing import List, Tuple

def load_bed_file(bed_file: str) -> pd.DataFrame:
    """Load BED file into DataFrame."""
    return pd.read_csv(bed_file, sep='\t', header=None, 
                      names=['chrom', 'start', 'end', 'name', 'score'])

def generate_random_regions(n_regions: int, chrom_sizes_file: str, 
                          region_size: int) -> pd.DataFrame:
    """Generate random genomic regions of specified size."""
    chrom_sizes = pd.read_csv(chrom_sizes_file, sep='\t', header=None, 
                             names=['chrom', 'size'])
    valid_chroms = [f'chr{i}' for i in range(1,23)] + ['chrX']
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(valid_chroms)]
    
    random_regions = []
    while len(random_regions) < n_regions:
        chrom = np.random.choice(valid_chroms)
        chrom_size = chrom_sizes[chrom_sizes['chrom'] == chrom]['size'].iloc[0]
        
        start = np.random.randint(0, chrom_size - region_size)
        end = start + region_size
        
        if start >= 0 and end <= chrom_size:
            random_regions.append([chrom, start, end])
    
    return pd.DataFrame(random_regions, columns=['chrom', 'start', 'end'])

def get_phylop_scores(regions: pd.DataFrame, bigwig_file: str) -> List[np.ndarray]:
    """Get phyloP scores for regions from bigWig file."""
    bw = pyBigWig.open(bigwig_file)
    scores = []
    
    for _, row in regions.iterrows():
        try:
            vals = bw.values(row['chrom'], row['start'], row['end'])
            scores.append(np.array([x if x is not None else np.nan for x in vals]))
        except Exception as e:
            print(f"Error processing {row['chrom']}:{row['start']}-{row['end']}: {str(e)}")
            continue
            
    bw.close()
    return scores

def plot_phylop_comparison(target_scores: List[np.ndarray], 
                          random_scores: List[np.ndarray],
                          output_path: str):
    """Plot comparison of phyloP scores between target and random regions."""
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics
    target_means = [np.nanmean(scores) for scores in target_scores]
    random_means = [np.nanmean(scores) for scores in random_scores]
    
    # Plot 1: Distribution of mean scores
    plt.subplot(2, 1, 1)
    plt.hist(target_means, bins=30, alpha=0.5, label='Target Regions', 
             color='blue', density=True)
    plt.hist(random_means, bins=30, alpha=0.5, label='Random Regions', 
             color='orange', density=True)
    plt.xlabel('Mean phyloP Score')
    plt.ylabel('Density')
    plt.title('Distribution of Mean phyloP Scores')
    plt.legend()
    
    # Add statistics as text
    stats_text = (
        f'Target: {np.mean(target_means):.3f}±{np.std(target_means):.3f}\n'
        f'Random: {np.mean(random_means):.3f}±{np.std(random_means):.3f}'
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Average profile
    plt.subplot(2, 1, 2)
    max_len = max(max(len(s) for s in target_scores), 
                 max(len(s) for s in random_scores))
    
    # Pad arrays to same length for averaging
    target_aligned = np.array([np.pad(s, (0, max_len - len(s)), 
                             constant_values=np.nan) for s in target_scores])
    random_aligned = np.array([np.pad(s, (0, max_len - len(s)), 
                             constant_values=np.nan) for s in random_scores])
    
    # Calculate means and confidence intervals
    target_mean = np.nanmean(target_aligned, axis=0)
    target_std = np.nanstd(target_aligned, axis=0)
    random_mean = np.nanmean(random_aligned, axis=0)
    random_std = np.nanstd(random_aligned, axis=0)
    
    x = np.arange(max_len)
    plt.plot(x, target_mean, label='Target Regions', color='blue')
    plt.fill_between(x, target_mean - target_std, target_mean + target_std,
                    color='blue', alpha=0.2)
    plt.plot(x, random_mean, label='Random Regions', color='orange')
    plt.fill_between(x, random_mean - random_std, random_mean + random_std,
                    color='orange', alpha=0.2)
    
    plt.xlabel('Position in Region')
    plt.ylabel('Average phyloP Score')
    plt.title('Average phyloP Score Profile')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze phyloP scores")
    parser.add_argument('--bed_file', default='/home/mica/gamba/data_processing/data/conserved_elements/hg38_UCNE_coordinates.bed', 
                       help='Path to target regions BED file')
    parser.add_argument('--chrom_sizes', default='/home/mica/gamba/data_processing/data/240-mammalian/hg38.chrom.sizes',
                       help='Path to chromosome sizes file')
    parser.add_argument('--bigwig', default='/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig',
                       help='Path to phyloP scores bigWig file')
    parser.add_argument('--output',
                       default='/home/mica/gamba/data_processing/data/conserved_elements/phyloP_plot.png')
    
    args = parser.parse_args()
    
    # Load target regions
    target_regions = load_bed_file(args.bed_file)
    avg_size = int(np.mean(target_regions['end'] - target_regions['start']))
    
    # Generate random regions of similar size
    random_regions = generate_random_regions(
        len(target_regions), args.chrom_sizes, avg_size)
    
    # Get phyloP scores
    print("Getting phyloP scores for target regions...")
    target_scores = get_phylop_scores(target_regions, args.bigwig)
    print("Getting phyloP scores for random regions...")
    random_scores = get_phylop_scores(random_regions, args.bigwig)
    
    # Plot comparison
    print("Creating plots...")
    plot_phylop_comparison(target_scores, random_scores, args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()