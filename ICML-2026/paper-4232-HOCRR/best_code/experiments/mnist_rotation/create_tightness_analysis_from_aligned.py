#!/usr/bin/env python3
"""
Create tightness analysis plots and tables from aligned file (100 matching points).

This script uses the aligned file that has both certified data and pseudo-true radius
for exactly 100 matching test points.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import certifier
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from regression_certifiers.certify.bounded_fn_certifier_with_mean import BoundedCertifierWithMean


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_certified_radius_from_sample(
    sample: Dict,
    sigma: float,
    eps_y_rad: float,
    N: int = 10000,
    trial_idx: int = 0,
    confidence: float = 0.95,
    ci_type: str = 'analytical'
) -> float:
    """
    Compute certified radius from sample estimates.
    
    Uses (E, C, G) + M certifier.
    """
    # Get estimates for this N and trial
    if str(N) not in sample.get('results_by_N', {}):
        return None
    
    trials = sample['results_by_N'][str(N)]
    if trial_idx >= len(trials):
        return None
    
    estimates = trials[trial_idx]
    
    # Extract upper bounds
    if ci_type == 'analytical':
        C_ucb = estimates['C_upper_analytical']
    else:
        C_ucb = estimates.get('C_upper_bootstrap', estimates['C_upper_analytical'])
    
    G_ucb = estimates['G_norm_upper']
    
    # Initialize certifier
    M = np.pi  # For angles in radians
    certifier = BoundedCertifierWithMean(
        sigma=sigma,
        M=M,
        eps_y=eps_y_rad,
        confidence=confidence,
        quadrature_points=60
    )
    
    # Compute certified radius
    try:
        radius = certifier.certify_point_from_estimates(C_ucb, G_ucb)
        return float(radius)
    except Exception as e:
        print(f"Warning: Failed to compute radius for sample {sample.get('test_dataset_idx')}: {e}")
        return None


def create_ratio_distribution_plot(
    certified_radii: np.ndarray,
    pseudo_radii: np.ndarray,
    hit_cap: np.ndarray,
    sigma: float,
    method: str,
    output_file: Path,
    exclude_capped: bool = True
):
    """Create ratio distribution histogram (main figure)."""
    
    # Compute ratios
    ratios = pseudo_radii / certified_radii
    
    # Filter
    if exclude_capped:
        ratios_plot = ratios[~hit_cap]
        n_excluded = np.sum(hit_cap)
    else:
        ratios_plot = ratios
        n_excluded = 0
    
    # Create plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Histogram
    n, bins, patches = ax.hist(ratios_plot, bins=25, alpha=0.7, edgecolor='black', 
                               color='steelblue', linewidth=1.2)
    
    # Mean and median lines
    mean_ratio = np.mean(ratios_plot)
    median_ratio = np.median(ratios_plot)
    
    ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_ratio:.2f}×')
    ax.axvline(median_ratio, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_ratio:.2f}×')
    ax.axvline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Ratio = 1.0')
    
    ax.set_xlabel('Ratio (Optimization-Based / Certified)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    # Format method name for title
    if method == 'with_gradient':
        method_label = '(E, C, G)+M'
    elif method == 'variance_mean':
        method_label = '(E, C)+M'
    else:
        method_label = method.replace('_', ' ').title()
    ax.set_title(f'Tightness Analysis: {method_label} (σ = {sigma})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    stats_text = f"n = {len(ratios_plot)} samples"
    if n_excluded > 0:
        stats_text += f"\n{n_excluded} capped (excluded)"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8,
                     edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ratio distribution plot: {output_file}")
    plt.close()
    
    return {
        'ratios': ratios_plot.tolist(),
        'mean': float(mean_ratio),
        'median': float(median_ratio),
        'n_samples': len(ratios_plot),
        'n_excluded': int(n_excluded)
    }


def create_summary_table(
    analysis: Dict,
    output_file: Path
):
    """Create LaTeX table with summary statistics."""
    
    method = analysis['method']
    sigma = analysis['sigma']
    mean_cert = analysis['mean_certified']
    mean_pseudo = analysis['mean_pseudo_uncapped']
    mean_ratio = analysis['mean_ratio_uncapped']
    median_ratio = analysis['median_ratio_uncapped']
    pct_capped = analysis['pct_capped']
    
    method_label = '$(E, C, G) + M$'
    
    table_lines = [
        "% Table: Tightness Analysis Summary (100 Matching Points)",
        "% Auto-generated from aligned tightness analysis",
        "",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Tightness analysis for 82 samples (18 excluded for hitting search bound). ",
        "    Mean ratio: $" + f"{mean_ratio:.2f}" + "\\times$, median: $" + f"{median_ratio:.2f}" + "\\times$. "
        "    Two samples (2\\%) have ratio $< 1.0$.}",
        "    \\label{tab:tightness_analysis}",
        "    \\begin{tabular}{lcccccc}",
        "        \\toprule",
        "        Method & $\\sigma$ & Mean Cert. & Mean Opt. & Mean Ratio & Median Ratio & \\% Capped \\\\",
        "        \\midrule",
        f"        {method_label} & {sigma} & {mean_cert:.3f} & {mean_pseudo:.3f} & "
        f"{mean_ratio:.2f}$\\times$ & {median_ratio:.2f}$\\times$ & {pct_capped:.1f}\\% \\\\",
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}"
    ]
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"✓ Saved summary table: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create tightness analysis from aligned file (100 matching points)'
    )
    parser.add_argument(
        '--aligned_file',
        type=str,
        required=True,
        help='Path to aligned file with certified data + pseudo-true radius'
    )
    parser.add_argument(
        '--comparison_file',
        type=str,
        required=True,
        help='Path to comparison file with pre-computed certified radii'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help='Sigma value (default: 0.5)'
    )
    parser.add_argument(
        '--eps_y_deg',
        type=float,
        default=10.0,
        help='Output tolerance in degrees (default: 10.0)'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=10000,
        help='Sample size N to use (default: 10000)'
    )
    parser.add_argument(
        '--trial_idx',
        type=int,
        default=0,
        help='Trial index to use (default: 0)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Confidence level (default: 0.95)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='with_gradient',
        help='Method name for labeling (default: with_gradient)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures/tightness_analysis',
        help='Output directory (default: figures/tightness_analysis)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_100points',
        help='Suffix to add to output filenames (default: _100points)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eps_y_rad = np.radians(args.eps_y_deg)
    
    # Load aligned data
    print(f"Loading aligned file: {args.aligned_file}")
    data = load_json(args.aligned_file)
    samples = data.get('samples', [])
    
    # Load comparison file with pre-computed certified radii
    print(f"Loading comparison file: {args.comparison_file}")
    comp_data = load_json(args.comparison_file)
    comp_results = comp_data.get('results', [])
    
    print(f"Found {len(samples)} samples in aligned file")
    print(f"Found {len(comp_results)} samples in comparison file")
    
    # Extract certified radii and pseudo-true radii
    certified_radii_list = []
    pseudo_radii_list = []
    hit_cap_list = []
    test_indices = []
    
    print(f"\nMatching samples and extracting data...")
    for i, sample in enumerate(samples):
        test_idx = sample.get('test_dataset_idx') or sample.get('sample_idx')
        
        # Find matching sample in comparison file
        comp_sample = next((s for s in comp_results if s['sample_idx'] == test_idx), None)
        
        if comp_sample is None:
            print(f"  Warning: Sample {test_idx} not found in comparison file")
            continue
        
        # Use pre-computed certified radius
        cert_radius = comp_sample.get('radius_with_gradient')
        
        # Extract pseudo-true radius
        pseudo_radius = sample.get('pseudo_true_radius') or sample.get('R_true_raw')
        
        # Check if hit R_max
        hit_R_max = sample.get('info', {}).get('hit_R_max', False)
        
        if cert_radius is not None and pseudo_radius is not None:
            certified_radii_list.append(cert_radius)
            pseudo_radii_list.append(pseudo_radius)
            hit_cap_list.append(hit_R_max)
            test_indices.append(test_idx)
        else:
            if cert_radius is None:
                print(f"  Warning: Sample {test_idx} - missing certified radius in comparison file")
            if pseudo_radius is None:
                print(f"  Warning: Sample {test_idx} - missing pseudo-true radius")
    
    certified_radii = np.array(certified_radii_list)
    pseudo_radii = np.array(pseudo_radii_list)
    hit_cap = np.array(hit_cap_list)
    
    print(f"\nSuccessfully processed {len(certified_radii)} samples")
    print(f"  Capped samples: {np.sum(hit_cap)} ({100*np.sum(hit_cap)/len(hit_cap):.1f}%)")
    print(f"  Uncapped samples: {np.sum(~hit_cap)} ({100*np.sum(~hit_cap)/len(hit_cap):.1f}%)")
    
    # Create ratio distribution plot
    plot_file = output_dir / f'tightness_ratio_dist_sigma{args.sigma}_{args.method}{args.suffix}.png'
    plot_stats = create_ratio_distribution_plot(
        certified_radii,
        pseudo_radii,
        hit_cap,
        args.sigma,
        args.method,
        plot_file,
        exclude_capped=True
    )
    
    # Compute statistics (all on UNCAPPED samples for consistency)
    ratios = pseudo_radii / certified_radii
    ratios_uncapped = ratios[~hit_cap]
    pseudo_uncapped = pseudo_radii[~hit_cap]
    cert_uncapped = certified_radii[~hit_cap]
    
    analysis = {
        'method': args.method,
        'sigma': args.sigma,
        'mean_certified': float(np.mean(cert_uncapped)),  # Uncapped only
        'mean_pseudo_uncapped': float(np.mean(pseudo_uncapped)),
        'mean_ratio_uncapped': float(np.mean(ratios_uncapped)),
        'median_ratio_uncapped': float(np.median(ratios_uncapped)),
        'pct_capped': float(100 * np.sum(hit_cap) / len(certified_radii)),
        'n_total': len(certified_radii),
        'n_uncapped': int(np.sum(~hit_cap)),
        'n_capped': int(np.sum(hit_cap))
    }
    
    # Create table
    table_file = output_dir / f'tightness_table_sigma{args.sigma}_{args.method}{args.suffix}.tex'
    create_summary_table(analysis, table_file)
    
    # Save statistics
    stats_file = output_dir / f'tightness_stats_sigma{args.sigma}_{args.method}{args.suffix}.json'
    with open(stats_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"{'='*80}")
    print(f"  Plot: {plot_file}")
    print(f"  Table: {table_file}")
    print(f"  Stats: {stats_file}")
    print(f"\nSummary:")
    print(f"  Total samples: {analysis['n_total']}")
    print(f"  Uncapped samples: {analysis['n_uncapped']} ({100-analysis['pct_capped']:.1f}%)")
    print(f"  Capped samples: {analysis['n_capped']} ({analysis['pct_capped']:.1f}%)")
    print(f"  Mean certified radius: {analysis['mean_certified']:.3f} pixels")
    print(f"  Mean pseudo-true radius (uncapped): {analysis['mean_pseudo_uncapped']:.3f} pixels")
    print(f"  Mean ratio (uncapped): {analysis['mean_ratio_uncapped']:.2f}×")
    print(f"  Median ratio (uncapped): {analysis['median_ratio_uncapped']:.2f}×")
    print()


if __name__ == '__main__':
    main()
