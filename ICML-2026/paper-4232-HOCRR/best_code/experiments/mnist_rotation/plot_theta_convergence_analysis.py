#!/usr/bin/env python3
"""
Plotting Script for θ = ||G||² Convergence Analysis

This script plots results from the updated mnist_rotation_convergence.py that focuses
on θ = ||G||² (squared gradient norm) instead of ||G|| (gradient norm).

Generates comprehensive plots showing:
- Part 1: Convergence of C (variance) and θ (squared gradient norm)
- Part 1: Coverage validation for C and θ CIs
- Part 2: Convergence of certified radius

Usage:
    python experiments/mnist_rotation/plot_theta_convergence_analysis.py <json_file>
    
Example:
    python experiments/mnist_rotation/plot_theta_convergence_analysis.py mnist_rotation_convergence_img0_20251101_123456.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def extract_part1_data(part1):
    """Extract convergence and coverage data from Part 1 results"""
    N_values = part1['N_values']
    C_true = part1['ground_truth']['C']
    theta_true = part1['ground_truth']['theta']
    n_trials = part1['n_trials']
    sigma = part1.get('sigma', None)  # Extract sigma from part1
    
    # Initialize storage
    C_means = []
    C_stds = []
    theta_means = []
    theta_stds = []
    C_coverage = []
    theta_coverage = []
    
    for N in N_values:
        # Handle both string and int keys
        key = str(N) if str(N) in part1['results_by_N'] else N
        trials = part1['results_by_N'][key]
        
        # Extract estimates
        C_hats = [t['C_hat'] for t in trials]
        theta_hats = [t['theta_hat'] for t in trials]
        
        # Compute means and standard deviations
        C_means.append(np.mean(C_hats))
        C_stds.append(np.std(C_hats, ddof=1))
        theta_means.append(np.mean(theta_hats))
        theta_stds.append(np.std(theta_hats, ddof=1))
        
        # Compute coverage
        C_cov = np.mean([t['C_lower'] <= C_true <= t['C_upper'] for t in trials])
        theta_cov = np.mean([t['theta_lower'] <= theta_true <= t['theta_upper'] for t in trials])
        C_coverage.append(C_cov)
        theta_coverage.append(theta_cov)
    
    return {
        'N_values': N_values,
        'C_true': C_true,
        'theta_true': theta_true,
        'C_means': C_means,
        'C_stds': C_stds,
        'theta_means': theta_means,
        'theta_stds': theta_stds,
        'C_coverage': C_coverage,
        'theta_coverage': theta_coverage,
        'n_trials': n_trials,
        'sigma': sigma
    }


def compute_variance_only_radius(C, eps_y, sigma):
    """
    Compute variance-only certified radius using the formula:
    R = σ√(log(1 + ε²/C))
    
    Args:
        C: Variance (or upper confidence bound)
        eps_y: Output tolerance
        sigma: Noise standard deviation
        
    Returns:
        Certified radius (variance-only)
    """
    if C <= 0:
        return 0.0
    return sigma * np.sqrt(np.log(1 + (eps_y**2) / C))


def extract_part2_data(part2, part1):
    """Extract radius convergence data from Part 2 results and compute variance-only radii"""
    N_values = part2['N_values']
    r_theoretical_full = part2['theoretical_radius']  # Variance + Gradient (theoretical)
    n_trials = part2['n_trials']
    
    # Extract sigma and eps_y from JSON
    sigma = part2.get('sigma', part1.get('sigma', None))
    eps_y = part2.get('eps_y', part1.get('eps_y', None))
    
    if sigma is None or eps_y is None:
        raise ValueError("sigma and eps_y must be present in JSON for variance-only radius calculation")
    
    # Get parameters from ground truth
    C_true = part1['ground_truth']['C']
    
    # Compute theoretical variance-only radius
    r_theoretical_variance_only = compute_variance_only_radius(C_true, eps_y, sigma)
    
    r_empirical_full_means = []
    r_empirical_full_stds = []
    r_empirical_variance_only_means = []
    r_empirical_variance_only_stds = []
    
    for N in N_values:
        # Handle both string and int keys
        key = str(N) if str(N) in part2['results_by_N'] else N
        trials = part2['results_by_N'][key]
        
        # Empirical radius (Variance + Gradient)
        r_empiricals_full = [t['r_empirical'] for t in trials]
        r_empirical_full_means.append(np.mean(r_empiricals_full))
        r_empirical_full_stds.append(np.std(r_empiricals_full, ddof=1))
        
        # Get corresponding Part 1 trials for C_upper (for variance-only empirical)
        part1_key = str(N) if str(N) in part1['results_by_N'] else N
        part1_trials = part1['results_by_N'][part1_key]
        
        # Compute variance-only empirical radius for each trial
        r_variance_only_trials = []
        for trial in part1_trials:
            C_upper = trial['C_upper']
            r_vo = compute_variance_only_radius(C_upper, eps_y, sigma)
            r_variance_only_trials.append(r_vo)
        
        r_empirical_variance_only_means.append(np.mean(r_variance_only_trials))
        r_empirical_variance_only_stds.append(np.std(r_variance_only_trials, ddof=1))
    
    return {
        'N_values': N_values,
        'sigma': sigma,
        'eps_y': eps_y,
        'r_theoretical_full': r_theoretical_full,
        'r_theoretical_variance_only': r_theoretical_variance_only,
        'r_empirical_full_means': r_empirical_full_means,
        'r_empirical_full_stds': r_empirical_full_stds,
        'r_empirical_variance_only_means': r_empirical_variance_only_means,
        'r_empirical_variance_only_stds': r_empirical_variance_only_stds,
        'n_trials': n_trials,
        'C_true': C_true
    }


def plot_comprehensive_analysis(part1_data, part2_data, output_prefix='theta_convergence'):
    """
    Create comprehensive convergence analysis plots.
    
    Creates a 2×2 grid:
    - Top row: Convergence of C and θ to ground truth
    - Bottom left: Coverage validation for C and θ CIs
    - Bottom right: Certified radius convergence
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    N_values = part1_data['N_values']
    n_trials = part1_data['n_trials']
    sigma = part1_data.get('sigma', None)  # Get sigma from part1_data
    
    # Format sigma for display (use 2 decimal places, remove trailing zeros if needed)
    if sigma is not None:
        sigma_str = f"{sigma:.2f}".rstrip('0').rstrip('.')
    else:
        sigma_str = "N/A"
    
    # ===== Plot 1: Variance (C) Convergence =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot mean ± SEM (standard error of the mean)
    sem_C = [std / np.sqrt(n_trials) for std in part1_data['C_stds']]
    ax1.errorbar(N_values, part1_data['C_means'], yerr=[1.96*s for s in sem_C],
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label='C_hat (mean ± 95% CI)', color='#2E86AB')
    ax1.axhline(part1_data['C_true'], color='#A23B72', linestyle='--', 
               linewidth=2.5, label=f'C_true = {part1_data["C_true"]:.6f}')
    
    ax1.set_xlabel('Sample Size N', fontweight='bold')
    ax1.set_ylabel('Variance C (rad²)', fontweight='bold')
    ax1.set_title(f'(a) Variance Convergence (σ = {sigma_str})', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Plot 2: Theta (θ = ||G||²) Convergence =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot mean ± SEM
    sem_theta = [std / np.sqrt(n_trials) for std in part1_data['theta_stds']]
    ax2.errorbar(N_values, part1_data['theta_means'], yerr=[1.96*s for s in sem_theta],
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label='θ_hat (mean ± 95% CI)', color='#F18F01')
    ax2.axhline(part1_data['theta_true'], color='#A23B72', linestyle='--',
               linewidth=2.5, label=f'θ_true = {part1_data["theta_true"]:.8f}')
    
    ax2.set_xlabel('Sample Size N', fontweight='bold')
    ax2.set_ylabel('θ = ||G||² (rad²/px²)', fontweight='bold')
    ax2.set_title(f'(b) Squared Gradient Norm Convergence (σ = {sigma_str})', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ===== Plot 3: Coverage Validation =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(N_values, part1_data['C_coverage'], 'o-', linewidth=2.5, markersize=10,
            label='C CI Coverage', color='#2E86AB')
    ax3.plot(N_values, part1_data['theta_coverage'], 's-', linewidth=2.5, markersize=10,
            label='θ CI Coverage', color='#F18F01')
    ax3.axhline(0.95, color='#A23B72', linestyle='--', linewidth=2.5,
               label='Nominal 95%')
    
    ax3.set_xlabel('Sample Size N', fontweight='bold')
    ax3.set_ylabel('CI Coverage', fontweight='bold')
    ax3.set_title(f'(c) Confidence Interval Coverage Validation (σ = {sigma_str})', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_ylim([0.5, 1.05])
    ax3.legend(loc='best', framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal lines for reference
    ax3.axhline(0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax3.axhline(1.00, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # ===== Plot 4: Certified Radius Convergence (4 curves comparison) =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Theoretical radii (horizontal lines)
    ax4.axhline(part2_data['r_theoretical_variance_only'], color='#FF6B6B', linestyle='--',
               linewidth=2.5, label=f'Theoretical (Variance-Only)', zorder=1)
    ax4.axhline(part2_data['r_theoretical_full'], color='#4ECDC4', linestyle='--',
               linewidth=2.5, label=f'Theoretical (Variance+Gradient)', zorder=1)
    
    # Empirical radii (converging curves with error bars)
    sem_r_full = [std / np.sqrt(part2_data['n_trials']) for std in part2_data['r_empirical_full_stds']]
    ax4.errorbar(part2_data['N_values'], part2_data['r_empirical_full_means'], 
                yerr=[1.96*s for s in sem_r_full],
                fmt='s-', capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='Empirical (Variance+Gradient)', color='#4ECDC4', alpha=0.8, zorder=3)
    
    sem_r_vo = [std / np.sqrt(part2_data['n_trials']) for std in part2_data['r_empirical_variance_only_stds']]
    ax4.errorbar(part2_data['N_values'], part2_data['r_empirical_variance_only_means'], 
                yerr=[1.96*s for s in sem_r_vo],
                fmt='o-', capsize=4, capthick=1.5, linewidth=2, markersize=7,
                label='Empirical (Variance-Only)', color='#FF6B6B', alpha=0.8, zorder=2)
    
    ax4.set_xlabel('Sample Size N', fontweight='bold')
    ax4.set_ylabel('Certified Radius (px)', fontweight='bold')
    ax4.set_title(f'(d) Certified Radius Convergence Comparison (σ = {sigma_str})', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.legend(loc='best', framealpha=0.95, fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle(f'MNIST Rotation Convergence Analysis: θ = ||G||² (Squared Gradient Norm), σ = {sigma_str}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Format sigma for filename (replace dots with underscores)
    if sigma is not None:
        sigma_filename = f"sigma_{str(sigma).replace('.', '_')}"
        output_file = f'{output_prefix}_{sigma_filename}_comprehensive.png'
    else:
        output_file = f'{output_prefix}_comprehensive.png'
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    return fig


def plot_detailed_convergence(part1_data, part2_data, output_prefix='theta_convergence'):
    """
    Create detailed convergence plot showing individual trials.
    
    Shows one representative trial's CI shrinkage for C and θ.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    N_values = part1_data['N_values']
    C_true = part1_data['C_true']
    theta_true = part1_data['theta_true']
    sigma = part1_data.get('sigma', None)  # Get sigma from part1_data
    
    # Format sigma for display (use 2 decimal places, remove trailing zeros if needed)
    if sigma is not None:
        sigma_str = f"{sigma:.2f}".rstrip('0').rstrip('.')
    else:
        sigma_str = "N/A"
    
    # Select middle trial as representative
    rep_trial_idx = len(part1_data['C_means']) // 2
    
    # Note: This would require storing individual trial CIs in the extraction function
    # For now, just plot the means
    
    # Plot 1: Variance with CI band
    ax1 = axes[0]
    ax1.plot(N_values, part1_data['C_means'], 'o-', linewidth=2, markersize=8, 
            label='C_hat', color='#2E86AB')
    ax1.fill_between(N_values, 
                     [m - 1.96*s/np.sqrt(part1_data['n_trials']) for m, s in zip(part1_data['C_means'], part1_data['C_stds'])],
                     [m + 1.96*s/np.sqrt(part1_data['n_trials']) for m, s in zip(part1_data['C_means'], part1_data['C_stds'])],
                     alpha=0.3, color='#2E86AB', label='95% CI band')
    ax1.axhline(C_true, color='#A23B72', linestyle='--', linewidth=2, label='C_true')
    ax1.set_xlabel('Sample Size N', fontweight='bold')
    ax1.set_ylabel('Variance C (rad²)', fontweight='bold')
    ax1.set_title(f'Variance Convergence with CI Band (σ = {sigma_str})', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Theta with CI band
    ax2 = axes[1]
    ax2.plot(N_values, part1_data['theta_means'], 's-', linewidth=2, markersize=8,
            label='θ_hat', color='#F18F01')
    ax2.fill_between(N_values,
                     [m - 1.96*s/np.sqrt(part1_data['n_trials']) for m, s in zip(part1_data['theta_means'], part1_data['theta_stds'])],
                     [m + 1.96*s/np.sqrt(part1_data['n_trials']) for m, s in zip(part1_data['theta_means'], part1_data['theta_stds'])],
                     alpha=0.3, color='#F18F01', label='95% CI band')
    ax2.axhline(theta_true, color='#A23B72', linestyle='--', linewidth=2, label='θ_true')
    ax2.set_xlabel('Sample Size N', fontweight='bold')
    ax2.set_ylabel('θ = ||G||² (rad²/px²)', fontweight='bold')
    ax2.set_title('Squared Gradient Norm Convergence with CI Band', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed_convergence.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_detailed_convergence.png")
    
    return fig


def print_summary(part1_data, part2_data):
    """Print numerical summary of results"""
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nGround Truth Values:")
    print(f"  C (variance)           : {part1_data['C_true']:.6f} rad²")
    print(f"  θ = ||G||² (grad norm²): {part1_data['theta_true']:.8f} rad²/px²")
    print(f"  ||G|| = √θ             : {np.sqrt(part1_data['theta_true']):.6f} rad/px")
    print(f"  σ (noise std)          : {part2_data['sigma']:.2f}")
    print(f"  ε_y (tolerance)        : {part2_data['eps_y']:.4f} rad")
    
    print("\nTheoretical Certified Radii:")
    print(f"  Variance-Only          : {part2_data['r_theoretical_variance_only']:.6f} px")
    print(f"  Variance + Gradient    : {part2_data['r_theoretical_full']:.6f} px")
    improvement = (part2_data['r_theoretical_full'] / part2_data['r_theoretical_variance_only'] - 1) * 100
    print(f"  Improvement from gradient: {improvement:+.2f}%")
    
    print("\nPart 1: Estimator Performance (at largest N)")
    largest_idx = -1
    N_large = part1_data['N_values'][largest_idx]
    C_mean_large = part1_data['C_means'][largest_idx]
    theta_mean_large = part1_data['theta_means'][largest_idx]
    C_cov_large = part1_data['C_coverage'][largest_idx]
    theta_cov_large = part1_data['theta_coverage'][largest_idx]
    
    C_bias = abs(C_mean_large - part1_data['C_true']) / part1_data['C_true'] * 100
    theta_bias = abs(theta_mean_large - part1_data['theta_true']) / part1_data['theta_true'] * 100
    
    print(f"  N = {N_large}")
    print(f"  C_hat    : {C_mean_large:.6f} (bias: {C_bias:.2f}%, coverage: {C_cov_large:.1%})")
    print(f"  θ_hat    : {theta_mean_large:.8f} (bias: {theta_bias:.2f}%, coverage: {theta_cov_large:.1%})")
    
    print("\nPart 2: Radius Performance (at largest N)")
    r_full_mean_large = part2_data['r_empirical_full_means'][largest_idx]
    r_vo_mean_large = part2_data['r_empirical_variance_only_means'][largest_idx]
    r_full_bias = abs(r_full_mean_large - part2_data['r_theoretical_full']) / part2_data['r_theoretical_full'] * 100
    r_vo_bias = abs(r_vo_mean_large - part2_data['r_theoretical_variance_only']) / part2_data['r_theoretical_variance_only'] * 100
    
    print(f"  Empirical (Variance+Gradient) : {r_full_mean_large:.6f} (bias: {r_full_bias:.2f}%)")
    print(f"  Theoretical (Variance+Gradient): {part2_data['r_theoretical_full']:.6f}")
    print(f"  Empirical (Variance-Only)     : {r_vo_mean_large:.6f} (bias: {r_vo_bias:.2f}%)")
    print(f"  Theoretical (Variance-Only)   : {part2_data['r_theoretical_variance_only']:.6f}")
    
    print("\nCoverage Across All N:")
    print(f"  {'N':<8} {'C Coverage':<12} {'θ Coverage':<12}")
    print("-" * 32)
    for i, N in enumerate(part1_data['N_values']):
        print(f"  {N:<8} {part1_data['C_coverage'][i]:<12.1%} {part1_data['theta_coverage'][i]:<12.1%}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot theta convergence analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/mnist_rotation/plot_theta_convergence_analysis.py mnist_rotation_convergence_img0_20251101_123456.json
  python experiments/mnist_rotation/plot_theta_convergence_analysis.py results.json --output-prefix my_analysis
        """
    )
    parser.add_argument('json_file', type=str,
                       help='Path to JSON results file from mnist_rotation_convergence.py')
    parser.add_argument('--output-prefix', type=str, default='theta_convergence',
                       help='Prefix for output plot files (default: theta_convergence)')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed convergence plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return
    
    print("="*80)
    print("THETA CONVERGENCE ANALYSIS PLOTTER")
    print("="*80)
    print(f"Loading: {args.json_file}")
    
    # Load data
    data = load_results(args.json_file)
    
    # Extract data
    part1_data = extract_part1_data(data['part1'])
    part2_data = extract_part2_data(data['part2'], data['part1'])
    
    print(f"✓ Loaded {part1_data['n_trials']} trials across {len(part1_data['N_values'])} sample sizes")
    
    # Print summary
    print_summary(part1_data, part2_data)
    
    # Create plots
    print("\nGenerating plots...")
    plot_comprehensive_analysis(part1_data, part2_data, args.output_prefix)
    
    if not args.no_detailed:
        plot_detailed_convergence(part1_data, part2_data, args.output_prefix)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {args.output_prefix}_comprehensive.png (2×2 grid: convergence + coverage + radius)")
    if not args.no_detailed:
        print(f"  - {args.output_prefix}_detailed_convergence.png (detailed C and θ convergence)")
    print("="*80)


if __name__ == '__main__':
    main()

