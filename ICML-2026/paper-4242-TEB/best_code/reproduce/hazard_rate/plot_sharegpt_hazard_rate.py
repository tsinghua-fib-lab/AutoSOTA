#!/usr/bin/env python3
"""
Plot hazard rate of ShareGPT output lengths to check if it's IFR (Increasing Failure Rate).

Hazard rate definition:
  h(t) = f(t) / S(t)
where:
  f(t) = probability density function (PDF)
  S(t) = survival function = P(X > t) = 1 - F(t)

If h(t) is monotonically increasing, the distribution is IFR.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import load_sharegpt_prompts


def compute_empirical_hazard_rate(data: np.ndarray, num_bins: int = 100):
    """
    Compute empirical hazard rate from data.

    Returns:
        t: bin centers (x-axis)
        hazard: hazard rate values
        survival: survival function values
        pdf: probability density function values
    """
    # Remove outliers (top 1%)
    data = data[data <= np.percentile(data, 99)]

    # Create histogram for PDF estimation
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # PDF: f(t)
    pdf = counts

    # CDF: F(t) = integral of f(t) from 0 to t
    cdf = np.cumsum(pdf) * bin_width

    # Survival function: S(t) = 1 - F(t) = P(X > t)
    survival = 1 - cdf

    # Avoid division by zero
    survival = np.maximum(survival, 1e-10)

    # Hazard rate: h(t) = f(t) / S(t)
    hazard = pdf / survival

    return bin_centers, hazard, survival, pdf


def compute_kaplan_meier_hazard(data: np.ndarray):
    """
    Compute hazard rate using Kaplan-Meier estimator (more robust).

    Returns:
        t: sorted unique values
        hazard: hazard rate at each point
        survival: survival probability
    """
    n = len(data)
    sorted_data = np.sort(data)

    # Get unique values and their counts
    unique_vals, counts = np.unique(sorted_data, return_counts=True)

    # Number at risk at each time point
    at_risk = n - np.concatenate([[0], np.cumsum(counts)[:-1]])

    # Hazard rate: h(t) = d(t) / n(t) where d(t) is events at t, n(t) is at risk
    hazard = counts / at_risk

    # Survival probability (Kaplan-Meier)
    survival = np.cumprod(1 - hazard)

    return unique_vals, hazard, survival


def plot_hazard_rate(output_lengths: np.ndarray, output_path: str = None):
    """Plot hazard rate and related distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of output lengths
    ax1 = axes[0, 0]
    ax1.hist(output_lengths, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Output Length (tokens)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Output Lengths')
    ax1.axvline(np.mean(output_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(output_lengths):.1f}')
    ax1.axvline(np.median(output_lengths), color='green', linestyle='--', label=f'Median: {np.median(output_lengths):.1f}')
    ax1.legend()

    # 2. Empirical CDF and Survival function
    ax2 = axes[0, 1]
    sorted_data = np.sort(output_lengths)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n
    survival = 1 - ecdf

    ax2.plot(sorted_data, ecdf, label='CDF F(t)', color='blue')
    ax2.plot(sorted_data, survival, label='Survival S(t) = 1 - F(t)', color='red')
    ax2.set_xlabel('Output Length (tokens)')
    ax2.set_ylabel('Probability')
    ax2.set_title('CDF and Survival Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Hazard rate (empirical binned method)
    ax3 = axes[1, 0]
    t, hazard, _, _ = compute_empirical_hazard_rate(output_lengths, num_bins=80)

    # Smooth the hazard rate
    hazard_smooth = gaussian_filter1d(hazard, sigma=2)

    ax3.plot(t, hazard, alpha=0.3, color='blue', label='Raw')
    ax3.plot(t, hazard_smooth, color='red', linewidth=2, label='Smoothed')
    ax3.set_xlabel('Output Length (tokens)')
    ax3.set_ylabel('Hazard Rate h(t)')
    ax3.set_title('Hazard Rate h(t) = f(t) / S(t)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add trend line to check if IFR
    # Fit a polynomial to check monotonicity
    valid_idx = np.isfinite(hazard_smooth) & (hazard_smooth > 0)
    if np.sum(valid_idx) > 10:
        t_valid = t[valid_idx]
        h_valid = hazard_smooth[valid_idx]

        # Linear fit
        slope, intercept = np.polyfit(t_valid, h_valid, 1)
        trend_line = slope * t_valid + intercept
        ax3.plot(t_valid, trend_line, '--', color='green', linewidth=2,
                 label=f'Linear trend (slope={slope:.2e})')
        ax3.legend()

        # Check if increasing
        if slope > 0:
            ax3.text(0.05, 0.95, 'Trend: INCREASING', transform=ax3.transAxes,
                    fontsize=12, verticalalignment='top', color='green', fontweight='bold')
        else:
            ax3.text(0.05, 0.95, 'Trend: DECREASING', transform=ax3.transAxes,
                    fontsize=12, verticalalignment='top', color='red', fontweight='bold')

    # 4. Cumulative hazard rate (for better visualization)
    ax4 = axes[1, 1]

    # Kaplan-Meier based hazard
    t_km, hazard_km, survival_km = compute_kaplan_meier_hazard(output_lengths)

    # Cumulative hazard H(t) = -log(S(t))
    cumulative_hazard = -np.log(np.maximum(survival_km, 1e-10))

    ax4.plot(t_km, cumulative_hazard, color='purple', alpha=0.5)
    # Smooth version
    window = min(100, len(cumulative_hazard) // 10)
    if window > 1:
        cumulative_hazard_smooth = np.convolve(cumulative_hazard,
                                                np.ones(window)/window, mode='valid')
        t_smooth = t_km[window//2:window//2 + len(cumulative_hazard_smooth)]
        ax4.plot(t_smooth, cumulative_hazard_smooth, color='red', linewidth=2, label='Smoothed')

    ax4.set_xlabel('Output Length (tokens)')
    ax4.set_ylabel('Cumulative Hazard H(t)')
    ax4.set_title('Cumulative Hazard H(t) = -log(S(t))')
    ax4.grid(True, alpha=0.3)

    # For IFR, cumulative hazard should be convex (second derivative > 0)
    ax4.text(0.05, 0.95, 'IFR if convex (curves upward)', transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()

    return fig


def analyze_ifr(output_lengths: np.ndarray):
    """Analyze if the distribution is IFR."""
    print("\n" + "="*60)
    print("IFR (Increasing Failure Rate) Analysis")
    print("="*60)

    # Basic statistics
    print(f"\nData Statistics:")
    print(f"  N samples: {len(output_lengths)}")
    print(f"  Mean: {np.mean(output_lengths):.2f}")
    print(f"  Median: {np.median(output_lengths):.2f}")
    print(f"  Std: {np.std(output_lengths):.2f}")
    print(f"  Min: {np.min(output_lengths)}")
    print(f"  Max: {np.max(output_lengths)}")
    print(f"  25th percentile: {np.percentile(output_lengths, 25):.2f}")
    print(f"  75th percentile: {np.percentile(output_lengths, 75):.2f}")
    print(f"  99th percentile: {np.percentile(output_lengths, 99):.2f}")

    # Coefficient of variation
    cv = np.std(output_lengths) / np.mean(output_lengths)
    print(f"\n  Coefficient of Variation (CV): {cv:.3f}")
    print(f"    (CV < 1: possibly IFR, CV > 1: possibly DFR)")

    # Compute hazard rate trend
    t, hazard, survival, pdf = compute_empirical_hazard_rate(output_lengths, num_bins=50)
    hazard_smooth = gaussian_filter1d(hazard, sigma=2)

    valid_idx = np.isfinite(hazard_smooth) & (hazard_smooth > 0)
    if np.sum(valid_idx) > 10:
        t_valid = t[valid_idx]
        h_valid = hazard_smooth[valid_idx]

        # Linear fit
        slope, intercept = np.polyfit(t_valid, h_valid, 1)

        # Check monotonicity
        diffs = np.diff(h_valid)
        pct_increasing = np.sum(diffs > 0) / len(diffs) * 100

        print(f"\nHazard Rate Analysis:")
        print(f"  Linear trend slope: {slope:.6f}")
        print(f"  Percentage of increasing segments: {pct_increasing:.1f}%")

        if slope > 0 and pct_increasing > 60:
            print(f"\n  >>> CONCLUSION: Distribution appears to be IFR <<<")
            print(f"      (hazard rate is generally increasing)")
        elif slope < 0 and pct_increasing < 40:
            print(f"\n  >>> CONCLUSION: Distribution appears to be DFR <<<")
            print(f"      (hazard rate is generally decreasing)")
        else:
            print(f"\n  >>> CONCLUSION: Distribution is neither clearly IFR nor DFR <<<")
            print(f"      (hazard rate shows mixed behavior)")

    # Test against common distributions
    print(f"\nDistribution Fit Tests:")

    # Test exponential (constant hazard rate)
    loc, scale = stats.expon.fit(output_lengths, floc=0)
    _, p_exp = stats.kstest(output_lengths, 'expon', args=(0, scale))
    print(f"  Exponential fit: scale={scale:.2f}, KS p-value={p_exp:.4f}")
    print(f"    (Exponential has CONSTANT hazard rate)")

    # Test gamma (IFR when shape > 1, DFR when shape < 1)
    alpha, loc, beta = stats.gamma.fit(output_lengths, floc=0)
    _, p_gamma = stats.kstest(output_lengths, 'gamma', args=(alpha, 0, beta))
    print(f"  Gamma fit: shape={alpha:.3f}, scale={beta:.2f}, KS p-value={p_gamma:.4f}")
    if alpha > 1:
        print(f"    (shape > 1 indicates IFR)")
    elif alpha < 1:
        print(f"    (shape < 1 indicates DFR)")
    else:
        print(f"    (shape = 1 is exponential, constant hazard)")

    # Test Weibull (IFR when k > 1, DFR when k < 1)
    try:
        k, loc, scale = stats.weibull_min.fit(output_lengths, floc=0)
        _, p_weibull = stats.kstest(output_lengths, 'weibull_min', args=(k, 0, scale))
        print(f"  Weibull fit: k={k:.3f}, scale={scale:.2f}, KS p-value={p_weibull:.4f}")
        if k > 1:
            print(f"    (k > 1 indicates IFR)")
        elif k < 1:
            print(f"    (k < 1 indicates DFR)")
        else:
            print(f"    (k = 1 is exponential, constant hazard)")
    except Exception as e:
        print(f"  Weibull fit failed: {e}")

    # Test log-normal (NOT IFR or DFR in general)
    shape, loc, scale = stats.lognorm.fit(output_lengths, floc=0)
    _, p_lognorm = stats.kstest(output_lengths, 'lognorm', args=(shape, 0, scale))
    print(f"  Log-normal fit: sigma={shape:.3f}, scale={scale:.2f}, KS p-value={p_lognorm:.4f}")
    print(f"    (Log-normal is generally NOT IFR or DFR)")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot hazard rate of ShareGPT output lengths')
    parser.add_argument('--dataset-path', type=str,
                        default='./ShareGPT_V3_unfiltered_cleaned_split.json',
                        help='Path to ShareGPT JSON file')
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Maximum number of samples to load')
    parser.add_argument('--output', type=str, default='sharegpt_hazard_rate.png',
                        help='Output plot filename')
    parser.add_argument('--use-tokenizer', action='store_true',
                        help='Use tokenizer for accurate token counting')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B',
                        help='Model for tokenizer (if --use-tokenizer)')
    args = parser.parse_args()

    # Load tokenizer if requested
    tokenizer = None
    if args.use_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {args.model}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            print("Falling back to word-based estimation")

    # Load ShareGPT data
    try:
        prompts, input_lengths, output_lengths = load_sharegpt_prompts(
            json_path=args.dataset_path,
            max_samples=args.max_samples,
            tokenizer=tokenizer
        )
    except FileNotFoundError:
        print(f"Dataset file not found: {args.dataset_path}")
        print("Please download ShareGPT dataset or specify correct path with --dataset-path")
        sys.exit(1)

    output_lengths = np.array(output_lengths)

    # Filter out very short outputs (likely incomplete)
    min_len = 4
    output_lengths = output_lengths[output_lengths >= min_len]
    print(f"\nFiltered to {len(output_lengths)} samples with output >= {min_len} tokens")

    # Analyze IFR property
    analyze_ifr(output_lengths)

    # Plot
    plot_hazard_rate(output_lengths, args.output)


if __name__ == '__main__':
    main()
