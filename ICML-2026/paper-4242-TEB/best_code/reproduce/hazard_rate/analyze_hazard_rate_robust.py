#!/usr/bin/env python3
"""
Improved Hazard Rate analysis script.
Uses more robust statistics to handle outliers.
"""

import json
import os
import sys
from collections import defaultdict
import numpy as np
from scipy import stats

def load_throughput_data(base_dir):
    """Load all throughput data."""
    configs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.endswith(("_shape0.5", "_shape1.0", "_shape2.0")):
            configs.append(name)

    data = {}
    for config in sorted(configs):
        config_dir = os.path.join(base_dir, config)
        fixed_throughputs = defaultdict(list)

        for filename in os.listdir(config_dir):
            if filename.startswith("bench_fixed") and filename.endswith(".json"):
                parts = filename.replace("bench_fixed", "").replace(".json", "")
                k_str = parts.split("_run")[0]
                k = int(k_str)

                filepath = os.path.join(config_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        d = json.load(f)
                        throughput = d.get("output_throughput", 0)
                        fixed_throughputs[k].append(throughput)
                except:
                    pass

        data[config] = dict(fixed_throughputs)

    return data


def robust_estimate(values, method='trimmed_mean'):
    """Estimate central tendency with robust statistics."""
    values = np.array(values)

    if method == 'trimmed_mean':
        # Drop the top and bottom 20% and take the mean
        return stats.trim_mean(values, 0.2)

    elif method == 'median':
        return np.median(values)

    elif method == 'winsorized_mean':
        # Winsorize: replace extreme values with percentile bounds
        lower = np.percentile(values, 10)
        upper = np.percentile(values, 90)
        clipped = np.clip(values, lower, upper)
        return np.mean(clipped)

    elif method == 'iqr_filter':
        # Filter outliers using the IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = values[(values >= lower) & (values <= upper)]
        return np.mean(filtered) if len(filtered) > 0 else np.mean(values)

    else:  # simple mean
        return np.mean(values)


def find_optimal_k(throughputs, method='trimmed_mean'):
    """Find optimal k*."""
    estimates = {}
    for k, values in throughputs.items():
        estimates[k] = robust_estimate(values, method)

    best_k = max(estimates, key=estimates.get)
    return best_k, estimates


def bootstrap_confidence_interval(values, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return np.mean(bootstrap_means), lower, upper


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_hazard_rate_robust.py <output_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]

    # Read N
    config_file = os.path.join(base_dir, "experiment_config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            N = config.get("fixed_params", {}).get("N", 256)
    else:
        N = 256

    print(f"N = {N}")
    print("=" * 70)

    data = load_throughput_data(base_dir)

    methods = ['simple', 'trimmed_mean', 'median', 'winsorized_mean', 'iqr_filter']

    results = {}
    for method in methods:
        results[method] = {}
        print(f"\nmethod: {method}")
        print("-" * 50)

        for config in sorted(data.keys()):
            best_k, estimates = find_optimal_k(data[config], method)
            theta_star = best_k / N
            results[method][config] = best_k

            print(f"  {config}: k* = {best_k}, θ* = {theta_star:.4f}")

    print("\n" + "=" * 70)
    print("Ordering verification (expected: DFR < CFR < IFR)")
    print("=" * 70)

    for method in methods:
        r = results[method]
        keys = sorted(r.keys())

        dfr_k = r.get([k for k in keys if 'DFR' in k][0], 0)
        cfr_k = r.get([k for k in keys if 'CFR' in k][0], 0)
        ifr_k = r.get([k for k in keys if 'IFR' in k][0], 0)

        expected = dfr_k < cfr_k < ifr_k

        print(f"\n{method:20s}: DFR={dfr_k:3d}, CFR={cfr_k:3d}, IFR={ifr_k:3d}")
        print(f"                      ordering: {'ok' if expected else 'mismatch'}")

    # Bootstrap analysis
    print("\n" + "=" * 70)
    print("Bootstrap confidence-interval analysis (95% CI)")
    print("=" * 70)

    for config in sorted(data.keys()):
        print(f"\n{config}:")
        throughputs = data[config]

        # Compute a confidence interval for each k
        ci_results = []
        for k in sorted(throughputs.keys()):
            mean, lower, upper = bootstrap_confidence_interval(throughputs[k])
            ci_results.append((k, mean, lower, upper))

        # Find points whose CIs overlap the optimum
        best_k, best_mean, _, _ = max(ci_results, key=lambda x: x[1])

        print(f"  optimal k* = {best_k} (mean = {best_mean:.1f})")
        print(f"  Points whose CIs overlap the optimum:")

        for k, mean, lower, upper in ci_results:
            _, best_lower, best_upper = bootstrap_confidence_interval(throughputs[best_k])
            # Check for overlap
            if not (upper < best_lower or lower > best_upper):
                print(f"    k={k:3d}: {mean:.1f} [{lower:.1f}, {upper:.1f}]")


if __name__ == "__main__":
    main()
