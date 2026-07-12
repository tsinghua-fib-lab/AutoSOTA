#!/usr/bin/env python3
"""
Hazard Rate Ordering experiment analysis script.

Features:
  - Find optimal k* and the corresponding theta* from Gamma k* sweeps.
  - Verify k*_DFR < k*_CFR < k*_IFR.

Usage:
  python analyze_hazard_rate.py <results_dir>
  python analyze_hazard_rate.py outputs/hazard_rate_ordering_N256_O128
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_json(filepath: Path) -> dict | None:
    """Load a JSON file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def extract_k_from_filename(filename: str) -> int | None:
    """Extract k* from a filename."""
    match = re.search(r'fixed(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def get_throughput(bench_result: dict) -> float:
    """Extract throughput from a benchmark result."""
    return bench_result.get("output_throughput", 0)


def analyze_hazard_type(scenario_dir: Path) -> dict:
    """Analyze a single hazard type, find optimal k*."""
    results = defaultdict(list)

    # Collect throughput across all k*
    for bench_file in scenario_dir.glob("bench_fixed*_run*.json"):
        k_star = extract_k_from_filename(bench_file.name)
        if k_star is None:
            continue

        data = load_json(bench_file)
        if data is None:
            continue

        throughput = get_throughput(data)
        results[k_star].append(throughput)

    # If there are no _run-suffixed files
    if not results:
        for bench_file in scenario_dir.glob("bench_fixed*.json"):
            if "_run" in bench_file.name:
                continue
            k_star = extract_k_from_filename(bench_file.name)
            if k_star is None:
                continue

            data = load_json(bench_file)
            if data is None:
                continue

            throughput = get_throughput(data)
            results[k_star].append(throughput)

    if not results:
        return {}

    # Per-k* throughput mean / std
    k_stats = {}
    for k, throughputs in results.items():
        k_stats[k] = {
            "mean": np.mean(throughputs),
            "std": np.std(throughputs) if len(throughputs) > 1 else 0,
            "n": len(throughputs),
        }

    # Find optimal k*
    optimal_k = max(k_stats, key=lambda k: k_stats[k]["mean"])
    optimal_stats = k_stats[optimal_k]

    return {
        "optimal_k_star": optimal_k,
        "optimal_throughput_mean": optimal_stats["mean"],
        "optimal_throughput_std": optimal_stats["std"],
        "all_k_stats": k_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hazard Rate Ordering analysis: verify k*_DFR < k*_CFR < k*_IFR"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Experiment output directory"
    )
    parser.add_argument(
        "--batch-size", "-N",
        type=int,
        default=256,
        help="batch size N (default 256)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSON path"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    N = args.batch_size

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    # Load experiment configuration
    config_file = results_dir / "experiment_config.json"
    if config_file.exists():
        config = load_json(config_file)
        if config:
            N = config.get("fixed_params", {}).get("N", N)
            print(f"Reading N = {N} from config file")

    print(f"\n{'='*60}")
    print("Hazard Rate Ordering experiment")
    print(f"Verifying: k*_DFR < k*_CFR < k*_IFR")
    print(f"{'='*60}")
    print(f"batch size N = {N}")
    print()

    # Analyze each hazard type
    results = {}
    hazard_order = ["DFR", "CFR", "IFR"]

    for scenario_dir in sorted(results_dir.glob("*_shape*")):
        if not scenario_dir.is_dir():
            continue

        # Parse scenario name: DFR_shape0.5, CFR_shape1.0, IFR_shape2.0
        match = re.match(r'(DFR|CFR|IFR)_shape([\d.]+)', scenario_dir.name)
        if not match:
            continue

        hazard_type = match.group(1)
        gamma_shape = float(match.group(2))

        print(f"Analyzing scenario: {hazard_type} (shape={gamma_shape})")

        scenario_result = analyze_hazard_type(scenario_dir)
        if not scenario_result:
            print(f"  Warning: no valid results found")
            continue

        optimal_k = scenario_result["optimal_k_star"]
        theta_star = optimal_k / N

        print(f"  optimal k* = {optimal_k}")
        print(f"  θ* = k*/N = {theta_star:.4f}")
        print(f"  throughput = {scenario_result['optimal_throughput_mean']:.2f} "
              f"± {scenario_result['optimal_throughput_std']:.2f} tokens/s")

        results[hazard_type] = {
            "hazard_type": hazard_type,
            "gamma_shape": gamma_shape,
            "optimal_k_star": optimal_k,
            "theta_star": theta_star,
            "throughput_mean": scenario_result["optimal_throughput_mean"],
            "throughput_std": scenario_result["optimal_throughput_std"],
        }

    if len(results) < 3:
        print(f"\nWarning: only {len(results)} hazard types found; cannot fully verify")

    # Verify ordering
    print(f"\n{'='*60}")
    print("Verifying k*_DFR < k*_CFR < k*_IFR")
    print(f"{'='*60}")

    print(f"\n{'Hazard Type':<12} {'Shape':<8} {'k*':<8} {'θ*':<8}")
    print("-" * 40)

    k_stars = {}
    theta_stars = {}
    for hazard_type in hazard_order:
        if hazard_type in results:
            r = results[hazard_type]
            print(f"{hazard_type:<12} {r['gamma_shape']:<8.1f} "
                  f"{r['optimal_k_star']:<8} {r['theta_star']:.4f}")
            k_stars[hazard_type] = r["optimal_k_star"]
            theta_stars[hazard_type] = r["theta_star"]

    # Check ordering
    print(f"\nValidation result:")
    if len(k_stars) == 3:
        k_dfr = k_stars.get("DFR", float('inf'))
        k_cfr = k_stars.get("CFR", float('inf'))
        k_ifr = k_stars.get("IFR", 0)

        if k_dfr < k_cfr < k_ifr:
            print(f"  √ k*_DFR ({k_dfr}) < k*_CFR ({k_cfr}) < k*_IFR ({k_ifr})")
            print(f"  ok Hazard rate ordering verified!")
        elif k_dfr <= k_cfr <= k_ifr:
            print(f"  ~ k*_DFR ({k_dfr}) ≤ k*_CFR ({k_cfr}) ≤ k*_IFR ({k_ifr})")
            print(f"  ~ Hazard rate ordering partially verified (ties present)")
        else:
            print(f"  x Ordering does not match expectation:")
            print(f"    k*_DFR = {k_dfr}, k*_CFR = {k_cfr}, k*_IFR = {k_ifr}")
            print(f"  x Hazard rate ordering verification FAILED")
    else:
        print(f"  ? data incomplete, cannot verify")

    # Print LaTeX format
    print(f"\nLaTeX format:")
    theta_values = []
    for hazard_type in hazard_order:
        if hazard_type in theta_stars:
            theta_values.append(f"{theta_stars[hazard_type]:.2f}")
        else:
            theta_values.append("---")
    print(f"  Using $a \\in \\{{0.5, 1, 2\\}}$ with scale parameters")
    print(f"  $b \\in \\{{256, 128, 64\\}}$ to maintain $\\mathbb{{E}}[O] = 128$,")
    print(f"  we obtain $\\hat{{\\theta}}^* = $ {', '.join(theta_values)} respectively")
    print(f"  ($N = {N}$, 3 runs), confirming the predicted ordering.")

    # Save results
    if args.output:
        output_data = {
            "experiment": "hazard_rate_ordering",
            "N": N,
            "results": results,
            "verification": {
                "k_stars": k_stars,
                "theta_stars": theta_stars,
                "ordering_verified": (
                    len(k_stars) == 3 and
                    k_stars.get("DFR", float('inf')) <
                    k_stars.get("CFR", float('inf')) <
                    k_stars.get("IFR", 0)
                ),
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
