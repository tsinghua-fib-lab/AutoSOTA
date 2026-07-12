"""
Reproduction script for paper 4762: Online Learning with Recency:
Algorithms for Sliding-window Streaming Multi-armed Bandits.

Targets: exploration experiment with n=1000, W=50, memory=50,
Algorithm 1 (bucket-based epsilon-exploration), 10 runs.
"""

import numpy as np
import sys
import json
from pathlib import Path

# Import from the repo
from testSlidingWindowMAB import Bernoulli_Arm, Arm_Reading_Buffer
from testSlidingWindowAlgorithm import bucket_exploration
from explorationExperiment import ExplorationExperiment

def main():
    # Exact rubric parameters
    n = 1000
    window_size = 50
    memory = 50  # the rubric says memory=50
    num_experiments = 10
    delta = 0.01

    print(f"=== Reproduction: Paper 4762 Exploration Experiment ===")
    print(f"n_arms={n}, window_size={window_size}, memory={memory}, n_runs={num_experiments}")
    print(f"Algorithm: bucket_exploration (Algorithm 1)")
    print()

    # Run the experiment using the existing ExplorationExperiment class
    experiment = ExplorationExperiment(
        n=n,
        window_size=window_size,
        memory_sizes=[memory],  # single memory point: 50
        num_experiments=num_experiments,
        baseline_expiration_type='clear',
        data_type='synthetic',
        delta=delta,
        arm_setting=None  # default: 'mix_in_setting' = Bern(p), p~Uniform[0,1]
    )
    experiment.run_experiment()

    # Extract algorithm results
    alg = experiment.algorithm_results
    # y_mean, y_median, y_max are shape (len(memory_sizes), num_experiments)
    mean_of_mean = np.mean(alg.y_mean[0])
    mean_of_median = np.mean(alg.y_median[0])
    mean_of_max = np.mean(alg.y_max[0])

    # Extract baseline results
    baseline = experiment.baseline_results
    bl_mean_of_mean = np.mean(baseline.y_mean[0])
    bl_mean_of_median = np.mean(baseline.y_median[0])
    bl_mean_of_max = np.mean(baseline.y_max[0])

    print("=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"n_arms: {n}")
    print(f"window_size (W): {window_size}")
    print(f"memory: {memory}")
    print(f"delta: {delta}")
    print(f"num_experiments (runs): {num_experiments}")
    print()
    print("--- Algorithm (bucket_exploration / Algorithm 1) ---")
    print(f"Mean Difference:   {mean_of_mean:.6f}")
    print(f"Median Difference: {mean_of_median:.6f}")
    print(f"Max Difference:    {mean_of_max:.6f}")
    print()
    print("--- Baseline (top-k adaptation of streaming MABs) ---")
    print(f"Mean Difference:   {bl_mean_of_mean:.6f}")
    print(f"Median Difference: {bl_mean_of_median:.6f}")
    print(f"Max Difference:    {bl_mean_of_max:.6f}")
    print()
    print("--- Paper Target Values (from Figure 2) ---")
    print(f"Mean Difference:   0.08")
    print(f"Median Difference: 0.12")
    print(f"Max Difference:    0.28")
    print()
    print("--- Baseline Target Values ---")
    print(f"Mean Difference:   0.33")
    print(f"Median Difference: 0.36")
    print(f"Max Difference:    0.45")
    print()

    print("Per-run algorithm results:")
    for i in range(num_experiments):
        print(f"  Run {i+1}: mean={alg.y_mean[0][i]:.6f}, median={alg.y_median[0][i]:.6f}, max={alg.y_max[0][i]:.6f}")

    # Check against rubric bounds
    print()
    print("--- Rubric Bounds Check ---")
    rubric_mean_lo, rubric_mean_hi = 0.055, 0.33
    rubric_median_lo, rubric_median_hi = 0.096, 0.36
    rubric_max_lo, rubric_max_hi = 0.263, 0.45

    checks = {
        "Mean Difference": (mean_of_mean, rubric_mean_lo, rubric_mean_hi),
        "Median Difference": (mean_of_median, rubric_median_lo, rubric_median_hi),
        "Max Difference": (mean_of_max, rubric_max_lo, rubric_max_hi),
    }
    for name, (val, lo, hi) in checks.items():
        ok = lo <= val <= hi
        status = "WITHIN BOUNDS" if ok else "OUTSIDE BOUNDS"
        print(f"  {name}: {val:.6f} [{lo:.4f}, {hi:.2f}] {status}")

    # Output JSON for easy parsing
    result = {
        "n": n,
        "window_size": window_size,
        "memory": memory,
        "num_experiments": num_experiments,
        "algorithm": {
            "mean_difference": float(mean_of_mean),
            "median_difference": float(mean_of_median),
            "max_difference": float(mean_of_max),
        },
        "baseline": {
            "mean_difference": float(bl_mean_of_mean),
            "median_difference": float(bl_mean_of_median),
            "max_difference": float(bl_mean_of_max),
        },
    }
    print()
    print("JSON_RESULT:", json.dumps(result))

if __name__ == "__main__":
    main()
