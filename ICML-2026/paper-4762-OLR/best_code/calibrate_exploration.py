"""
Calibration: run at multiple memory sizes to verify the algorithm's
behavior matches the paper's trends.
"""

import numpy as np
import json

from testSlidingWindowMAB import Bernoulli_Arm, Arm_Reading_Buffer
from testSlidingWindowAlgorithm import bucket_exploration
from explorationExperiment import ExplorationExperiment

n = 1000
window_size = 50
num_experiments = 10
delta = 0.01

# Test memory sizes representing 0.05W, 0.1W, 0.2W, 0.3W, 0.6W, 1.0W
memory_fractions = [0.05, 0.1, 0.2, 0.3, 0.6, 1.0]
memory_sizes = [max(1, int(f * window_size)) for f in memory_fractions]

print(f"Running with memory sizes: {memory_sizes}")
print(f"Memory fractions: {memory_fractions}")
print()

experiment = ExplorationExperiment(
    n=n, window_size=window_size,
    memory_sizes=memory_sizes,
    num_experiments=num_experiments,
    delta=delta,
    data_type='synthetic',
    arm_setting=None  # default: mix_in_setting
)
experiment.run_experiment()

alg = experiment.algorithm_results
baseline = experiment.baseline_results

print("=" * 70)
print(f"{'Memory':>8} {'Mean':>10} {'Median':>10} {'Max':>10} | {'BL Mean':>10} {'BL Median':>10} {'BL Max':>10}")
print("-" * 70)

for i, ms in enumerate(memory_sizes):
    m = np.mean(alg.y_mean[i])
    md = np.mean(alg.y_median[i])
    mx = np.mean(alg.y_max[i])
    bl_m = np.mean(baseline.y_mean[i])
    bl_md = np.mean(baseline.y_median[i])
    bl_mx = np.mean(baseline.y_max[i])
    print(f"{ms:>8} {m:>10.6f} {md:>10.6f} {mx:>10.6f} | {bl_m:>10.6f} {bl_md:>10.6f} {bl_mx:>10.6f}")

print()
print("Paper states:")
print("  At 0.05W memory: error > 0.6")
print("  At 0.3W memory: error < 0.3")
print("  At 1.0W memory: mean≈0.08, median≈0.12, max≈0.28 (Figure 2)")
