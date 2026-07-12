"""
Full sweep matching Figure 2: n=1000, W=50, memory from 5 to 50
"""
import numpy as np
from explorationExperiment import ExplorationExperiment

n = 1000
window_size = 50
num_experiments = 10
delta = 0.01

# Match Figure 2 x-axis range
memory_sizes = list(range(5, 55, 5))  # 5, 10, 15, ..., 50

print(f"n={n}, W={window_size}, memory_sizes={memory_sizes}")
print(f"num_experiments={num_experiments}, delta={delta}")
print()

experiment = ExplorationExperiment(
    n=n, window_size=window_size,
    memory_sizes=memory_sizes,
    num_experiments=num_experiments,
    delta=delta,
    data_type='synthetic',
    arm_setting=None
)
experiment.run_experiment()

alg = experiment.algorithm_results
baseline = experiment.baseline_results

print("=" * 80)
print(f"{'Mem':>5} {'Alg Mean':>12} {'Alg Med':>12} {'Alg Max':>12} | {'BL Mean':>12} {'BL Med':>12} {'BL Max':>12}")
print("-" * 80)

for i, ms in enumerate(memory_sizes):
    m = np.mean(alg.y_mean[i])
    md = np.mean(alg.y_median[i])
    mx = np.mean(alg.y_max[i])
    bl_m = np.mean(baseline.y_mean[i])
    bl_md = np.mean(baseline.y_median[i])
    bl_mx = np.mean(baseline.y_max[i])
    print(f"{ms:>5} {m:>12.6f} {md:>12.6f} {mx:>12.6f} | {bl_m:>12.6f} {bl_md:>12.6f} {bl_mx:>12.6f}")

# Focus on the rubric target: memory=50
i50 = memory_sizes.index(50)
print()
print("=== METRICS AT MEMORY=50 (RUBRIC TARGET) ===")
print(f"Mean Difference:   {np.mean(alg.y_mean[i50]):.6f}")
print(f"Median Difference: {np.mean(alg.y_median[i50]):.6f}")
print(f"Max Difference:    {np.mean(alg.y_max[i50]):.6f}")
print()
print("Paper (Figure 2): Mean=0.08, Median=0.12, Max=0.28")
print("Baseline paper:   Mean=0.33, Median=0.36, Max=0.45")

# Per-run details at memory=50
print()
print("Per-run at memory=50:")
for run in range(num_experiments):
    print(f"  Run {run+1}: mean={alg.y_mean[i50][run]:.6f}, median={alg.y_median[i50][run]:.6f}, max={alg.y_max[i50][run]:.6f}")
