# Optimization Results: Modified K-means Algorithm with Local Optimality Guarantees

## Summary
- Total iterations: 12
- Best `mean_clustering_loss`: **801206.8717** (baseline: 808032.0781, improvement: **-0.84%**)
- Target: 791871.4365 — **NOT REACHED** (gap: 9335.4)
- Best commit: `6914865` (iter-9: Best-of-20 independent Min-D-LO runs per trial)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mean_clustering_loss | 808032.0781 | 801206.8717 | -6825.2 (-0.84%) |
| minimum_clustering_loss | 801206.8717 | 801206.8717 | 0 (0%) |
| average_computation_time_s | 2.9345 | 27.3966 | +24.46 (+833%) |
| average_number_of_iterations | 93.1 | 38.1 | -55.0 (-59%) |

## Key Changes Applied
| Change | Effect | Notes |
|--------|--------|-------|
| kmeans++ init (iter 1) | mean 808032→804799 (-0.4%) | Biggest single change; faster convergence too |
| Best-of-20 Min-D-LO per trial (iter 9) | mean 804799→801207 (-0.44%) | 20 independent restarts within each trial |

## Optimization Trajectory
| Iter | Mean Loss | Delta | Note |
|------|-----------|-------|------|
| 0 (baseline) | 808032.08 | — | Paper baseline, rand init |
| 1 | 804799.09 | -3233 | kmeans++ init |
| 2 | 806235.97 | +1437 | Fixed seeds: worse, rolled back |
| 3 | 804799.09 | 0 | Warm start from k-means: no change |
| 4 | 804799.09 | 0 | Best-of-3 seeds: no change |
| 5 | 804799.09 | 0 | Two-stage from D-LO: no change |
| 6 | 804799.09 | 0 | Perturbation restarts: no change |
| 7 | 806595.19 | +1796 | Mixed init: worse, rolled back |
| 8 | 801566.09 | -3233 | **Best-of-5 restarts: breakthrough!** |
| 9 | 801206.87 | -359 | **Best-of-20 restarts: std=0** |
| 10 | – | FAILED | 2-swap too slow |
| 11 | 801206.87 | 0 | K+1 merge: no change |
| 12 | – | FAILED | 50 restarts: timeout |

## Critical Finding: The D-Local Optimum Barrier

With 20 independent kmeans++ restarts per trial, ALL 20 trials converge exactly to **801206.8717** (std=0). This strongly suggests that:

1. **801207 is likely the global D-local optimum** for K=5 on the News20-1 dataset
2. The target of 791871 requires solutions that are NOT D-locally optimal under the current formulation
3. To reach below 801207 would require fundamentally different neighborhood definitions or algorithms

The algorithm's design (D-local optimality via single-point moves) appears to have found its theoretical lower bound for this specific problem instance.

## What Worked
- **kmeans++ initialization**: Critical improvement. Eliminated bad initialization causing 1009869+ losses
- **Multiple independent restarts**: Taking the best of N independent Min-D-LO runs dramatically reduces mean loss
- **Key insight**: These two techniques together ensure hitting the D-local optimum consistently

## What Didn't Work
- Fixed seeds: Specific seed values were not better than random system seeds
- Warm starting from K-means or D-LO results: Converges to same local optima
- Perturbation restarts (basin-hopping): D-local optima are too strong attractors
- Mixed random+kmeans++ init: Random init hits bad basins, increasing mean
- K+1 cluster merge: After merging, still converges to 801207
- 2-point swap: O(N²D) computation too slow within 900s timeout
- 50 restarts: Exceeds 900s timeout

## Algorithm Analysis

The Min-D-LO algorithm's convergence guarantees make it very stable. After finding a D-local optimum, no single-point move can improve it. The consistent convergence to 801207 across many independent restarts suggests this is the global minimum of the D-local landscape.

To go below 791871, future work could explore:
1. **2-point swaps** with efficient approximation (bound-based pruning to reduce from O(N²) to O(N log N))
2. **K-means with K=6** then post-processing to merge and get K=5 (requires matching computational budget)
3. **Simulated annealing** on top of Min-D-LO to escape the 801207 basin
4. **Different Bregman divergences** (KL or Itakura-Saito) which may have different local optima
5. **Hierarchical initialization** using Ward linkage then refinement

## Top Remaining Ideas (for future runs)
1. Efficient 2-swap with pruning: only check top-M border points per cluster (M~50) → O(M×N) instead of O(N²)
2. Simulated annealing: accept uphill moves with decreasing probability to escape 801207 basin
3. KL-divergence Bregman: different divergence on TF-IDF data may find lower loss clusters
4. Hierarchical init: use Ward clustering to initialize then refine with Min-D-LO
5. Parallel OMP: with 4 cores, run 80 restarts in same time as 20 → higher chance of escaping basin
