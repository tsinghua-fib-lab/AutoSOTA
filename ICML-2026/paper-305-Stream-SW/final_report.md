# Final Report: paper-305

- Title: Streaming Sliced Optimal Transport
- Primary metric: `Wasserstein-2_score` (lower)
- Records: 13
- Generated: 2026-07-04T18:37:34Z

## Best Result

- Iteration: 12
- Idea: CODE-11 — Final 5-seed validation with c=1.0
- Primary metric: 1.0345
- Commit: `df0fcc67f898e473c254f03f66cc31e3e18a7db5`
- Notes: 5-seed validation of best config (Adam+OneCycleLR+GradClip+c=1.0). Mean=1.0345, seeds: [1.0177, 0.9727, 1.1691, 1.0025, 1.0105], std=0.069. Seeds 1,2,4,5 all near or below 1.02. Seed 3 is the outlier at 1.169. Total improvement from baseline (1.9636): -47.3%. Approach: Adam handles noisy KLL gradients better than SGD, OneCycleLR enables faster early convergence, gradient clipping stabilizes high-LR phase, c=1.0 maximizes KLL sketch accuracy.
