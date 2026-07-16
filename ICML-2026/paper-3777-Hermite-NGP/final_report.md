# Final Report: paper-3777

- Title: Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs
- Primary metric: `Relative L2 Error` (lower)
- Records: 8
- Generated: 2026-07-16T01:15:40Z

## Best Result

- Iteration: 2
- Idea: SOTA-ALGO-01 — Sobol QMC collocation sampling
- Primary metric: 1.6229e-05
- Commit: `2d8c3f32860756cd354e0361662786523b085a8c`
- Notes: Replaced uniform random collocation points with scrambled Sobol quasi-Monte Carlo sequences. Pre-generated 10M-point Sobol pool, random slice per epoch. Combined with hash-size 14. Achieved 1.62e-05, beating the paper large-model result (1.81e-05) and exceeding the rubric lower bound (2.25e-05). 73% total reduction from baseline (6.06e-05).
