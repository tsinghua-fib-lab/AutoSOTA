# Final Report: paper-3235

- Title: Estimating Continuous Treatment Effects with Two-Stage Kernel Ridge Regression
- Primary metric: `MISE` (lower)
- Records: 32
- Generated: 2026-07-10T04:39:11Z

## Best Result

- Iteration: 27
- Idea: ALGO-01 — 30-seed: Gaussian + nu_H=3.5 + ell_x=500 + ell_t=3000
- Primary metric: 0.8819
- Commit: `dc44c667acd4d82b9f8c54dbe81711c11a708a4e`
- Notes: 30-seed validation confirms breakthrough: MISE=0.882 (SE=0.167) vs baseline 1.247 (-29.3%). Gaussian kernel for stage-1 dominates Laplace. Result is 2.2 SE below baseline - statistically significant.
