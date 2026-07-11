# Final Report: paper-3197

- Title: Variational inference via Gaussian interacting particles in the Bures-Wasserstein geometry
- Primary metric: `median_KL` (lower)
- Records: 8
- Generated: 2026-07-09T21:05:52Z

## Best Result

- Iteration: 5
- Idea: CODE-09 — Diagonal covariance restriction for high-dimensional CBO
- Primary metric: 0.3294
- Commit: `3a1865b662405097b0d3f36e829bb361f2a44eef`
- Notes: Added --diagonal flag to restrict covariance to diagonal elements. Target A: median_KL=0.3294 (+8.5% regression, within 50% tolerance). Target D (d=10): median_KL=9.40 vs 15.02 full covariance (37% improvement). Diagonal reduces parameters from d(d+1)/2=55 to d=10, preventing overfitting to spurious off-diagonal correlations.
