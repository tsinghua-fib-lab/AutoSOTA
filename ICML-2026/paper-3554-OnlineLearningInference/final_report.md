# Final Report: paper-3554

- Title: Online Learning and Inference for Cox Proportional Hazards Model Using Renewable Sieve Estimation
- Primary metric: `Correctly Recovered Inference Results` (higher)
- Records: 11
- Generated: 2026-07-10T09:31:07Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — Alpha-nu grid optimization (best: alpha=0.80, nu=0.25)
- Primary metric: 23.0
- Commit: `75b40cd1416fbe8dfad01e81e7980a7027497cc3`
- Notes: Grid search over 8 alpha x 7 nu = 56 combinations. Best: alpha=0.80, nu=0.25 (p0=5). Achieves 23/23, Pearson r=0.99998, MAD=0.00023 — a 45% reduction in MAD vs baseline (0.00042). Also best: alpha=1.20, nu=0.20 (p0=5) with identical metrics. Large p0 values (>8) cause convergence failures. Small nu (<0.15) with small alpha gives worse metrics.
