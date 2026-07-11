# Final Report: paper-4071

- Title: Statistical Early Stopping for Reasoning Models
- Primary metric: `Power` (higher)
- Records: 6
- Generated: 2026-07-11T03:01:50Z

## Best Result

- Iteration: 3
- Idea: PARAM-2 — Alpha=0.045 achieves Pareto improvement (FPR=4%, Power=50%)
- Primary metric: 50.0
- Commit: `f5d65ad9ac9890c9b2f3bfd21253263cf48074ce`
- Notes: Fixed alpha propagation bug (--alpha was parsed but ignored). Alpha sweep revealed alpha=0.045 gives FPR=4.00% (below 5% target) at Power=50.00% (same as baseline 50.00%) — Pareto-dominant: same Power with lower FPR. TokensSaved=38.74% (slightly below baseline 39.17%).
