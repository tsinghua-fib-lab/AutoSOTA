# Final Report: paper-4877

- Title: Skipping the Zeros in Diffusion Models for Sparse Data Generation
- Primary metric: `SCC` (higher)
- Records: 8
- Generated: 2026-07-12T15:09:47Z

## Best Result

- Iteration: 6
- Idea: ALGO-4 — DDIM sqrt 150 steps (refined sqrt spacing) - BEST
- Primary metric: 0.9761
- Commit: `3e0e38e469c3c791f5c1939895d538499ee8efd8`
- Notes: DDIM 150 steps with sqrt spacing. SCC=0.9761 (+2.72% over baseline 0.9503). MMD=0.0437 (-65.2% over baseline 0.1257). Marginally better SCC and MMD than sqrt 100-step. Best overall result: Pareto improvement across all metrics.
