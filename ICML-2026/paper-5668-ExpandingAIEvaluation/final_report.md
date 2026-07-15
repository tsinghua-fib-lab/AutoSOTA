# Final Report: paper-5668

- Title: Expanding the AI Evaluation Toolbox with Statistical Models
- Primary metric: `Bias` (lower)
- Records: 9
- Generated: 2026-07-14T08:08:22Z

## Best Result

- Iteration: 3
- Idea: ALGO-06+ALGO-04 — Aggressive REML (40/37.5) + bias correction (coeff=0.40)
- Primary metric: 0.000607
- Commit: `341e0eec5e9b2de4f04495c89cd9b5952b04ce04`
- Notes: Sweep-optimized parameters: REML factor=40/37.5 (SE scale 1.033), bias coefficient=0.40 (link-scale correction proportional to SE^2). Bias reduced 64.3% (0.001701->0.000607), now 44% better than RF (0.00109). Coverage improved to 0.952 (above nominal). CI_Width 0.160 still below RF (0.163). Pareto improvement on all fronts vs baseline.
