# Final Report: paper-1153

- Title: DAVE: Distribution-Aware Attribution via ViT Gradient Decomposition
- Primary metric: `GridPG` (higher)
- Records: 10
- Generated: 2026-07-06T13:48:10Z

## Best Result

- Iteration: 9
- Idea: PARAM-01 — noise_alpha=0.85 + 125 MC steps
- Primary metric: 85.8
- Commit: `0ae0679fe042ef374c512e27049e46d54a2c1c25`
- Notes: Further increased MC steps to 125 with noise_alpha=0.85. EnergyPG: 85.66->85.80 (+0.14 p.p.). Diminishing returns: 50->75: +0.54, 75->100: +0.40, 100->125: +0.14. Approaching convergence. Cumulative: 84.25->85.80 (+1.55 p.p.).
