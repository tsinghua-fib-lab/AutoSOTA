# Final Report: paper-3674

- Title: Neuro-Symbolic AI for Analytical Solutions of Differential Equations
- Primary metric: `Relative L2 Error` (lower)
- Records: 13
- Generated: 2026-07-10T12:37:16Z

## Best Result

- Iteration: 11
- Idea: PARAM-02-v2 — Moderate hybrid: 3-phase Adam polish + n_samples=100
- Primary metric: 4.766801e-18
- Commit: `acf49ccfeff35d017ce31c4bb3a116488aa5cb4a`
- Notes: Optimized both stages: Stage I n_samples=100 (2.2s), Stage II hybrid L-BFGS + moderate 3-phase Adam (3000+3000+3000, 3 fine restarts). Total: 55.2s vs baseline 191.1s (71% reduction). Relative L2 Error preserved at exactly 4.766801e-18 (machine epsilon). PARETO improvement over all previous iterations.
