# Final Report: paper-2081

- Title: $\texttt{ShaplEIG}$: Bayesian Experimental Design for Shapley Value Estimation
- Primary metric: `MSE` (lower)
- Records: 9
- Generated: 2026-07-17T10:26:34Z

## Best Result

- Iteration: 4
- Idea: ALGO-03 — Complement pairing for initial design
- Primary metric: 2.00276e-07
- Commit: `29186d498a8c2323937e98eee64013d1af11e5ba`
- Notes: ALGO-03: Complement pairing (Covert & Lee 2021). For each sampled initial coalition S, also add N\S. MSE 2.00e-07 vs baseline 2.27e-07: ~12%% improvement. Seed 2 shows dramatic improvement (1.30e-07 vs 2.19e-07).
