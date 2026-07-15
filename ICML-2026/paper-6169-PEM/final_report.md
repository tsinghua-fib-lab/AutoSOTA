# Final Report: paper-6169

- Title: Depth over Fidelity in Fixed-Budget Noisy Evolution Strategies
- Primary metric: `log10_regret_RB_PEM` (lower)
- Records: 13
- Generated: 2026-07-14T10:36:24Z

## Best Result

- Iteration: 12
- Idea: ALGO-12 — HeteroRobust + pop=40 combo
- Primary metric: 2.9159
- Commit: `d69553a9081d55678b9894bdcece82575228795c`
- Notes: BEST! HeteroRobust(winsorized z-pool, trimmed bootstrap) + pop=40 gives 2.9159 vs baseline 3.24 (-10.0%). Gap to Resample widens to 0.58. All guardrails passed.
