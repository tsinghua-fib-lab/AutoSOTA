# Final Report: paper-1806

- Title: Beyond Procedure: Substantive Fairness in Conformal Prediction
- Primary metric: `maxROR` (lower)
- Records: 8
- Generated: 2026-07-07T13:11:13Z

## Best Result

- Iteration: 6
- Idea: PARAM-01 — HPO Re-enablement for SAPS (T=0.625, lambda=0.245)
- Primary metric: 1.044
- Commit: `df73fb04bc50aa0c36d3dc8021f2cc3fbed9ff55`
- Notes: HPO with 50 Optuna TPE trials found T=0.625, lambda=0.245 (vs paper T=0.74, lambda=0.28). FIRST maxROR improvement: 1.104->1.044 (-5.4 pct). Accuracy slightly regressed (84.0->83.46 pct, -0.5 pct) within tolerance. Set Size increased (1.705->1.795, +5.3 pct) within tolerance. Marginal method achieved even better maxROR=1.037 with 84.5 pct accuracy. Bounded search: T in [0.1,2.0], lambda in [0.01,0.5].
