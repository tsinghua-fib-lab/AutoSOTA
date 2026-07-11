# Final Report: paper-4232

- Title: Higher-Order Certified Robustness for Regression
- Primary metric: `Absolute_Accuracy_R0.15` (higher)
- Records: 6
- Generated: 2026-07-11T05:28:11Z

## Best Result

- Iteration: 5
- Idea: PARAM-02 — Increase MC samples to N=20000 with n_trials=5 median aggregation
- Primary metric: 95.0
- Commit: `562bedd152fc87aa174f94c6c3cd0e549bffef58`
- Notes: Best result so far. Combined CODE-01 (one-sided CI) + PARAM-01 (n_trials=5 median aggregation) + PARAM-02 (N=20000). N=20000 reduces SE by sqrt(2) vs N=10000, tightening CIs further. AbsAcc_R0.15 = 95.0 (matches paper). AbsAcc_R0.20 = 22.0 (baseline 14.0, +8pp). Mean radius = 0.189 (baseline ~0.17). MeanDist_R0.15 stable at 5.09. AbsAcc_R0.25 = 4.0 (baseline 1.0).
