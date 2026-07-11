# Final Report: paper-4194

- Title: Estimation of Treatment Effects Under Nonstationarity via the Truncated Policy Gradient Estimator
- Primary metric: `MAE_k1` (lower)
- Records: 6
- Generated: 2026-07-11T04:55:31Z

## Best Result

- Iteration: 5
- Idea: IDEA-05 — Mixing-rate-proportional softmax k-ensemble
- Primary metric: 23.2
- Commit: `b5ca193919739da117022249f7a747d555695569`
- Notes: Sigmoid-weighted ensemble: higher mixing rates get more weight on k=3/k=5. MAE 23.20 vs baseline 29.73. STD 0.369 vs 0.246. Net RMSE improved 9.2pct. Demonstrates paper bias-variance tradeoff.
