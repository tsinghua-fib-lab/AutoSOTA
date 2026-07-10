# Final Report: paper-2217

- Title: CLARITree: Cholesky and Lookahead Accelerations for Regression with Interpretable Piecewise Linear Trees
- Primary metric: `Test R2` (higher)
- Records: 7
- Generated: 2026-07-09T13:17:44Z

## Best Result

- Iteration: 5
- Idea: PARAM-01 — n_thresholds=35 with baseline hyperparams
- Primary metric: 0.76
- Commit: `f318e03d5dd8faaed979ecd2aea6e803591afc1f`
- Notes: PARAM-01: n_thresholds=35. Test R2 0.7600 (+0.0101 vs baseline 0.7499). +0.0041 over n_thresholds=30. Train R2 0.7715 (+0.0058 vs baseline). More thresholds continue to improve split quality.
