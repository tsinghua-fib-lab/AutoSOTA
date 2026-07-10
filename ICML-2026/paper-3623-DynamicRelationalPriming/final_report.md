# Final Report: paper-3623

- Title: Dynamic Relational Priming Improves Transformer in Multivariate Time Series
- Primary metric: `MSE` (lower)
- Records: 9
- Generated: 2026-07-10T09:12:47Z

## Best Result

- Iteration: 6
- Idea: IDEA-05 — Per-head Temperature Scaling (Pattern 3)
- Primary metric: 0.253935
- Commit: `e03a8688139de9618f760561d6259561de3fce95`
- Notes: Added learnable per-head temperature parameter tau_h to PrimeFilterAttention softmax, initialized at 1.0 (identity). Each of the 8 heads can independently control its attention entropy. MSE improved from 0.253951 to 0.253935 (-0.006%%). MAE improved from 0.274657 to 0.274644 (-0.005%%). Cumulative: MSE -0.212%% from baseline 0.254474, MAE -0.928%% from baseline 0.277217.
