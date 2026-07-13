# Final Report: paper-5179

- Title: Delving into Non-Exchangeability for Conformal Prediction in Graph-Structured Multivariate Time Series
- Primary metric: `Coverage` (higher)
- Records: 5
- Generated: 2026-07-13T03:24:01Z

## Best Result

- Iteration: 2
- Idea: ALGO-5179-06 — Quantile Smoothness Regularization + Fix crossing penalty
- Primary metric: 0.8946
- Commit: `c08fa3dcdf3aa2d92ba72b89319c49dd6ea1500e`
- Notes: Added SmoothQuantileLoss with crossing penalty (from config, previously unused in Lightning runner) and smoothness regularization (weight=0.5). Coverage improved +0.31pp to 0.8946, PI-Width unchanged at 8.97, Winkler nearly unchanged at 14.36 (+0.02). Discovered bug: crossing_penalty was never applied during Lightning training.
