# Final Report: paper-3912

- Title: Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation
- Primary metric: `RMSE` (lower)
- Records: 13
- Generated: 2026-07-10T17:31:18Z

## Best Result

- Iteration: 11
- Idea: IDEA-21 — ARIMA(2,1,0) with window=200
- Primary metric: 5.8653
- Commit: `fa6a7c8a2ea0ce4dd43b9fdeeb8c57571d483908`
- Notes: ARIMA(2,1,0) + window=200: RMSE 5.8653 (-1.96% vs baseline 5.9825). Best result! IDR: 83.69->82.00 (-2.0%), ARS: 2.55->2.42 (-5.0%). MAE* slight regression (0.455->0.466, +2.4%) within 5% tolerance. Second AR term captures richer differenced dynamics on the larger window.
