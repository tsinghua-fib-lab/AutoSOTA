# Final Report: paper-2344

- Title: IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection
- Primary metric: `AUC` (higher)
- Records: 13
- Generated: 2026-07-09T07:54:46Z

## Best Result

- Iteration: 11
- Idea: ALGO-10b — Reconstruction weight 0.02 (reduced)
- Primary metric: 77.43
- Commit: `9eda9a92ec31cbe8dfc6fb965291881397efd05f`
- Notes: Reduced reconstruction auxiliary loss weight from 0.05 to 0.02. AUC=77.43% (+2.59pp vs baseline 74.84%). Beats paper 75.97% by 1.46pp. First time all 5 runs >75%: 80.13, 75.20, 75.33, 78.83, 77.66. Lower variance (std 1.93%). Reconstruction with weight 0.02 provides optimal TCN feature stabilization without dominating deviation loss.
