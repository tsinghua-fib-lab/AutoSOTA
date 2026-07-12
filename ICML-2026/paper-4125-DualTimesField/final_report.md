# Final Report: paper-4125

- Title: DualTimesField: Rethinking Time Series as Continuous-Time Trends and Events
- Primary metric: `MSE` (lower)
- Records: 13
- Generated: 2026-07-11T14:07:52Z

## Best Result

- Iteration: 9
- Idea: PARAM-P1-HUBER-DGF-WEIGHT-03 — Huber beta=0.01 + DGF weight 0.2->0.3
- Primary metric: 0.000313
- Commit: `0cb243bec4be30a083790a9d61c6a6005c40915c`
- Notes: Increased DGF weight from 0.2 to 0.3 with Huber beta=0.01. MSE improved from 0.000320 to 0.000313 (-2.2%). MAE improved from 0.010987 to 0.010909 (-0.7%). Active atoms=1. Trend: higher DGF weight continues to help under Huber loss.
