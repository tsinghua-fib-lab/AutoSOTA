# Final Report: paper-643

- Title: DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting
- Primary metric: `MSE` (lower)
- Records: 10
- Generated: 2026-07-05T07:15:14Z

## Best Result

- Iteration: 6
- Idea: param-batch128-lowp — Batch size=128 + p=[0.02,0.4] -- BEST
- Primary metric: 0.386
- Commit: `db6cb6df3c8ba9c4c1a06f34ec032e24ecd94e6a`
- Notes: Best result. Batch size 128 with narrower dropout bounds [0.02, 0.4]. MSE=0.3860 (-0.44% vs baseline 0.3877). MAE=0.4036 (= baseline). Lower dropout range provides gentler regularization. H=720 benefits most (MSE=0.4395 vs baseline 0.4460, -1.5%).
