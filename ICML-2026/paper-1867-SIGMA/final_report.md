# Final Report: paper-1867

- Title: Generalizing Multi-Scale Time-Series Modeling with a Single Operator
- Primary metric: `MSE` (lower)
- Records: 11
- Generated: 2026-07-08T12:19:48Z

## Best Result

- Iteration: 6
- Idea: ALGO-006 — Jittering augmentation (sigma=0.01)
- Primary metric: 0.164466
- Commit: `9e6a3e69398f286bfb46bcb842abe66d77bf2253`
- Notes: Jittering (sigma=0.01) + Huber(delta=1.0) + gradient_clip(1.0) + d_model=32. Per-horizon: 96=0.137 (-2.6pct), 192=0.153 (-1.3pct), 336=0.171 (+0.4pct slightly worse), 720=0.197 (-2.1pct BEST 720 YET). Avg MSE=0.1645 vs baseline 0.167 (-1.7pct). Jittering most effective for longest horizon. Slight regression on 336 suggests sigma may be too high for medium horizons.
