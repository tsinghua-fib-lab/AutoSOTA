# Final Report: paper-3215

- Title: Beyond Point Predictions: Manifold Expansion and Dual Alignment for Robust Time Series Distillation
- Primary metric: `MSE` (lower)
- Records: 14
- Generated: 2026-07-10T00:52:05Z

## Best Result

- Iteration: 5
- Idea: PARAM-CAPACITY,ALGO-04 — Increase d_model=256 d_ff=1024 + freq loss lambda=0.1
- Primary metric: 0.428
- Commit: `f9d793e28e0f3cc4e5d84b9e76c4664f8676caca`
- Notes: Increased student capacity: d_model=128→256, d_ff=512→1024. Frequency loss lambda_freq=0.05→0.1. MSE improved 0.429→0.428 (marginal). Per-horizon: pl96=0.3674 (-0.0009 vs baseline), pl192=0.4193 (-0.0039), pl336=0.4552 (+0.0019), pl720=0.4708 (-0.0014). Params: 0.18M→0.28M (+55%). MAE stable: 0.435→0.4344.
