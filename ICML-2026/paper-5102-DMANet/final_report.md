# Final Report: paper-5102

- Title: Anti-Aliasing Matters: A Dynamic Network for Time Series Forecasting
- Primary metric: `MSE` (lower)
- Records: 13
- Generated: 2026-07-12T19:51:49Z

## Best Result

- Iteration: 12
- Idea: PARAM-kernel5 — kernel_size=5 with batch_size=16 + dropout=0.05 + weight_decay
- Primary metric: 0.4263
- Commit: `6398df7eb8057f3213482c8c1c6f00734f9a8794`
- Notes: kernel_size=5 (wider receptive field) on top of IDEA-01+IDEA-08+batch_size=16+dropout=0.05. NEW BEST! MSE 0.4269->0.4263. Big improvement on pred_len=720 (0.4655->0.4629) from larger kernels capturing longer-range dependencies. Overall 1.71% below baseline (0.4337->0.4263). Per-horizon: 96=0.3643, 192=0.4182, 336=0.4600, 720=0.4629.
