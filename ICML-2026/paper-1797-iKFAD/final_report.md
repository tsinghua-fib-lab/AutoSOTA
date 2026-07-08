# Final Report: paper-1797

- Title: Adaptive Momentum and Nonlinear Damping for Neural Network Training
- Primary metric: `Test Loss` (lower)
- Records: 13
- Generated: 2026-07-07T12:50:25Z

## Best Result

- Iteration: 9
- Idea: PARAM-01 — Schedule: h_min=0.08, alpha_min=0.8, warmup=500
- Primary metric: 1.5802
- Commit: `7018a3a6f05a151cc1dd643ea0cb84bf5e140fbf`
- Notes: h_min=0.08, alpha_min=0.8, warmup=500. Best val=1.5802 at step 4900. -0.0599 vs baseline (-3.7%). Alpha_min=0.8 beats alpha_min=1.0 by -0.0046.
