# Final Report: paper-3942

- Title: Hide&Seek: Learning to Explain in an End-to-End Differentiable Network
- Primary metric: `TPR` (higher)
- Records: 11
- Generated: 2026-07-10T19:39:48Z

## Best Result

- Iteration: 5
- Idea: PARAM-03 — Lambda=0.28 + warmup=100
- Primary metric: 99.21
- Commit: `0926668a1d2c0fdb8f1ac7033a2af382f5275334`
- Notes: Lambda=0.28 (slightly lower than paper 0.3) with warmup=100. Best TPR so far (99.21 vs baseline 99.02). FDR=4.40 within guardrail (<4.54). Lower lambda allows more features selected, boosting TPR. TPR distribution tightened: [98.6-99.7].
