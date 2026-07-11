# Final Report: paper-3199

- Title: Learning High-Dimensional Parity Functions with Product Networks using Gradient Descent
- Primary metric: `Validation Accuracy` (higher)
- Records: 11
- Generated: 2026-07-09T23:16:06Z

## Best Result

- Iteration: 9
- Idea: CODE-09+ALGO-05 — Nesterov + LR=0.10 at N=16: convergence step 25, coverage 0.0046
- Primary metric: 100.0
- Commit: `a3fdd44449b2061cc80b912f812a6890030e3971`
- Notes: N=16. Nesterov (0.9) + LR=0.10. 100% at step 25 (vs baseline step 1000). Coverage 0.0046 (vs baseline 0.0281). 40x faster convergence, 6.1x lower coverage. Loss at step 100: 0.0005.
