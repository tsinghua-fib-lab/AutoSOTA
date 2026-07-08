# Final Report: paper-2138

- Title: Deep Single-Index Fréchet Regression
- Primary metric: `MPE` (lower)
- Records: 4
- Generated: 2026-07-08T01:55:58Z

## Best Result

- Iteration: 3
- Idea: C-07 — Patience=20 only (C-07, 35 seeds)
- Primary metric: 0.3126
- Commit: `5aed556827c4bd4ef7cf4e627f3eb16800be3e29`
- Notes: Only change from baseline: PATIENCE=20 (was 10). All other settings unchanged (Adam+StepLR, no grad clip). MPE=0.3126 vs baseline 0.3179. Improvement of -0.005. Simply allowing more training epochs helps with the 80-sample training set.
