# Final Report: paper-2343

- Title: Von Mises-Fisher Mixture Model with Dynamic Shrinkage for Realistic Test-Time Transduction
- Primary metric: `Top-1 Accuracy` (higher)
- Records: 13
- Generated: 2026-07-08T13:26:27Z

## Best Result

- Iteration: 11
- Idea: PARAM-05 — Maximum prior weight: alpha=3.0, lambda_y_hat=2.0
- Primary metric: 91.61
- Commit: `f1ef6226debed4e3fe05e5089644a9c281ef33b1`
- Notes: Increased lambda_y_hat to 2.0 with alpha=3.0. MOON: 91.61% (+8.78pp over baseline). The trend is clear: at very low Keff (1-4), stronger prior guidance (higher lambda_y_hat) consistently improves accuracy.
