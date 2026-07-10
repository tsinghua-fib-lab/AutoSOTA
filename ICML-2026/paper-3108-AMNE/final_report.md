# Final Report: paper-3108

- Title: Aggregate Models, Not Explanations: Improving Feature Importance Estimation
- Primary metric: `R2 Score` (higher)
- Records: 5
- Generated: 2026-07-10T02:31:05Z

## Best Result

- Iteration: 1
- Idea: CODE-01 — max_iter=2000, n_iter_no_change=50, validation_fraction=0.2
- Primary metric: 0.6053
- Commit: `833717e0dde27b7ea4f9ad2b0863b4334aacccf0`
- Notes: Increased max_iter from 500 to 2000 with n_iter_no_change=50 and validation_fraction=0.2. R2 +5.6%, MSE -28.7%, ROC AUC +4.4%. All metrics improved. Asymptotic LOCO took 3799s (63min), total 97min.
