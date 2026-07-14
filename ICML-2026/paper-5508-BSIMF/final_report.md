# Final Report: paper-5508

- Title: A Bayesian Approach to Quantify the Uncertainty of Human Ratings in a Single-Instance Multimodal Framework
- Primary metric: `R_X_sq` (higher)
- Records: 7
- Generated: 2026-07-13T22:55:16Z

## Best Result

- Iteration: 1
- Idea: CODE-01 — Fix z90 from 1.26 to 1.645 for two-sided 90% CI
- Primary metric: 0.9961
- Commit: `1df8913f26467ec39d069fa16a4c166e3b7d4e56`
- Notes: Fixed hardcoded z90=1.26 (one-sided 80% quantile) to _z_for_central_coverage(0.90)=1.645 (two-sided 90% central interval). PICP90 improved from 0.809 to 0.901 (exceeds 0.90 target). R_X_pred_sq improved from 0.679 to 0.701. MPIW90 increased from 11.912 to 15.370 (expected, wider intervals for correct coverage). RMSE_X and RMSE_Y both improved. Best epoch: 67/100.
