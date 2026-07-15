# Final Report: paper-6081

- Title: RSA-CP: Efficient Conformal Prediction in Small-Sample Regimes via Random Score Alignment
- Primary metric: `Coverage_RSA_CP_OT_alpha_0.05` (higher)
- Records: 11
- Generated: 2026-07-14T09:34:23Z

## Best Result

- Iteration: 9
- Idea: CODE-008 — Combined: prior_scale=0.5 + real_weight=1.1
- Primary metric: 0.9334
- Commit: `c8c83b12b3ac2cddb8737f0e5ca6f8b8e794cf2a`
- Notes: CODE-008: Combined weaker Beta-Binomial prior (prior_scale=0.5) with weighted real scores in conformal quantile (real_weight=1.1). Best result: Coverage 0.9334 (closest to 0.95 target), Set Size 6.75 (+8.2% vs baseline 6.24, within 20% tolerance). Both components contribute independently: prior_scale widens rank windows, real_weight increases global quantile cap.
