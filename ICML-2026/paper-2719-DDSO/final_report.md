# Final Report: paper-2719

- Title: From Individual Calibration to Reliable Classifiers: ALD Parameterization with mPAIC Guarantees
- Primary metric: `ACC` (higher)
- Records: 13
- Generated: 2026-07-09T10:07:20Z

## Best Result

- Iteration: 11
- Idea: final — FINAL: t=0.25 + BatchNorm verification
- Primary metric: 60.667
- Commit: `3bb4c0843c2d6849ce32f48e5c465a46b1284d56`
- Notes: Final verification of best state (t=0.25 + BN). ACC=60.667 (+1.0pp over baseline 59.667, +0.667pp over paper 60.000). ECE=0.094 stable vs baseline 0.092. KCE improved (0.344 vs 0.351). GroupX_ECE improved (0.335 vs 0.349). All guardrails within tolerance. Best configuration: ICALD_Classifier with BatchNorm1d, t=0.25, lambda_reg=0.9.
