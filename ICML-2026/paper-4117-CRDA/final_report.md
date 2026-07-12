# Final Report: paper-4117

- Title: Counterfactual Residual Data Augmentation for Regression
- Primary metric: `aug_mse` (lower)
- Records: 13
- Generated: 2026-07-11T12:16:40Z

## Best Result

- Iteration: 9
- Idea: IDEA-RESIDUAL-CLIP — Residual winsorizing + feature clipping
- Primary metric: 0.000755
- Commit: `d72fcccea2f467397dc44a7423033b19c3700f5c`
- Notes: New best! aug_mse=0.000755 (-1.0% vs baseline 0.000763). Residual winsorizing at 3-std reduces noise from poorly-fit models, making counterfactuals more robust. Combined with feature clipping.
