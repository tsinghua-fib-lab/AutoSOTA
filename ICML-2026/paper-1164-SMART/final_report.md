# Final Report: paper-1164

- Title: Sample Margin-Aware Recalibration of Temperature Scaling
- Primary metric: `ECE` (lower)
- Records: 13
- Generated: 2026-07-06T15:22:41Z

## Best Result

- Iteration: 6
- Idea: PARAM-01 — Optimal sigma=0.04 (from grid search)
- Primary metric: 1.63
- Commit: `c50b321897f8a1beb117fa896e4f357512e1c9bf`
- Notes: Grid search over CharbonnierSoftECE sigma found sigma=0.04 optimal (vs default 0.05). ECE improved from 1.83 to 1.63 (-0.20%). AdaECE also improved from 2.03 to 1.92. Narrower soft bins (sigma=0.04) provide more precise ECE approximation during training. Accuracy preserved at 77.30%. Other sigma values: 0.03->2.05, 0.045->1.67, 0.06->1.76, 0.07->1.80, 0.08->2.08. Delta tuning showed no additional benefit.
