# Final Report: paper-1168

- Title: Same Graph Cross-Task Transfer in GNNs: Protocols and Predictors
- Primary metric: `NC_joint_test_acc` (higher)
- Records: 13
- Generated: 2026-07-07T06:33:47Z

## Best Result

- Iteration: 12
- Idea: CODE-1-V3 — Balanced gradient clipping (max_norm=2.0) with LP dropout 0.5 and extended epochs
- Primary metric: 75.12
- Commit: `01bac2d1e71c1302113c1d0667f9c0f103432b9e`
- Notes: Grad clip max_norm=2.0, LP dropout 0.5, extended epochs 400/100. Best balanced result: NC 75.12% (+0.26pp vs baseline, above paper 75.1%), LP 90.34% (-0.12pp vs baseline, above paper 90.2%). Both core metrics exceed paper reported values.
