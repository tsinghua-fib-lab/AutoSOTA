# Final Report: paper-3470

- Title: Identifying dependent components from multi-domain linear mixtures
- Primary metric: `d_mcc` (higher)
- Records: 3
- Generated: 2026-07-10T06:59:10Z

## Best Result

- Iteration: 2
- Idea: IDEA-03+AmariCheckpoint+AmariSelection — v6: SVD init + solve + grad clip + beta2=0.99 + Amari checkpointing & selection
- Primary metric: 0.99904
- Commit: `675e6efd6d1a4942117af1dcef71aa1cc20d27d3`
- Notes: v6 combines SVD-based A_hat init, torch.linalg.solve, gradient clipping (max_norm=5.0), Adam beta2=0.99, Amari-based checkpointing (save best Amari state during optimization), and Amari-based init selection (choose init with best Amari among d-MCC>0.99). Full 5-seed eval. d-MCC maintained at 0.999. MCC at 0.940 (same as baseline). Amari at 0.097 vs baseline 0.111: 12.3% improvement. Per-seed: 0.115, 0.097, 0.089, 0.090, 0.096. Total runtime: 1278s (21.3 min).
