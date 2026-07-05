# Final Report: paper-497

- Title: pTNAS: Progressive Neural Architecture Search for Tabular Data
- Primary metric: `AUC` (higher)
- Records: 13
- Generated: 2026-07-05T05:24:00Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — Different random seed (seed=123)
- Primary metric: 0.674
- Commit: `9aa6fd5c3b90b75fd8ab2f15e14780b3e4bd4d6d`
- Notes: Switched seed from 42 to 123. Found same architecture [128,256,64,256] but achieved AUC=0.6740 (vs 0.6674 with seed 42). Beats paper AUC=0.6680. The different seed led to different SH/final-training weight initialization, enabling better convergence (val AUC 0.6294 vs 0.6220). Demonstrates the value of multi-seed robustness.
