# Final Report: paper-3227

- Title: Zero-Flow Encoders
- Primary metric: `AUC` (higher)
- Records: 9
- Generated: 2026-07-09T20:21:53Z

## Best Result

- Iteration: 4
- Idea: ZFE-003+ZFE-004 — Paper Config Corrections: lr=1e-4, l1_lambda=3e-9
- Primary metric: 0.9784
- Commit: `da2f9448a373bbda5e3ac242f9b28d5f3f17dfac`
- Notes: Corrected lr from 1e-3 to 1e-4 and l1_lambda from 1e-9 to 3e-9 to match paper Appendix D. On top of Conv1dEncoder + Beta(4,4). AUC 0.9784 vs 0.9764 (+0.0020). Individual seeds: 0.9478-0.9920. L1 penalty values more stable (~17K-20K final), zero-flow penalty smaller (~0.0005). Lower lr may need more training iterations to fully converge.
