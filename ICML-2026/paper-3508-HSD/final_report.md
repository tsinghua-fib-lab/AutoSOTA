# Final Report: paper-3508

- Title: Topology-Preserving Neural Operator Learning via Hodge Decomposition
- Primary metric: `Grad_Fid` (higher)
- Records: 7
- Generated: 2026-07-10T06:51:56Z

## Best Result

- Iteration: 3
- Idea: A-02 — Learned Diagonal Spectral Projection (mode_weights)
- Primary metric: 0.991153
- Commit: `2532115953c4e785f9a823655c30194b76b1c6d9`
- Notes: C-01 + C-02 + A-02 combined. A-02 adds learned per-mode diagonal weights (896 params) with L2 reg (1e-5). Massive improvements: MSE -49% from baseline (0.00464->0.00236), Energy_Fid +1.99%, Spec_Fid +3.49%, Grad_Fid +0.96%. Enst_Fid slightly regressed vs iter 2 (-0.6%) but still +12.3% above baseline. All guardrails safe.
