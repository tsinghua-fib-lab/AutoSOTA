# Final Report: paper-1261

- Title: Transformed Latent Variable Multi-Output Gaussian Processes
- Primary metric: `MSE` (lower)
- Records: 13
- Generated: 2026-07-06T08:48:07Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — M=350 + cosine annealing LR (0.02->1e-5)
- Primary metric: 0.053
- Commit: `70da2e2cce07218a447b1fe40594cea375b52cb9`
- Notes: M=350 + cosine annealing LR (lr=0.02 to 1e-5 over 1000 epochs). Test MSE=0.053 (52% reduction from baseline 0.111). Train MSE=0.020. Test NLL=0.581 (within guardrail <=1.124). Cosine annealing enables larger early steps for fast convergence + fine-tuning late. Loss values stabilized at 220K-240K (vs 280K-1M for baseline).
