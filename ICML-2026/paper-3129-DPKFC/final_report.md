# Final Report: paper-3129

- Title: DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning
- Primary metric: `Test Accuracy` (higher)
- Records: 9
- Generated: 2026-07-15T19:44:27Z

## Best Result

- Iteration: 8
- Idea: idea-1+2+6 — EMA + alpha_schedule 2.0->0.5 + precond_steps=5: Combined Fisher optimization (Ideas 1+2+6)
- Primary metric: 95.6
- Commit: `370778d78bce589ee93f9c6e1901dc4fb22aea44`
- Notes: Combined best techniques: EMA Fisher accumulation (cov_ema_decay=0.95) + alpha schedule 2.0->0.5 over 5 precond_steps. 95.60% (+0.64% vs baseline 94.96%). Synergistic improvement: EMA smooths + scheduling captures coarse-then-fine + multi-step reduces variance. Best result across all iterations.
