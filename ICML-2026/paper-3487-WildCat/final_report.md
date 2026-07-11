# Final Report: paper-3487

- Title: WildCat: Near-Linear Attention in Theory and Practice
- Primary metric: `IS Degradation (%)` (lower)
- Records: 14
- Generated: 2026-07-10T11:02:48Z

## Best Result

- Iteration: 7
- Idea: IDEA-04 — beta=2 sampling + r=160 B=8
- Primary metric: -0.1109
- Commit: `fe388293b34ac5a348cb8f2701b1feaeea465566`
- Notes: IDEA-04 beta=2 adaptive sampling with r=160 B=8. IS Degradation -0.11% NEGATIVE: WildCat BEATS exact attention on IS! FID Degradation -0.39%. Speed-up 1.02x. Steinerberger 2024: beta=2 sampling gives better Frobenius-norm contraction per step. 5-seed eval.
