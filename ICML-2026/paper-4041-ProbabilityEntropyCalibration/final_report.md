# Final Report: paper-4041

- Title: Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning
- Primary metric: `Pass@1` (higher)
- Records: 4
- Generated: 2026-07-16T09:14:13Z

## Best Result

- Iteration: 2
- Idea: PARAM-1-LR3e5 — Lower LR: lr=3e-5 cosine
- Primary metric: 91.2
- Commit: `14663c9e99259758c4a255eb2d1304a5c56b84c1`
- Notes: Best result! lr=3e-5 improves both Pass@1 (68.79, +0.98%) and Pass@16 (91.2, +0.6%) over baseline (67.81/90.6). Lower LR provides better convergence with RankTuner loss on short 1-epoch training.
