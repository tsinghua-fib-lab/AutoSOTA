# Final Report: paper-3124

- Title: Towards the Training of Deeper Predictive Coding Neural Networks
- Primary metric: `Test Accuracy` (higher)
- Records: 7
- Generated: 2026-07-10T07:43:52Z

## Best Result

- Iteration: 5
- Idea: CODE-3-v2 — 60 epochs with iter-3 schedule (warmup=15%, peak=1.3x, MixUp+LS+grad_clip)
- Primary metric: 0.9065
- Commit: `fbc3dd3104e3abc8897ca30545f621469156638a`
- Notes: 60 epochs with warmup=15%, peak=1.3x, end=0.05x, grad_clip=1.0, MixUp α=0.4, LS=0.1. avg=90.65% (±0.11%). Per-seed: [90.80, 90.47, 90.73, 90.57, 90.67]. +0.12pct over iter-3 (50 epochs). Extra epochs with well-calibrated schedule improve convergence.
