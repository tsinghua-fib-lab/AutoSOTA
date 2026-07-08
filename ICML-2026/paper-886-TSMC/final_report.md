# Final Report: paper-886

- Title: Twice Sequential Monte Carlo for Tree Search
- Primary metric: `avg_episode_return` (higher)
- Records: 12
- Generated: 2026-07-06T12:13:18Z

## Best Result

- Iteration: 11
- Idea: EXTEND-FINAL — Full paper budget: 1250 iterations (10.24M steps) - approaching paper value
- Primary metric: 94.16
- Commit: `16a88d2c3ffef2a7de4c8da91ac77c5cedf98f7c`
- Notes: FULL PAPER BUDGET: 1250 iterations (10.24M steps). FINAL BEST: avg_episode_return=94.16 (+64.71 over baseline 29.45, +219.7%!). Within ~6% of paper's ~100 at 1e7 steps. Max return hit 131.0. Learning curve: step 750=82.98, step 1000=90.80, step 1250=94.16. Combined improvements: twisted KL proposal + Q-transform + 8 particles/depth 12 + larger model (CNN 16+32, MLP 256x256) + full 1250 iters. 11 optimization iterations total.
