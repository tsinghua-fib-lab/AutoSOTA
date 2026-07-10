# Final Report: paper-3566

- Title: Rank-Learner: Orthogonal Ranking of Treatment Effects
- Primary metric: `AUTOC` (higher)
- Records: 15
- Generated: 2026-07-10T08:37:38Z

## Best Result

- Iteration: 14
- Idea: 3566-CODE-01 — XL nuisance + ranker wd=5e-5 (fine-tuning)
- Primary metric: 1.3373
- Commit: `df7831f4a96e132b4381e2991f2db61e5be55321`
- Notes: XL nuisance config with slightly higher ranker weight_decay (5e-5 vs 1e-5): AUTOC=1.3373 (essentially same as iter 10 1.3371). weight_decay tuning has minimal effect.
