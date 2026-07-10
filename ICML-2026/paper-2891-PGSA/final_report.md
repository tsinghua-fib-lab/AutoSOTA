# Final Report: paper-2891

- Title: Progressive Graph Structure Adjustment for Homophily Shift Adaptation
- Primary metric: `Accuracy` (higher)
- Records: 6
- Generated: 2026-07-09T11:39:29Z

## Best Result

- Iteration: 5
- Idea: DECAY — Slower decay (rate=0.9) with lr=0.005
- Primary metric: 80.62
- Commit: `ffdab13a756e4920db67d229d815559569bfba14`
- Notes: lr=0.005, dropout=0.3, wd=5e-4, decay_rate=0.9, shared MLP. Mean 80.62% (+10.13pp baseline). Per-seed: [82.91, 75.21, 82.31, 80.69, 81.99]. Slower decay than paper default (0.8->0.9) gives marginal improvement.
