# Final Report: paper-5494

- Title: Adaptive Multiscale Binary Expansion Tests for Independence
- Primary metric: `Power` (higher)
- Records: 8
- Generated: 2026-07-13T12:40:17Z

## Best Result

- Iteration: 5
- Idea: CODE-5494-01 — K=5 n_folds=20 hyperparameter combo
- Primary metric: 0.914
- Commit: `b9f44e54b2d66d383984dfe03309ed080633634f`
- Notes: CODE-1: K=5 with n_folds=20 at b=0.1. Power=0.914 > K=4 baseline 0.880 (+3.9%). Type_I=0.064 within guardrail. At b=0.2: Power=0.970 < K=4 baseline 0.982. Trade-off: K=5 better for weak signals, K=4 better for strong signals. n_folds effect is minor within MC noise.
