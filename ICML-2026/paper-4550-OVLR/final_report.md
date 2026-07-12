# Final Report: paper-4550

- Title: OVLR: Efficient, Scalable, and Robust Training via Output-Level Variance-Reduced Likelihood Ratio
- Primary metric: `Accuracy` (higher)
- Records: 11
- Generated: 2026-07-11T14:06:56Z

## Best Result

- Iteration: 8
- Idea: IDEA-11 — Noise scale annealing 1.5->0.7 on top of extended warmup + transition + cosine
- Primary metric: 82.6
- Commit: `72606152428abcf6b486a85c8f32d10d748fac5c`
- Notes: Linear noise scale annealing from 1.5 to 0.7 over 40 post-warmup epochs. Best: 82.87% (epoch 46), Final: 82.60%. +0.5% over iter-5 (82.37%). Higher initial sigma (1.5 vs 1.0) provides better exploration early; lower final sigma (0.7 vs 1.0) gives more precise gradient estimates near convergence. Time: 314.5s. This is the best result across all iterations.
