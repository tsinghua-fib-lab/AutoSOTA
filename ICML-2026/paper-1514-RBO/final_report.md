# Final Report: paper-1514

- Title: Robust Bayesian Optimisation with Unbounded Corruptions
- Primary metric: `Cumulative Regret` (lower)
- Records: 14
- Generated: 2026-07-07T21:48:26Z

## Best Result

- Iteration: 6
- Idea: PARAM-outer-plateau-0.2 — Tighter outer plateau width (0.5->0.2)
- Primary metric: 181.31
- Commit: `c78771f476663108ab783026ace5a9b5f8a6c10f`
- Notes: First improvement! Tightening outer plateau_width from 0.5 to 0.2 makes the model more skeptical of latest observations, preventing corruption-induced exploration away from x=1.0. (181.31 vs 185.84 baseline, -2.4%). Within paper CI.
