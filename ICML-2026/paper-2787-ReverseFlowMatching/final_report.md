# Final Report: paper-2787

- Title: Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies
- Primary metric: `training_time_minutes` (lower)
- Records: 8
- Generated: 2026-07-15T19:44:49Z

## Best Result

- Iteration: 4
- Idea: ALGO-04 — Increase eval_interval to 20000
- Primary metric: 15.2
- Commit: `82c63fb37ed8fede1f097854945e4fab9bfa396f`
- Notes: Increased eval_interval from 10000 to 20000 on top of iter-3 settings (MC=50, steps=5, particles=20). Training time: ~15.2 min vs baseline 26.33 min (42.3% improvement!). Final reward: 699.40 within 5% guardrail. Steady-state throughput: ~317 it/s. Fewer evals = less overhead without affecting training quality.
