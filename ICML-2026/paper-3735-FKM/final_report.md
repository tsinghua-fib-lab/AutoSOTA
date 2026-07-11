# Final Report: paper-3735

- Title: Fast kernel methods: Sobolev, physics-informed, and additive models
- Primary metric: `running_time` (lower)
- Records: 8
- Generated: 2026-07-10T13:22:12Z

## Best Result

- Iteration: 5
- Idea: CODE-05 — eps=1e-4 for 2D NUFFT + 20-coarse search
- Primary metric: 10.116
- Commit: `986497dbe17c2a8b2dc8e637dec78e096a62c2fa`
- Notes: Relaxed 2D NUFFT eps further to 1e-4. Reduced coarse search from 30 to 20 lambdas (42 total validations). Test MSE preserved at 7.65e-05. 46.5% total speedup from baseline.
