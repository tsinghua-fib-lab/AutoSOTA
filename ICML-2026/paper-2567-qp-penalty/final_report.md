# Final Report: paper-2567

- Title: A Penalty Approach For Differentiation Through Black-box Quadratic Programming Solvers
- Primary metric: `Backward` (lower)
- Records: 15
- Generated: 2026-07-08T15:57:23Z

## Best Result

- Iteration: 4
- Idea: PARAM-01 — Increase beta to 1e-5
- Primary metric: 0.581
- Commit: `93dca920a88029f586fcaa0044c4cd1f050db82f`
- Notes: Beta increased from 1e-6 to 1e-5. Backward improved 8% (0.63→0.58ms). Total improved 9% (17.4→15.9ms). Larger beta improves Hessian conditioning, making factorization faster.
