# Final Report: paper-4196

- Title: An Exterior Method for Nonnegative Matrix Factorization
- Primary metric: `Reconstruction Error` (lower)
- Records: 10
- Generated: 2026-07-11T06:03:20Z

## Best Result

- Iteration: 8
- Idea: PARAM-6 — Reduced ADMM max_iter 1500→1000
- Primary metric: 12557.4778
- Commit: `f4ef17892bf3794b71c7eb59a58ce68b17ca304b`
- Notes: Reduced ADMM max_iter from 1500 to 1000. Rotation time: 8.6s. Rotation quality remains sufficient — HALS needs 22.6s to reach target. Total Runtime: 34.9s (vs baseline 103.6s, -66.3%). RE unchanged at 12557.48. Combined optimizations: alpha=1.5, rho=10 with rho_mode=1, max_iter=1000, Nesterov HALS beta=0.5.
