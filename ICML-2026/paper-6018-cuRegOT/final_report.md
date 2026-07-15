# Final Report: paper-6018

- Title: cuRegOT: A GPU-Accelerated Solver for Entropic-Regularized Optimal Transport
- Primary metric: `Marginal_Error` (lower)
- Records: 10
- Generated: 2026-07-14T09:36:10Z

## Best Result

- Iteration: 9
- Idea: ALGO-07 — Increase density to 0.10 with sparsity_pattern_cycle=5
- Primary metric: 5.06740965e-12
- Commit: `46b86287e8d5cb99626ca8188b2c643428435c84`
- Notes: Increased density to 0.10 with sparsity_pattern_cycle=5, candidate_sinkhorn_iter=10, CODE-01. With ~192K Hessian elements and twice-frequency pattern updates, the Newton steps achieve near-perfect accuracy. Achieved best Marginal_Error (5.07e-12 vs baseline 9.45e-11, 18.7x improvement). All 10 seeds converge to e-12 accuracy. This approaches double-precision numerical limits for a 1.92M-element transport plan. Runtime increased to 273s but the accuracy gain is substantial.
