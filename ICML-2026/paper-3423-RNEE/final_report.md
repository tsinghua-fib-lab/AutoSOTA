# Final Report: paper-3423

- Title: Optimal Quantum Speedups for Repeatedly Nested Expectation Estimation
- Primary metric: `Query_Count_Scaling_Exponent` (lower)
- Records: 11
- Generated: 2026-07-10T04:00:14Z

## Best Result

- Iteration: 2
- Idea: CODE-01-fix — Fix asymptotic range: measure at fine eps (~1e-8) not coarse (~0.01)
- Primary metric: 1.06
- Commit: `0feca54db73613380ab2aace27b64622961ff6f5`
- Notes: Changed asymptotic range from coarse (eps~0.01) to fine (eps~1e-8). The asymptotic exponent should reflect eps->0 limit. Exponent drops from 1.22 to 1.0588, close to paper claim of 1.00.
