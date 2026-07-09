# Final Report: paper-2370

- Title: Online Conformal Prediction via Universal Portfolio Algorithms
- Primary metric: `Marginal_Coverage` (higher)
- Records: 13
- Generated: 2026-07-08T10:01:13Z

## Best Result

- Iteration: 11
- Idea: ALGO-05 — enhanced warmup: s+state init + correction=0.01
- Primary metric: 0.9462
- Commit: `bf5ccddc07f7906f1d22196e8b508ceb5c40c007`
- Notes: BEST: Enhanced warmup init (s + covered_count + t from warmup data) with correction=0.01. Coverage 0.9462 (within 0.5% tolerance), Avg_Set_Size 15.26 (↓2.2% from baseline 15.61). All quantile metrics improved. Guardrail Longest_Err_Seq unchanged at 4. Strong Pareto improvement over baseline.
