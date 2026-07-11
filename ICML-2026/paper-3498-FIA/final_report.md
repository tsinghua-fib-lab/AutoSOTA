# Final Report: paper-3498

- Title: Fairness in Aggregation: Optimal Top-$k$ and Improved Full Ranking
- Primary metric: `Spearman_Objective_Cost` (lower)
- Records: 10
- Generated: 2026-07-10T05:07:15Z

## Best Result

- Iteration: 9
- Idea: ID-01+ID-02+ID-04+ID-05-wider — Systematic LS with wider window (50) and more passes (30)
- Primary metric: 120312.0
- Commit: `1df2c4d8342091b603b71254923dbd675ab23694`
- Notes: Spearman: 120826 -> 120312 (-514, -0.43%). Kendall-Tau: 95181 -> 95217 (+0.04%, within tolerance). Wider window (50) and more passes found additional small improvements.
