# Final Report: paper-6233

- Title: OC-space: a Unifying Perspective on Verification of Tree Ensembles
- Primary metric: `OC_Tree_Verification_Time_ms` (lower)
- Records: 14
- Generated: 2026-07-14T23:50:36Z

## Best Result

- Iteration: 11
- Idea: PARAM-04c — Adjust OCTree max depth to 8
- Primary metric: 0.011727
- Commit: `a164ca1be919d6bbac53ec56669fb2c623e3ab07`
- Notes: Set OCTree max depth to 8 (between original 10 and tuned 5). Geometric mean 0.011727 ms — best so far. Model 44 improved from 2.436ms (depth 5) to 0.877ms. Model 6 stable at 0.155ms. Depth 8 balances small and large tree-count models. Cumulative: 0.011727 ms vs baseline 0.0197 ms (40.5% faster).
