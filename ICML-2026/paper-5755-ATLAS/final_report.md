# Final Report: paper-5755

- Title: Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks
- Primary metric: `Ability_MAE` (lower)
- Records: 9
- Generated: 2026-07-15T18:21:42Z

## Best Result

- Iteration: 1
- Idea: CODE-08 — Negative discrimination item pre-filtering (a1 > 0.1)
- Primary metric: 0.1575
- Commit: `20b590ca860adecc0b323a2b852900ce12229259`
- Notes: Filtered 600 items with a1 <= 0.1 from the 5600-item bank before CAT. Removes noise from Fisher information. All three metrics improved: Ability_MAE 0.161->0.1575 (-2.2%), Accuracy_MAE 0.021->0.0207, Average_Items 41.0->36.5 (-11.0%).
