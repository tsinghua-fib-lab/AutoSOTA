# Final Report: paper-4965

- Title: Multimarginal flow matching with optimal transport potentials
- Primary metric: `W1` (lower)
- Records: 7
- Generated: 2026-07-16T17:08:31Z

## Best Result

- Iteration: 6
- Idea: ITER6-w500 — w=500 (midpoint) + best config: slope=24, epochs=200, curriculum fixed, sampling=100
- Primary metric: 0.4413
- Commit: `5ec0a4f51430c40cfb477cff89b7e11a6a2bc6c9`
- Notes: Reduced w from 1000 to 500. Slight improvement over Iter 5 (0.4413 vs 0.4416). Both folds improved marginally. Suggests w=500 may be a better sweet spot than w=1000 for this 5D problem, though the difference is within noise. Best epochs: 60 for both folds.
