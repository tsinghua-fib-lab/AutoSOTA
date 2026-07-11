# Final Report: paper-3396

- Title: Multicalibration Yields Better Matchings
- Primary metric: `Utility_Improvement` (higher)
- Records: 8
- Generated: 2026-07-10T22:00:20Z

## Best Result

- Iteration: 5
- Idea: IDEA-03 — Eps annealing 0.5->0.1 over 2048 iterations
- Primary metric: 0.08337
- Commit: `876a93fb783d0ad0eafd752ca2bba9a741aa8c83`
- Notes: BREAKTHROUGH: Utility_Improvement=0.08337 (+24.2% over baseline 0.06715). eps annealed from 0.5 to 0.1 over 2048 iters. Utility_Gap=-0.00797 (negative = calibrated beats best grid, BETTER than positive). MSE_Reduction=0.01244 (slightly below baseline 0.01502 due to trade-off). Paper reference: 0.085. N=10 runs all completed.
