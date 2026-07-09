# Final Report: paper-2558

- Title: Minimizing Upper Confidence Bounds: A Data-Driven Framework for Stochastic Programming
- Primary metric: `Iteration` (lower)
- Records: 14
- Generated: 2026-07-08T17:06:33Z

## Best Result

- Iteration: 6
- Idea: ALGO-01-B — Multi-Alpha CVaR Cuts (0.05/0.1/0.15/0.2) + Expected-Value Cut
- Primary metric: 6.367
- Commit: `a7e1571cdf16476a017cc82b66c5b639c40d38b5`
- Notes: ALGO-01-B: extended complementary cut with multi-alpha CVaR cuts at alpha=[0.05, 0.1, 0.15, 0.2] plus expected-value cut. Each alpha level captures a different tail quantile of the recourse distribution, forming a piecewise-linear lower bound approximation. Iteration reduced from 9.17 to 6.37 (-30.5%). Time reduced from 4.51s to 2.64s (-41.5%). APUB-M now converges faster than SAA-M (6.37 vs 7.97 iterations). This is a novel contribution combining risk-aware cutting planes at multiple confidence levels.
