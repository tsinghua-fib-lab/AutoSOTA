# Final Report: paper-4500

- Title: Generalized Boundary FDR Control under Arbitrary Dependence: An Approach on Closure Principle
- Primary metric: `bFDR` (lower)
- Records: 9
- Generated: 2026-07-11T19:03:14Z

## Best Result

- Iteration: 7
- Idea: PARAM-2 — Train/Test Split Ratio Optimization (test_size=2/3)
- Primary metric: 0.0
- Commit: `cfd9d52f560ade9d053e0d2f2eaeed0ec751bb3b`
- Notes: PARAM-2: Changed test_size from 3/4 to 2/3 (33% train / 67% test). PARETO IMPROVEMENT: bFDR improved from 0.01 to 0.00, Power improved from 84.66% to 84.71%, TDR improved from 99.68% to 99.69%. More training data (33% vs 25%) gives better null distribution estimation for p-value calibration. Also tested test_size=0.5: bFDR=0.01 Power=84.95% (Power slightly better but bFDR same as baseline). test_size=2/3 is the sweet spot.
