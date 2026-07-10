# Final Report: paper-3099

- Title: Incorporating Importance Weighting in Optimal Transport Based Domain Alignment
- Primary metric: `Accuracy` (higher)
- Records: 3
- Generated: 2026-07-09T23:55:57Z

## Best Result

- Iteration: 1
- Idea: ALGO-1+CODE-4 — BatchNorm1d in MLP hidden layers
- Primary metric: 93.37
- Commit: `8d71438dfed6cb03f2573a51d93e345a13abe07b`
- Notes: Added BatchNorm1d after fc0 and fc1 in MLP. Best run 93.37% (seed 3), mean 91.15%, std 3.80%. Per-run: [93.07, 92.13, 84.40, 93.37, 92.78]. Baseline was 87.89% best, 81.98% mean. Config flag use_batchnorm=True. Massive improvement in stability - only 1/5 runs below 90%.
