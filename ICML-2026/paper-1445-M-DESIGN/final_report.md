# Final Report: paper-1445

- Title: Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design
- Primary metric: `Accuracy` (higher)
- Records: 11
- Generated: 2026-07-06T21:29:11Z

## Best Result

- Iteration: 6
- Idea: ALGO-5-NOEST — No estimator + epsilon-greedy + optimized params
- Primary metric: 89.54
- Commit: `bc987b5dc8f6328189c5384b9c7894e971da9a1d`
- Notes: Removed --use_estimator flag. Without ECC predictor noise, Bayesian optimization found better architecture: edge_index + rel_lepe + mean + add + 4 + node_adaptive = 89.54% (+-0.32%). Fewer layers (4 vs 6) with node_adaptive aggregation. +1.04pp above baseline.
