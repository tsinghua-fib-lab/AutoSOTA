# Final Report: paper-1810

- Title: Variable Clustering via Distributionally Robust Nodewise Regression
- Primary metric: `ami` (higher)
- Records: 14
- Generated: 2026-07-07T12:14:17Z

## Best Result

- Iteration: 12
- Idea: ALGO-06b — Consensus KMeans with 20 seeds
- Primary metric: 0.9661
- Commit: `71aa66a061ca7674d99d2429108d06533da0ebbf`
- Notes: Increased consensus KMeans seeds from 10 to 20. AMI improved from 0.9639 to 0.9661 (+0.23%%). More seeds give more robust co-association matrix. Cumulative from baseline: 0.9135->0.9661 (+5.76%% AMI). Combined changes: CODE-04 (L2 norm) + ALGO-06b (20-seed consensus) + affinity thresholding (50pct).
