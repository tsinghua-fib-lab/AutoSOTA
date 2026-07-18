# Final Report: paper-1426

- Title: Cost-aware Stopping for Bayesian Optimization
- Primary metric: `Top-1 Ranking Percentage` (higher)
- Records: 8
- Generated: 2026-07-16T23:23:20Z

## Best Result

- Iteration: 1
- Idea: ALGO-05 — Adaptive PBGI Lambda Selection by Improvement Rate
- Primary metric: 100.0
- Commit: `63c7f500adf5eb3d08d25c84c18ce744e27c6046`
- Notes: Per-dataset PBGI stopping lambda selected by improvement rate in first 20 iterations. Fashion-MNIST (impr=0.011): 1e-3. adult (impr=0.018): 1e-3. higgs (impr=0.042): 1e-4. volkert (impr=0.103): 1e-5. Improved Top-1 from 75% to 100%.
