# Final Report: paper-4553

- Title: Semi-knockoffs: a model-agnostic conditional independence testing method with finite-sample guarantees
- Primary metric: `Power` (higher)
- Records: 7
- Generated: 2026-07-11T18:48:33Z

## Best Result

- Iteration: 3
- Idea: ALGO-5 — BH procedure with FDR=0.05 for feature selection + n_perm=5
- Primary metric: 1.0
- Commit: `a3888b12451549af051dde105a18ea579653f86d`
- Notes: ALGO-5: Benjamini-Hochberg procedure with FDR=0.05. Power maintained at 1.0000. Type-I Error dramatically reduced 0.0479->0.0147. BH adapts threshold per-run based on p-value distribution, controlling FDR across 50 features. Best result so far.
