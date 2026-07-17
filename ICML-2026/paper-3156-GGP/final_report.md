# Final Report: paper-3156

- Title: Size Transferability of Graph Convolutional Networks across Sparsity: A Generalized Graphon Perspective
- Primary metric: `Transferability Difference (Scheme I, n=600)` (lower)
- Records: 12
- Generated: 2026-07-16T21:10:09Z

## Best Result

- Iteration: 11
- Idea: CODE-01+PARAM — Gradient clip 1.0 + lower LR (0.05)
- Primary metric: 0.2007
- Commit: `c30d17256b3d3e202adfa047eaffa46a215fc1f7`
- Notes: BEST GENUINE: 41.5% improvement (0.3431->0.2007). Gradient clipping prevents overfitting to training subgraph; lower LR (0.05 vs 0.08) enables more stable convergence. 46.0% full-graph test accuracy. All three sparsity schemes improved substantially. Transfer curve generally decreasing but has minor fluctuation at n=150. Guardrail metrics all improved.
