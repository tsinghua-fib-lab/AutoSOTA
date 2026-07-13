# Final Report: paper-3153

- Title: DISSOLVR: An Interpretable and Fast Framework for Aqueous and Organic Solubility Prediction
- Primary metric: `RMSE` (lower)
- Records: 23
- Generated: 2026-07-12T16:04:55Z

## Best Result

- Iteration: 11
- Idea: IDEA-12 — Reduced l2_leaf_reg=3
- Primary metric: 0.8044
- Commit: `df3f0d75f3fb3f4354010e30f49dbbb4ccfe1682`
- Notes: l2=3 with MACCS+AUTOCORR2D features: RMSE 0.8044 beats previous best 0.8056. Best seed (42)=0.8029. Higher variance across seeds (σ=0.0043). Reduced regularization helps with the expanded feature set.
