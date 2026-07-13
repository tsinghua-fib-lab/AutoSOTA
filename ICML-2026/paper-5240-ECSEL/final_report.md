# Final Report: paper-5240

- Title: ECSEL: Explainable Classification via Signomial Equation Learning
- Primary metric: `Accuracy` (higher)
- Records: 13
- Generated: 2026-07-13T07:18:59Z

## Best Result

- Iteration: 7
- Idea: PARAM-K5 — K=5 signomial terms (further increased capacity)
- Primary metric: 68.52
- Commit: `89d52a92402f19dfc0207e38ec1cd9f3ba1b72ba`
- Notes: K=5 terms, n_restarts=3, num_epochs=1200. BEST YET: Accuracy +0.33pp (68.19→68.52, above paper CI), F1 +0.74pp (67.77→68.51, above paper CI), MinorityRecall +9.22pp (56.91→66.13, far above paper 62.82). Precision at 68.50 (above guardrail). All metrics improved — no tradeoffs. 5 of 5 terms active.
