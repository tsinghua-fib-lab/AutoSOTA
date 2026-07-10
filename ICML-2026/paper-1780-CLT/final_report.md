# Final Report: paper-1780

- Title: Protein Circuit Tracing via Cross-layer Transcoders
- Primary metric: `F1` (higher)
- Records: 2
- Generated: 2026-07-09T11:13:27Z

## Best Result

- Iteration: 1
- Idea: CODE-01 — LogisticRegressionCV auto-tuned C per family
- Primary metric: 0.9026
- Commit: `9c8bcd9ced501bd73c7725e349b25b10401a9a9b`
- Notes: Replaced LogisticRegression(C=1.0) with LogisticRegressionCV(Cs=[0.001-100.0], cv=min(3,n_pos)). F1 improved from 0.8784 to 0.9026 (+0.0242, +2.76%). Clean F1 also up from 0.9330 to 0.9336. Recovery ratio 96.7% vs baseline 94.7%. 558 families evaluated.
