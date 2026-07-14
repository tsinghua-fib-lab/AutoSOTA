# Final Report: paper-4380

- Title: No Need to Train Your RDB Foundation Model
- Primary metric: `AUC` (higher)
- Records: 13
- Generated: 2026-07-13T21:50:22Z

## Best Result

- Iteration: 10
- Idea: ALGO-07 — n_estimators=32 with SUM aggregation
- Primary metric: 0.9009
- Commit: `4ec357e649ca20420cb75cf2273cb31900a726cc`
- Notes: Combined n_estimators=32 with SUM aggregation. AUC=0.9009 vs previous best 0.8997 (+0.0012). The interaction is synergistic: more estimators amplify the benefit of additional features. Broke the 0.90 threshold.
