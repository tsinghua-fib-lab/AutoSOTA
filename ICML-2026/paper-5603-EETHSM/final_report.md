# Final Report: paper-5603

- Title: Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models
- Primary metric: `Accuracy` (higher)
- Records: 17
- Generated: 2026-07-08T07:19:29Z

## Best Result

- Iteration: 15
- Idea: IDEA-003,IDEA-012 — Larger model: hidden_size=8 heads=2 num_examples=2000
- Primary metric: 0.17939
- Commit: `fc9037ae4c59b598b9f4f2c46f015524e032580e`
- Notes: BEST RESULT: hidden_size=8 (heads=2) with num_examples=2000. Accuracy 0.179, 108% improvement over baseline (0.086). Larger model (~2500 params) significantly improves selective copy capability.
