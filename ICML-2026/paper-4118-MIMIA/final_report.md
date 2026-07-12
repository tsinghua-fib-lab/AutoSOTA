# Final Report: paper-4118

- Title: The Hidden Risk: Membership Inference Attacks on Multimodal Federated Learning via Modality Imbalance
- Primary metric: `TPR_0.1pct_FPR` (higher)
- Records: 12
- Generated: 2026-07-11T15:40:08Z

## Best Result

- Iteration: 8
- Idea: PARAM-TEMP — AffineGapMIA temperature sweep (T=0.3 best)
- Primary metric: 11.79
- Commit: `37fd8fc`
- Notes: T=0.3 gives TPR@0.1%FPR=11.79 (+19.2% vs baseline 9.89). All metrics improved over baseline. Harder gate (T=0.3 vs 0.5) sharpens feature importance, helping extreme-FPR discrimination. T=1.0 too soft (1.90%).
