# Final Report: paper-6058

- Title: NExT-Guard: Training-Free Streaming Safeguard without Token-Level Labels
- Primary metric: `F1` (higher)
- Records: 7
- Generated: 2026-07-14T16:29:34Z

## Best Result

- Iteration: 3
- Idea: IDEA-001-topk-22 — K=22 + cal_ratio=0.85 (optimal feature count)
- Primary metric: 87.8
- Commit: `41393eac4bcac51e9daa4c754615f43868ececa3`
- Notes: K=22 features with cal_ratio=0.85 gives F1=87.80% (+2.47pp vs baseline). K=22 is the sweet spot: enough features to capture diverse safety patterns, few enough to avoid noisy/low-quality features. Top features: more selective than K=32, focusing on most discriminative SAE features.
