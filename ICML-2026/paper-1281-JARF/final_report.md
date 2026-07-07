# Final Report: paper-1281

- Title: A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth
- Primary metric: `Pearson Correlation` (higher)
- Records: 11
- Generated: 2026-07-06T17:05:56Z

## Best Result

- Iteration: 9
- Idea: CODE-3c — Hybrid subset: 7 top-gamma + 1 very-low-quality judge
- Primary metric: 0.9981
- Commit: `b42e0756a49914c6d8ac27b209dc19f6e9e8084e`
- Notes: 7 top-gamma judges + DeepSeek-R1-Distill (gamma=0.11). Pearson: 0.9981 (+0.0034 vs baseline 0.9947). Spearman: 0.9958 (+0.0038 vs baseline 0.9920). Weighted > unweighted on both: P 0.9981 > 0.9964, S 0.9958 > 0.9945. Best result satisfying all policies.
