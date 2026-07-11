# Final Report: paper-4219

- Title: Masked Multi-path Contrast with Confidence-Gated Semantic Imputation for Incomplete Multi-view Clustering
- Primary metric: `ACC` (higher)
- Records: 10
- Generated: 2026-07-11T09:05:15Z

## Best Result

- Iteration: 5
- Idea: beta-0.005 — Loss weight tuning: beta=0.005 (half semantic alignment weight)
- Primary metric: 90.24
- Commit: `1ae961c72dfdb8baaca1f1cd62c36582f3c57b78`
- Notes: Beta=0.005 reduces semantic alignment weight by 50%. Improvement over baseline (90.06): ACC +0.18pp, NMI +0.42pp, ARI +0.43pp. At 70% missing, only ~9% of sample pairs have both views visible, making the semantic alignment signal unreliable at full weight. Individual: seed=10 ACC=89.88, seed=20 ACC=90.87, seed=30 ACC=89.96.
