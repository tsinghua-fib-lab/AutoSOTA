# Final Report: paper-3779

- Title: MACD: Model-Aware Contrastive Decoding via Counterfactual Data for Video-LLMs
- Primary metric: `Precision` (higher)
- Records: 13
- Generated: 2026-07-10T18:09:23Z

## Best Result

- Iteration: 2
- Idea: ALGO-01 — Adaptive alpha via entropy scaling (scale=0.5)
- Primary metric: 0.813
- Commit: `b56325d2e2c578f93895d21fb4a3305eb8c5107e`
- Notes: ALGO-01: adaptive_alpha = base_alpha * (1 + 0.5 * entropy). Precision +0.9pp. Recall -0.7pp within 3% tolerance. F1/Accuracy maintained. Entropy-based scaling makes CD more aggressive on uncertain tokens — suppresses false positives.
