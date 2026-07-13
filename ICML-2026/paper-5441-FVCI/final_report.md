# Final Report: paper-5441

- Title: Function-Valued Causal Influence in Nonlinear Time Series
- Primary metric: `Pearson_r` (higher)
- Records: 12
- Generated: 2026-07-13T10:07:38Z

## Best Result

- Iteration: 11
- Idea: idea-12e — Optimal Capacity: maxlags=3, hidden_nodes=128, hidden_layers=2
- Primary metric: 0.9883
- Commit: `51f3a314f29087e32b0a8f1c5dd25700d1c20c4f`
- Notes: BEST RESULT: +0.0198 vs baseline (0.9685 -> 0.9883). Lowest std (0.0024). Hidden_nodes=128 gives marginal improvement over 64 (0.9883 vs 0.9878). The key improvements: maxlags=3 (temporal context), hidden_layers=2 (depth), hidden_nodes=64-128 (width).
