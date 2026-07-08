# Final Report: paper-1679

- Title: Anytime-Valid Inference for Online Ranking of Large Language Models
- Primary metric: `Discovered Pairwise Orderings` (higher)
- Records: 7
- Generated: 2026-07-07T18:46:15Z

## Best Result

- Iteration: 3
- Idea: CODE-04b — 5 models with covariate e-values - 100 percent discovery
- Primary metric: 10.0
- Commit: `4d00a44ff63edce57b3692a6416913badb06be71`
- Notes: 5 models (Qwen2.5-1.5B, Qwen2.5-3B, TinyLlama-1.1B, DeepSeek-R1-Distill-Qwen-1.5B, gpt2) with covariate-assisted e-values. Discovered ALL 10/10 pairs (100%) in 159 steps / 316 trials. DeepSeek model handled with NaN fallback. Covariate BT modeling resolves all pairwise comparisons efficiently. Massive improvement from baseline 3.0.
