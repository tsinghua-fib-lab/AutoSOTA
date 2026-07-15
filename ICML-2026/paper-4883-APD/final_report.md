# Final Report: paper-4883

- Title: Automatic Pruning Discovery for Large Language Models
- Primary metric: `perplexity` (lower)
- Records: 10
- Generated: 2026-07-12T16:40:36Z

## Best Result

- Iteration: 9
- Idea: ABLATE-01 — SparseGPT without EC nsamples=256
- Primary metric: 6.702
- Commit: `be695d28c67c5d0098c0708257a988c80e44105a`
- Notes: Ablation: removed EC from SparseGPT. ppl=6.7020 is BETTER than SparseGPT+EC (6.7628). EC is redundant/harmful because SparseGPT OBS already compensates pruned weights.
