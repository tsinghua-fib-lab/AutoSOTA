# Final Report: paper-2403

- Title: AReaL-DTA: Dynamic Tree Attention for Efficient Reinforcement Learning of Large Language Models
- Primary metric: `training_throughput_k_tok_per_s` (higher)
- Records: 9
- Generated: 2026-07-08T18:02:08Z

## Best Result

- Iteration: final
- Idea: ALL — Final verification: all optimizations + BLOCK_SIZE=256
- Primary metric: 12.14
- Commit: `f5182467464e28185405e84f452f9e8bd6eeac31`
- Notes: Final verification with all optimizations: ALGO-1 (high prefix sharing, tree_tokens=4096), ALGO-2 (max_autotune), ALGO-3 (no gradient checkpointing), ALGO-4 (BLOCK_SIZE=256). Dense: 12135.8 tok/s (12.14 K tok/s, +68.9% vs baseline 7.19), Tree: 27253.2 tok/s (27.25 K tok/s, +535% vs baseline 4.29), Speedup: 2.25x. Memory: 9.34 GB (within 15 GB guardrail).
