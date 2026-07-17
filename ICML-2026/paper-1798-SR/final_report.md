# Final Report: paper-1798

- Title: LLM Self-Recognition: Steering and Retrieving Activation Signatures
- Primary metric: `token_level_f1` (higher)
- Records: 7
- Generated: 2026-07-17T00:21:31Z

## Best Result

- Iteration: 3
- Idea: ALGO-4+ALGO-6 — Block-sparse_2 steering + confidence-weighted voting
- Primary metric: 0.9923
- Commit: `5157d4d4271186838be5e536c18a29757cee64b7`
- Notes: FURTHER IMPROVEMENT: Token F1 99.2% vs 97.4% (iter 2). Block_sparse_2 (2 heads, 128-dim) + confidence-weighted text voting. Near-perfect 2-class discrimination.
