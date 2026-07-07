# Final Report: paper-1063

- Title: Token-Free Hierarchical Indexing for RAG beyond LLM-based Summarization
- Primary metric: `Str-Acc` (higher)
- Records: 9
- Generated: 2026-07-06T06:32:22Z

## Best Result

- Iteration: 5
- Idea: ALGO-2 — k=7 + ALGO-2 adaptive coarse threshold (factor=0.3)
- Primary metric: 0.81
- Commit: `ac7f168560985425f47c1fa0c0a8896003adfbb1`
- Notes: ALGO-2: Adaptive threshold in _coarse_grained_matching() — only expand communities with sim >= max_sim * 0.3. Combined with k=7. Str-Acc 81.0% (best so far, +2.3pp from 78.7% baseline), LLM-Acc 83.6%. Adaptive filtering reduces noise from marginally relevant communities.
