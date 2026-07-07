# Final Report: paper-159

- Title: How to Correctly Report LLM-as-a-Judge Evaluations
- Primary metric: `Kendall's τ` (higher)
- Records: 8
- Generated: 2026-07-04T18:23:37Z

## Best Result

- Iteration: 3
- Idea: CODE-01+CODE-02+ALGO-03-optimal-q — q0=0.85 q1=0.82 + heterogeneous spread=0.02 + per-model + shrinkage
- Primary metric: 0.971
- Commit: `eb79e337520f5d3c281cef80eb3b9426ed21fe5c`
- Notes: Higher baseline q0=0.85 q1=0.82, heterogeneous spread=0.02, per-model + shrinkage. tau=0.971 (+0.075 vs reproduction baseline). Exact=78.0%. 9/100 corrected>naive.
