# Final Report: paper-911

- Title: Credit-assigned Policy Gradient for Early Stage Retrieval in Two-stage Ranking
- Primary metric: `Policy_Value_TOP1-PG_K50_50K` (higher)
- Records: 9
- Generated: 2026-07-06T19:02:45Z

## Best Result

- Iteration: 4
- Idea: ALGO-2+ALGO-4 — dim_emb=16 + n_moe=4 (1-seed screening)
- Primary metric: 7.4086
- Commit: `0c473530cb1d38bb28a39d7be71e4bc7cb782a63`
- Notes: ALGO-2+ALGO-4: dim_emb=16, n_moe=4. Single seed = 7.4086 (+15.7% over baseline 6.40). Better than n_moe=4 alone (7.15) and dim=32 alone (6.41). Best single-seed result so far.
