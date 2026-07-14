# Final Report: paper-5954

- Title: LLM-based Embeddings: Attention Values Encode Sentence Semantics Better Than Hidden States
- Primary metric: `NDCG@10` (higher)
- Records: 9
- Generated: 2026-07-14T05:15:30Z

## Best Result

- Iteration: 5
- Idea: ALGO-04 — Per-layer cosine-dissimilarity weighting
- Primary metric: 49.34
- Commit: `9ad0f80af864e27f01307b35932bc9996e61ce03`
- Notes: Weighted layers by cosine dissimilarity to group mean (diverse layers weighted higher). Combined with token variance weighting. NDCG improved from 49.26 to 49.34 (+0.08). Total improvement over baseline: +0.70.
