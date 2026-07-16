# Final Report: paper-998

- Title: Detecting Errors in AI-Generated Annotations: When and Why Semantic Neighbors Help
- Primary metric: `SAGE_AUROC` (higher)
- Records: 7
- Generated: 2026-07-15T17:45:22Z

## Best Result

- Iteration: 3
- Idea: ALGO-3 — Top-7 by Cosine + Exclude Self
- Primary metric: 68.0
- Commit: `86e9742331b531a72f1b363439b431e5f653b776`
- Notes: Implemented top_n_by_cosine=7 in extract_score: keep only 7 nearest neighbors by cosine similarity (drop 2 farthest), combined with exclude_self=True. SAGE AUROC: 68.00% (vs 67.89% exclude_self alone, +0.11pp; vs 67.30% baseline, +0.70pp). Direct and Random unchanged (analysis-only change).
