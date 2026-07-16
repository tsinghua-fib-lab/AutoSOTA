# Final Report: paper-3828

- Title: Beyond Explicit Edges: Robust Reasoning over Noisy and Sparse Knowledge Graphs
- Primary metric: `EM` (higher)
- Records: 8
- Generated: 2026-07-16T03:12:04Z

## Best Result

- Iteration: 6
- Idea: CODE-1+ALGO-5+ALGO-3+PARAM-1 — Dual-path confidence selection + source text + wider INSES search (WINNER)
- Primary metric: 0.62
- Commit: `aa71ffec1c43c3d2fdccb8e129ab6feb48e85026`
- Notes: 50-sample run. BEST RESULT: EM +0.06 vs baseline (0.56→0.62), LLM Judge +0.06 (0.52→0.58). Winning combo: dual-path confidence selection + source text in completeness checks + similarity_top_k=5.
