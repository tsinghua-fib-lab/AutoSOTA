# Final Report: paper-4842

- Title: Differentially Private Range Subgraph Counting
- Primary metric: `Relative Error` (lower)
- Records: 7
- Generated: 2026-07-12T14:03:42Z

## Best Result

- Iteration: 3
- Idea: IDEA-01 — Constrained inference on range tree (Hay et al. VLDB 2010)
- Primary metric: 0.54401502
- Commit: `fd3e7c9cd9b5f2d737011c73fc64581f4759d546`
- Notes: IDEA-01: Applied Hay et al. VLDB 2010 constrained inference to enforce parent=left+right consistency in range tree dataNodes. 29.5% Relative Error reduction (0.772->0.544). Query Time also improved to 67us. Preprocessing dropped to 40s. Total time 65s. Per-run std reduced from 0.094 to 0.039 confirming variance reduction. Pareto improvement across ALL metrics.
