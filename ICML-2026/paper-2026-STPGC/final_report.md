# Final Report: paper-2026

- Title: Scalable Topology-Preserving Graph Coarsening: Concepts and Algorithms
- Primary metric: `Accuracy` (higher)
- Records: 10
- Generated: 2026-07-07T18:21:47Z

## Best Result

- Iteration: 6
- Idea: CODE-P2-04 — Coarsening Parameter Tuning: θ1=25
- Primary metric: 82.72
- Commit: `c8524b97d118ca8651c690fa4a9d50c9f258a0de`
- Notes: Changed θ1 (degree threshold) from 15 to 25 for Cora at 0.5 ratio. More aggressive strong collapse produces better coarsened graph. ave_acc: 0.8272 +/- 0.0039. New best: 82.72 vs previous best 82.70 (θ1=15).
