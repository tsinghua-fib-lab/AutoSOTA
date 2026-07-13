# Final Report: paper-3143

- Title: Deep Multi-view Graph Clustering via Attribute-aware Bidirectional Structural Refinement and Pseudo-label Guided Multi-level Fusion
- Primary metric: `ACC` (higher)
- Records: 10
- Generated: 2026-07-12T13:37:11Z

## Best Result

- Iteration: 1
- Idea: CODE-P0-10 — Cosine Annealing LR Schedule
- Primary metric: 0.9527
- Commit: `c68d5403f00a2b4632b8b368785924925b1a306c`
- Notes: Added CosineAnnealingLR scheduler with T_max=epoch, eta_min=1e-6. ACC improved from 0.9514 to 0.9527 (+0.0013). All guardrails preserved. Metrics parsed from stdout Final Result line.
