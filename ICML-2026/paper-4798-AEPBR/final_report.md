# Final Report: paper-4798

- Title: Approximate Equivariance via Projection-Based Regularisation
- Primary metric: `Test Accuracy` (higher)
- Records: 7
- Generated: 2026-07-12T12:17:20Z

## Best Result

- Iteration: 5
- Idea: ALGO-03 — SO3 group c=2 capacity increase
- Primary metric: 0.9455
- Commit: `3e8c0f67ff462ab3812a59c1082530e6b8ecb45a`
- Notes: Switched from O(3) to SO(3) with c=2 (2.03M params). Max test acc=0.9455 (+1.83% over baseline 0.9272). Exceeds paper reported O3 result of 0.943. SO3 half-sized Q matrices enable c=2 training. Best epoch: 50.
