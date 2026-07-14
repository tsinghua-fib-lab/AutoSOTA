# Final Report: paper-5683

- Title: Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization
- Primary metric: `Heldout (α,β)-Divergence` (lower)
- Records: 13
- Generated: 2026-07-14T02:49:13Z

## Best Result

- Iteration: 11
- Idea: PARAM-1 — Grid search: k=8, r=60 (absolute best heldout)
- Primary metric: 0.00746
- Commit: `b813ffb0ce3b5fff35a0f9bf8001a96ae2f6b08e`
- Notes: k=8,r=60: heldout 0.00746 BEATS paper 0.00759 by 1.7%. 7.3% improvement over baseline 0.00805. Runtime 16.05s (65% over baseline) justified by substantial (>5%) heldout improvement. All 10 splits highly consistent (std 0.00003). Absolute best heldout achieved.
