# Final Report: paper-694

- Title: QuITE: Query-Based Irregular Time Series Embedding
- Primary metric: `MSE` (lower)
- Records: 9
- Generated: 2026-07-06T16:46:26Z

## Best Result

- Iteration: 1
- Idea: CODE-1 — Gradient Clipping + Epoch-Level Training Metrics
- Primary metric: 0.01816
- Commit: `055372e6cef8081bc3fd53014e699bece87260c4`
- Notes: Added grad_clip_norm=1.0, accumulated training loss across all 113 batches for epoch-level reporting. Best epoch 301 with patience=50. MSE -1.94%, MAE -2.13% vs baseline.
