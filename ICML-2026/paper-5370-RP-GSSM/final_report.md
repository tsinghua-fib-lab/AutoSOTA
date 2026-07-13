# Final Report: paper-5370

- Title: Maximum-Likelihood Learning of Latent Dynamics Without Reconstruction
- Primary metric: `Linear Regression R²` (higher)
- Records: 12
- Generated: 2026-07-13T06:09:25Z

## Best Result

- Iteration: 10
- Idea: IDEA-11-extended — AdamW + Cyclical beta + 12000 iters
- Primary metric: 0.185
- Commit: `89f6f4a1e5e576ba77b2134e8be5a7b4e0a060f3`
- Notes: Extended to 12000 iterations on AdamW+cyclical beta. Linear R² NEW BEST: 0.185 (+7% vs 10000 iter, +130% vs baseline). KRR R² 0.894 (slight decline from 0.9093 at 10k). Linear continues to improve monotonically with more iterations; KRR shows diminishing returns. Primary metric improvement justifies this trade-off.
