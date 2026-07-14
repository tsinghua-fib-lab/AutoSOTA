# Final Report: paper-4290

- Title: Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking
- Primary metric: `TPR` (higher)
- Records: 7
- Generated: 2026-07-13T16:41:03Z

## Best Result

- Iteration: 1
- Idea: A1 — Adaptive Per-Token Logit Biasing
- Primary metric: 1.0
- Commit: `2a5e3258a427338cf5c2ae64a25c2f8b27c944d9`
- Notes: Replaced fixed delta=6.0 with per-token adaptive delta based on base model probability mass on target tokens. Clamped to [3.0, 12.0]. All 200/200 watermarked texts detected after 10% synonym substitution. FPR=0.0 on 400 unwatermarked+natural texts.
