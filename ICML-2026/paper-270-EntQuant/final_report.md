# Final Report: paper-270

- Title: Float8@2bits: Entropy Coding Enables Data-Free Model Compression
- Primary metric: `c4_perplexity` (lower)
- Records: 13
- Generated: 2026-07-05T11:48:39Z

## Best Result

- Iteration: 12
- Idea: PARAM-1 — PARAM-1: LAMBDA=6.0 (best λ found)
- Primary metric: 7.2227
- Commit: `8478e08fa18219a26f01d0383b8e567b0f2c1a07`
- Notes: PARAM-1 sweep final. LAMBDA=6.0 → C4 PPL 7.2227. 4.3% better than baseline 7.5491. Represents ~4-bit quality. Trade-off: higher bit rate for better perplexity. Strong λ→PPL monotonic relationship confirmed.
