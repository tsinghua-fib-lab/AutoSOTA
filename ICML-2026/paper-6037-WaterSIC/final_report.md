# Final Report: paper-6037

- Title: WaterSIC: Information-Theoretically (Near) Optimal \\Linear Layer Quantization
- Primary metric: `PPL` (lower)
- Records: 3
- Generated: 2026-07-16T17:27:59Z

## Best Result

- Iteration: 2
- Idea: IDEA-01 — Enable qronos_adapt + attn_weighted adapt features
- Primary metric: 10.447
- Commit: `db2b3519ebdec8d9fcf074c9feffb0468a221b70`
- Notes: Enabled qronos_adapt, attn_weighted_qkv, attn_weighted_adapt_eps_joint, and w1w3_qronos_adapt during re-quantization. Pre-FT PPL improved from 10.69 to 10.573 (+0.12). Post-FT PPL: 10.447, exceeding paper SOTA 10.45 and improving baseline 10.51 by 0.064. Avg rate: 2.9999 bits. Quantization took 4h15m. This is the paper-native mechanism that was dropped in the baseline reproduction.
