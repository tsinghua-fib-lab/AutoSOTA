# Final Report: paper-1507

- Title: Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding
- Primary metric: `Acc_flexible_extract_pct` (higher)
- Records: 5
- Generated: 2026-07-07T23:29:16Z

## Best Result

- Iteration: 4
- Idea: CODE-1,CODE-2,CODE-3,ALGO-1 — Combined: All 4 fixes (reverify_count + margin_confidence + max_remask + multiplicative scoring)
- Primary metric: 79.15
- Commit: `107747468f0abac0f87a5e0a77a81f9277d4ce79`
- Notes: Combined CODE-1+CODE-2+CODE-3+ALGO-1. Best Acc=79.15% (+1.52pp vs 77.63 baseline). All fixes work synergistically.
