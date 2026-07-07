# Final Report: paper-804

- Title: CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels
- Primary metric: `Top-1 Accuracy` (higher)
- Records: 9
- Generated: 2026-07-05T09:37:11Z

## Best Result

- Iteration: 5
- Idea: CODE-02+ALGO-04 — Prompt ensemble + TTE (FiveCrop)
- Primary metric: 77.8
- Commit: `1756d9fa790d3c443b9e1df505e672987a692eb4`
- Notes: Combined prompt ensemble text init + FiveCrop TTE: 77.8% (+1.0pp over 76.8% baseline). many: 78.8%, med: 78.0%, few: 76.2%. Rare case improvement: few +1.7pp vs baseline.
