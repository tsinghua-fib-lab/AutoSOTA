# Final Report: paper-4765

- Title: AdaSCALE: Adaptive Scaling for OOD Detection
- Primary metric: `FPR@95` (lower)
- Records: 8
- Generated: 2026-07-12T05:31:47Z

## Best Result

- Iteration: 7
- Idea: ALGO-01 — AdaSCALE-A + AdaSCALE-L geometric mean score fusion
- Primary metric: 30.53
- Commit: `d906485b6019a3b8179ca47a09bb01ab7a44a2f5`
- Notes: Geometric mean fusion: conf_fused = sqrt(conf_a * conf_l). FPR95 30.53 massive improvement (baseline 52.28, -21.75). AUROC 90.53 (baseline 82.27, +8.26). ssb_hard: 54.19, ninco: 6.88. Far-OOD FPR95 7.23. Score fusion leverages complementary OOD signals from feature-level (A) and logit-level (L) scaling.
