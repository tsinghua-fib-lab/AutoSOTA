# Final Report: paper-3824

- Title: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models
- Primary metric: `exact_match` (higher)
- Records: 13
- Generated: 2026-07-16T03:12:23Z

## Best Result

- Iteration: 8
- Idea: CODE-3 — selection_temperature=0.7, lambda=2.2, beta=2.8
- Primary metric: 30.58
- Commit: `0f12992092bccff1044a502a0b34bbea0d52bb00`
- Notes: selection_temperature=0.7 → 30.58%% (+1.56 vs baseline 29.02%%, exceeds paper 30.02%%). sel_T controls distribution sharpness within kept set. 0.7 is the sweet spot.
