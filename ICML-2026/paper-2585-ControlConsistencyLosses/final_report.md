# Final Report: paper-2585

- Title: Control Consistency Losses for Diffusion Bridges
- Primary metric: `kl_to_solution` (lower)
- Records: 7
- Generated: 2026-07-15T13:41:59Z

## Best Result

- Iteration: 7
- Idea: COMBO-001 — Combo: STL + EMA 0.999 + traj_batch 128
- Primary metric: 0.019362
- Commit: `877a814ef70b27bb31eb687bdde7443b50fc63ee`
- Notes: Combining STL_adjustments + ema_rate=0.999 + traj_batch_size=128. BEST RESULT: KL 0.0194 vs baseline 0.0514 (62% improvement). 5 runs for statistical reliability.
