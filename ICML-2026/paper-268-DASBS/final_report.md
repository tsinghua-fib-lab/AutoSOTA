# Final Report: paper-268

- Title: Discrete Adjoint Schrödinger Bridge Sampler
- Primary metric: `ΔCorr` (lower)
- Records: 9
- Generated: 2026-07-05T10:00:01Z

## Best Result

- Iteration: 5
- Idea: CODE-1+ALGO-3+EXTENDED — 8 stages with t2 bregman + gradient clipping
- Primary metric: 0.001925
- Commit: `53c698d3f1097573c2fe3527452979216abc0f2f`
- Notes: Extended to 8 training stages (from 5) with t2 bregman + gradient clipping. BEST ΔCorr (0.001925) — beats paper-implied value (0.0023). ΔMag=0.00343 (better than iter3). EW2=2.24 (slightly worse than iter3-implied 2.00 but far below paper-implied 5.4). More stages improved correlation accuracy at slight EW2 cost. All three metrics excellent.
