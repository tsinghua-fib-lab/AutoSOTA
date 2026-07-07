# Final Report: paper-1599

- Title: InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimate
- Primary metric: `Time` (lower)
- Records: 9
- Generated: 2026-07-07T06:58:41Z

## Best Result

- Iteration: 8
- Idea: PARAM-1 — 10 slices + HyperTr-2 + 3 perms + FP16 + early-exit
- Primary metric: 0.591
- Commit: `19edab4d3a1207a96453011e1784dd5c46c01ea3`
- Notes: Aggressive: 10 slices (down from 25) + HyperTr-2 + 3 perms + FP16 + early-exit. Time=0.591s (75.3% reduction from 2.389s baseline, 4.0x speedup). Min=0.381s. MI=0.1428 vs baseline 0.1389 (+2.8%). Higher variance across runs (0.116-0.157) due to fewer Monte Carlo slices. Best speed configuration found.
