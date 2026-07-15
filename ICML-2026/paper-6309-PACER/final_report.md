# Final Report: paper-6309

- Title: PACER: Acyclic Causal Discovery from Large-scale Interventional Data
- Primary metric: `SHD` (lower)
- Records: 12
- Generated: 2026-07-15T05:41:44Z

## Best Result

- Iteration: 9
- Idea: COMBO-2 — ALGO-6 (Two-Stage) + PARAM-1 (MC=500 in Stage 1)
- Primary metric: 13.0
- Commit: `2383c0aa62f6df35e2e552fed9a550f8bb1f032a`
- Notes: BEST YET! ALGO-6 two-stage + PARAM-1 MC=500. SHD=13 (-31.6% vs baseline 19). FP reduced from 14 to 7 (-50%). Combined variance reduction enables Stage 1 to find an even sparser DAG. FDR=0.3889, F1=0.6286. All guardrails satisfied: TPR=0.6471 > 0.55, FDR=0.3889 < 0.65, F1=0.6286 > 0.50.
