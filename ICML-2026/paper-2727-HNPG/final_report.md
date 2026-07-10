# Final Report: paper-2727

- Title: Hyperbolic neural population geometry benefits computation
- Primary metric: `accuracy` (higher)
- Records: 10
- Generated: 2026-07-09T12:43:52Z

## Best Result

- Iteration: 7
- Idea: ALGO-5 — EMA + GELU Embedder
- Primary metric: 85.82
- Commit: `cc9673d262e44b64f1f095eaf4f2f8f24f4dfe75`
- Notes: ALGO-5 on GELU: EMA weight averaging decay=0.999. Result 85.82 pct, NEW BEST. +0.58 over baseline 85.24, +0.24 over GELU alone. Trials: 86.09, 87.13, 83.58, 84.77, 87.55. Std 1.65 pct best yet. EMA smooths optimization, helps worst seeds. Trial 5 achieved 87.55 pct single-run. Exceeds paper 85.52 by +0.30 pct.
