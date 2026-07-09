# Final Report: paper-2754

- Title: Conformal Policy Control
- Primary metric: `FDR` (lower)
- Records: 9
- Generated: 2026-07-09T07:26:55Z

## Best Result

- Iteration: 7
- Idea: PARAM-1 — PARAM-1: cal_frac=0.8 gives FDR=0.0489 Recall=0.9867 (small +0.0007 gain)
- Primary metric: 0.0489
- Commit: `8a7266c66ee7b9772a8c1687c565f758d7eefbe9`
- Notes: PARAM-1: cal_frac sweep with best blend (0.3,0.3,0.4). cal_frac=0.8 gives best tradeoff: FDR=0.0489, Recall=0.9867. All cal_frac values produce FDR <= 0.05 with Recall >= 0.985. The default cal_frac=0.7 is near-optimal; cal_frac=0.8 yields marginal recall improvement (+0.0007). Built on CODE-2+ALGO-2.
