# Final Report: paper-5507

- Title: Multi-Way Representation Alignment
- Primary metric: `Rank-1 Accuracy (Avg)` (higher)
- Records: 10
- Generated: 2026-07-13T23:46:05Z

## Best Result

- Iteration: 9
- Idea: CODE-3+CODE-1 — SVD init + gc_tau=0.06 + gc_lam=0.9 (higher consensus)
- Primary metric: 0.7057
- Commit: `03bc66e60dd6dfd719cd72c04d4d45b883590436`
- Notes: BEST on both metrics. Avg +0.08pp, Worst +0.52pp vs baseline. CAMEM->SPBER (bottleneck) improved from 0.5928 to 0.5980 (+0.52pp). SVD init + tighter trust (gc_tau=0.06) + stronger consensus (gc_lam=0.9) is the winning combination. GCPA maintains 1.03pp margin over GCCA.
