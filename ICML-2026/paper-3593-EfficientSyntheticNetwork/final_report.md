# Final Report: paper-3593

- Title: Efficient Synthetic Network Generation via Latent Embedding Reconstruction
- Primary metric: `Clus_RMSE` (lower)
- Records: 9
- Generated: 2026-07-18T02:38:34Z

## Best Result

- Iteration: 8
- Idea: CODE-06 — r=6 with Z shrinkage 10% - BEST: Clus_RMSE=0.01322 (-45.9%)
- Primary metric: 0.0132167
- Commit: `0295787b2d30bd400ad44e906d1eca9e094871ff`
- Notes: BEST RESULT. r=6 pre-fit LSM + shrink Z by 10% (multiply by 0.90), recalibrate sparsity via binary search. ALL four metrics simultaneously improved: Clus_RMSE=0.01322 (-45.9% vs 0.02445 baseline), Tri_RMSE=7.92e-05 (-5.3%), DegC_KS=0.03286 (-13.6%), Eig_MMD=0.03550 (-20.8%). Method: post-hoc L2 regularization by shrinking latent positions Z, prevents overfitting to training graph triangle patterns while preserving degree distribution through sparsity recalibration. SyNG-R remains distribution-free (instant inference).
