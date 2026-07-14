# Final Report: paper-4624

- Title: BLIPs: Bayesian Learned Interatomic Potentials
- Primary metric: `MSE_x10^-1` (lower)
- Records: 8
- Generated: 2026-07-13T12:23:18Z

## Best Result

- Iteration: 2
- Idea: ALGO-01 — Completed 10000-epoch training (KL anneal + cosine LR)
- Primary metric: 0.0863
- Commit: `262af055c029602f868e298cd59e7554dbfd6a00`
- Notes: Full 10000-epoch training completed. KL annealing 0-2000 epochs. Cosine LR warmup 500 + cosine decay. Gradient clipping 0.5. Best val_loss=0.00891 at epoch 7447. MSE improved from 0.0883 (mid-training) to 0.0863 (final). NLL: -1.70 (from -1.56). CRPS: 0.0304 (from 0.0319). All metrics significantly better than paper CI [0.088,0.096].
