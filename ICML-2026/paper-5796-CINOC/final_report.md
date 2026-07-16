# Final Report: paper-5796

- Title: CINOC: Cardinality-Invariant Neural Operator Policies for Scalable PDE Control
- Primary metric: `Tracking MSE` (lower)
- Records: 14
- Generated: 2026-07-14T05:57:13Z

## Best Result

- Iteration: 12
- Idea: ALGO-05 — Best noise (0.05/0.01) + 2000 epochs
- Primary metric: 2.3e-05
- Commit: `81b687c25db77618c043529dc0bc53e141afad1f`
- Notes: BEST RESULT. Fixed noise (u=0.05, z=0.01) with 2000 epochs. Tracking MSE=2.3e-5 improves 50% over baseline (4.6e-5). Tight 2-Sigma (±1.2e-5) indicates stable convergence. 2.0x improvement over paper result. Longer training with fixed noise provides substantially better regularization than paper zero-noise default.
