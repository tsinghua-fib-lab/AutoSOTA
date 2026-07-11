# Final Report: paper-3432

- Title: LoBCD-GW: A Fast and Data-Dependent Algorithm for Computing Gromov-Wasserstein Distance via Localized Block Coordinate Descent
- Primary metric: `Accuracy` (higher)
- Records: 13
- Generated: 2026-07-10T05:56:08Z

## Best Result

- Iteration: 12
- Idea: IDEA-02,03,04,08,10,12 — rho=0.12 + adaptive T_full + warmup=25 + sinkhorn=6 + checkpoint + num stab
- Primary metric: 98.73
- Commit: `896ccb44ac55b5f82fdea7df2d0177f3133fd9e2`
- Notes: Accuracy 98.73% matches iter-11 with faster time 0.588s/graph (-21% vs baseline). rho=0.12-0.15 is the optimal range. All optimizations combined: adaptive T_full + warmup=25 + sinkhorn_iters=6 + numerical stabilization + best checkpointing.
