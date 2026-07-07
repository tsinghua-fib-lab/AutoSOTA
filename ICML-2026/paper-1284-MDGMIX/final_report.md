# Final Report: paper-1284

- Title: MDGMIX: Boundary-Aware Subgraph Mixing for Multi-Domain Graph Pre-Training
- Primary metric: `Accuracy` (higher)
- Records: 13
- Generated: 2026-07-06T08:23:44Z

## Best Result

- Iteration: 6
- Idea: PARAM-1 — Cosine annealing LR schedule
- Primary metric: 44.74
- Commit: `e59629345ca0f4bc58ecc03c470f1c67d7e4c480`
- Notes: Added CosineAnnealingWarmRestarts with T_0=50 T_mult=2 eta_min=1e-5. Result Micro=44.74 (+0.44pp vs baseline 44.30). First improvement! LR schedule helps escape local minima in the 1-shot loss landscape.
