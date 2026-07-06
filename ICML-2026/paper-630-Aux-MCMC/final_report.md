# Final Report: paper-630

- Title: Markov Chain Monte Carlo without Evaluating the Target: an Auxiliary Variable Approach
- Primary metric: `PoissonMALA_ESS_s_Min` (higher)
- Records: 7
- Generated: 2026-07-05T07:48:39Z

## Best Result

- Iteration: 1
- Idea: CODE-01,CODE-02,CODE-03 — Diag precision + precompute L_over_lamM/LM + inline AliasSampler
- Primary metric: 1.7834
- Commit: `fba163f4f95a3bdc3ffaa7bb94d7eefaf362d6c4`
- Notes: Combined CODE optimizations: diagonal precision in grad_mala/grad_barker, precomputed L_over_lamM and LM in Params, inlined AliasSampler in quick_poisson. Single-round eval at rate 0.55 (round 200). PoissonMALA Min ESS/s improved from baseline 1.147 to 1.7834. Acceptance rates: PoissonMH 57%, PoissonMALA 55%.
