# Final Report: paper-2614

- Title: Loss-aware distributionally robust optimization via trainable optimal transport ambiguity sets
- Primary metric: `e_oos` (lower)
- Records: 14
- Generated: 2026-07-08T16:38:44Z

## Best Result

- Iteration: 7
- Idea: PARAM-02 — Lambda=1.0 penalization + data-driven L init + best-checkpoint
- Primary metric: 8.1483
- Commit: `00bd7c2813d91a11bb98def2b79e6d741e127c63`
- Notes: PARAM-02: lambda=1.0. Best checkpoint at iter 3700: e_oos=8.1483 (marginally best), e_wc=7.0099 (excellent, -14.6% vs baseline 8.2119). Lower lambda allows more aggressive e_wc minimization without significantly hurting e_oos. Data-driven L init dominates e_oos; lambda affects computational cost (more iterations needed with low lambda).
