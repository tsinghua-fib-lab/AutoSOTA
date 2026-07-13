# Final Report: paper-5057

- Title: Variance Driven Exploration: A Provable and Efficient Methodology for Pure Exploration in Highly Stochastic Environments
- Primary metric: `Error Probability (VarDE_lse-0.1)` (lower)
- Records: 13
- Generated: 2026-07-12T16:02:08Z

## Best Result

- Iteration: 12
- Idea: ALGO-05d — variance_exponent=0.70 with ema_alpha=0.10 (OPTIMAL)
- Primary metric: 6.725
- Commit: `0623a32a8ecb21af0dc85d80b14477bdc1038576`
- Notes: FINAL BEST: ve=0.70 + ema_alpha=0.10 + adaptive floor. Error: 6.725%. Total improvement: 7.34%->6.725% = -0.615pp (8.4% rel). Optimal config: tau=0.10, ve=0.70, ema_alpha=0.10, adaptive floor. CR-A stable at 9.725%.
