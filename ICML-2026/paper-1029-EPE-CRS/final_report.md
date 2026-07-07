# Final Report: paper-1029

- Title: Subgroup Discovery with the Cox Model
- Primary metric: `EPE` (lower)
- Records: 12
- Generated: 2026-07-06T04:40:43Z

## Best Result

- Iteration: 7
- Idea: ALGO-07 — Multi-feature Cox: CD4+karnofsky adjustment — breakthrough
- Primary metric: 0.329
- Commit: `e0903d395f41d2f1002485d4d079816f101a2286`
- Notes: Breakthrough: adding CD4 (immune marker) alongside karnofsky as Cox adjustment covariates. EPE improved 13.2% (0.379→0.329). C-Index improved to 0.856. Size increased to 0.168. Rej@10% dropped to 0.000 (perfect model fit within subgroup). All guardrails satisfied. Pareto improvement on all 4 metrics. Next: try 3 or 4 adjust features.
