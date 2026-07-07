# Final Report: paper-1650

- Title: Steering Large Language Models through the DMTA Cycle: Structure-Based Drug Design via Knowledge-Driven Bi-Level Thompson Sampling
- Primary metric: `Top1_Dock_Avg` (lower)
- Records: 13
- Generated: 2026-07-07T04:55:45Z

## Best Result

- Iteration: 10
- Idea: PARAM-3 — Midpoint SA penalty: lambda_qed=5, lambda_sa=7 on 10-variant ensemble
- Primary metric: -10.39
- Commit: `df6ac2996bd5359b8d6b4cc86f87ea541729d2e3`
- Notes: Similar to lambda_sa=8 (-10.38). The penalty surface is flat between lambda_sa=7-8. Confirms -10.38/-10.39 is the optimal trade-off point for the 10-variant ensemble with linear penalty.
