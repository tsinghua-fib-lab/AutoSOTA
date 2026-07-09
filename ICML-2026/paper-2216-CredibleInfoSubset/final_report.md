# Final Report: paper-2216

- Title: Credible Information Subset Decomposition: An End-to-End Multi-fidelity Learning Model by Modeling Label Information
- Primary metric: `MAE` (lower)
- Records: 8
- Generated: 2026-07-08T08:44:43Z

## Best Result

- Iteration: 3
- Idea: ALGO-04 — Rank feature skip connection: aux predictor ensemble
- Primary metric: 0.55
- Commit: `53368344d4f0d6c262508a77507f51325b03a70f`
- Notes: ALGO-04: Added auxiliary EvidentialRegressor on rank_feat (64-dim), ensemble mu=(mu_main+mu_aux)/2, loss on ensemble directly. MAE=0.550 (-2.8% vs baseline 0.566), RMSE=0.722 (-2.7%), tau_b=0.655 (+2.8%). All three metrics Pareto-dominant improvement.
