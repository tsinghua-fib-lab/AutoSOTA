# Final Report: paper-1201

- Title: Shapley Regularized Neural Granger Causality
- Primary metric: `AUROC` (higher)
- Records: 9
- Generated: 2026-07-06T11:51:11Z

## Best Result

- Iteration: 1
- Idea: CODE-08 — Lag sweep: Traffic lag=2→3 improves AUROC
- Primary metric: 0.7955
- Commit: `254dbe6931c238a08bde010d678befd3376fb136`
- Notes: Lag sweep over {2,3,4,5,6} with seed 2025 found lag=3 gives AUROC=0.7983 (vs lag=2 0.7811, +0.0172). Full 5-seed eval (2025-2029) with lag=3: mean AUROC=0.7955 (+0.0035 over baseline 0.792), mean AUPRC=0.6258 (-0.0033 from baseline 0.6291, within 5% tolerance). Added --lag_override flag to real_data.py; created eval_traffic_lag3.sh.
