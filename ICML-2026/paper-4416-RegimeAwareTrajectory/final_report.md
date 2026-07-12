# Final Report: paper-4416

- Title: A Regime-Aware Trajectory Prediction Framework for 1000+ Systems Biology Models
- Primary metric: `MSE` (lower)
- Records: 8
- Generated: 2026-07-11T23:26:53Z

## Best Result

- Iteration: 5
- Idea: I-01+I-02+I-05 — LR scheduler + Beta timestep + num_steps=2 (epoch 129)
- Primary metric: 0.00994
- Commit: `b57b4dba391622e6ef40a001c64e1b5b89c5e736`
- Notes: Resumed from epoch 79 with Beta(0.5,0.5) timestep distribution. Trained to epoch 129 (val_mse=0.0098). Eval with num_steps=2 on seed=42 test split. MSE -51% from baseline (0.0203→0.00994). MAE 0.0427 beats paper 0.044. CRPS=MAE because pred_n_samples=1. All three improvements (num_steps, LR scheduler, Beta timestep) compound.
