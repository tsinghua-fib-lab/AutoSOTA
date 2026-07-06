# Final Report: paper-463

- Title: Noise-Robust Density Estimation for Tabular Data Anomaly Detection
- Primary metric: `AUROC` (higher)
- Records: 7
- Generated: 2026-07-04T22:21:17Z

## Best Result

- Iteration: 5
- Idea: ALGO-04 — KL divergence regularization (kl_weight=0.01) + ensemble inference
- Primary metric: 92.03
- Commit: `aba6a2d8941d6dcb8f6a77a57fc9d387782b5382`
- Notes: KL divergence regularization on latent distribution toward N(0,I) with kl_weight=0.01. Ensemble AUROC 92.03 (+1.09 vs baseline 90.94, +0.41 vs ensemble-only 91.62). AUPRC 45.10 (+1.99 vs baseline 43.11). Per-seed mean AUROC 91.38. KL weight screening: kl=0.01 won (AUROC 91.65 single seed vs kl=0.0 at 90.05). Paper reports 91.7 AUROC — now matching/exceeding.
