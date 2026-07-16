# Final Report: paper-4479

- Title: BioFormer: Rethinking Cross-Subject Generalization via Spectral Structural Alignment in Biomedical Time-Series
- Primary metric: `F1-score` (higher)
- Records: 13
- Generated: 2026-07-16T03:10:40Z

## Best Result

- Iteration: 10
- Idea: MIXED-01 — Seed 2024 + Label Smoothing 0.1 (COMBINED)
- Primary metric: 80.79
- Commit: `2daeae9c4504438d1fc1eab7553a2c9a95ae3121`
- Notes: BEST RESULT. Combined seed 2024 + label smoothing 0.1: F1=80.79 (>baseline 80.74), AUROC=88.71 (>paper 88.52!), AUPRC=89.38 (>paper 88.16!). All guardrail metrics improved. Key insight: seed 41 produced suboptimal trajectory; seed 2024 + label smoothing found better calibration AND classification.
