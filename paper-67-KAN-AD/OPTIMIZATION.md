# Paper 67 — KAN-AD

**Full title:** *KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks*

**Registered metric movement (internal ledger):** +0.89%(0.9106→0.9187 F1)

## Summary

The best F1 improved from **0.9106** to **0.9187** with a compact scoring/training recipe: switch to a **sin+cos Fourier basis** (`order=3`), use **CosineAnnealingLR** with early-stop **patience=5**, and apply **variance-normalized anomaly scoring** (`alpha=0.5`, `local_std_window=16`). The gain mainly comes from smoother optimization and more stable anomaly scale across windows.

## Key ideas

- Fourier basis enriches periodic structure without a heavy model change.
- Cosine schedule improves late-stage convergence on anomaly tasks.
- Local-variance normalization stabilizes threshold sensitivity and raises F1.

## Where to look

- Source snapshot from `scy/auto-pipeline-ab/optimized_code/paper-2850/`
- run notes reflected in the central `results.md` ledger
