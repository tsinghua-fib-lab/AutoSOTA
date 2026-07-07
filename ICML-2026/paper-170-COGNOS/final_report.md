# Final Report: paper-170

- Title: COGNOS: Universal Enhancement for Time Series Anomaly Detection via Constrained Gaussian-Noise Optimization and Smoothing
- Primary metric: `Std-F1` (higher)
- Records: 8
- Generated: 2026-07-04T18:50:52Z

## Best Result

- Iteration: 5
- Idea: ALGO-01 — Replace BatchNorm1d with LayerNorm in KANAD
- Primary metric: 0.9339
- Commit: `5b660a28a0e04767365767c04034c18a22f14445`
- Notes: Replaced all BatchNorm1d with LayerNorm(seq_len) in KANADModel. Epoch 1 val_loss dropped from 8.02 (BatchNorm) to 0.79 (LayerNorm) — 10x improvement. LayerNorm eliminates batch dependence and provides per-sample normalization, which is critical for anomaly detection where anomalous samples distort batch statistics. Enormous gains: Std-F1 0.9164->0.9339 (+1.9%), Aff-F1 +3.5%, R-A-R +6.8%, V-R +7.9%. Training became unstable at epochs 3-4 (lr warmup peak) but EarlyStopping saved epoch-1 best model. Combined with cosine_warmup LR, gradient clipping, and numerical stability fixes.
