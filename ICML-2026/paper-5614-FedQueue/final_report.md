# Final Report: paper-5614

- Title: FedQueue: Queue-Aware Federated Learning for Cross-Facility HPC Training
- Primary metric: `Max-A` (higher)
- Records: 12
- Generated: 2026-07-15T00:33:57Z

## Best Result

- Iteration: 9
- Idea: 5614-10 — Residual Skip Connection in CNN on top of BatchNorm+WSD+LabelSmooth
- Primary metric: 99.39
- Commit: `42db1ba40f191b8f2465af7fec98ab2ef62aea43`
- Notes: Added 1x1 conv skip connection from conv1 output to conv2 output with adaptive_avg_pool2d. Max-A NEW BEST: 99.39% (+0.74% over baseline 98.65%). TTA 8.8s (-20.7% vs baseline). #Ek unchanged. Combined: BatchNorm+WSD+LabelSmooth+Skip.
