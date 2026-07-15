# Final Report: paper-5996

- Title: RTInfer: Real-Time Inference of Multiple DNNs on Edge GPUs
- Primary metric: `DMR` (lower)
- Records: 13
- Generated: 2026-07-14T02:58:07Z

## Best Result

- Iteration: 10
- Idea: NEW-2 — Memory-efficient downgrade replacement criterion
- Primary metric: 0.0
- Commit: `9e3ab2c6a77ece41827c9e9115e6e6239bea9f6a`
- Notes: Modified _downgrade_one() replacement selection from max(accuracy) to max(accuracy/memory^0.2). Picking memory-efficient downgrade targets reduces cascading downgrades. Accuracy 99.71->99.75 (+0.04pp). DMR stays 0.00%. Combined with NEW-1 variant selection bias.
