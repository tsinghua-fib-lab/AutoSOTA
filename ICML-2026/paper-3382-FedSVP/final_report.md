# Final Report: paper-3382

- Title: Beyond Description: Federated Adaptation via Semantic-Visual Prototype Alignment
- Primary metric: `Generalization Accuracy` (higher)
- Records: 15
- Generated: 2026-07-10T16:54:48Z

## Best Result

- Iteration: 6
- Idea: CODE-01+03+BS64 — Gradient Clipping + LR Warmup + Server Batch 64
- Primary metric: 73.67
- Commit: `7149c5f6741ad5c386e924db8cc376f909f58713`
- Notes: CODE-01+03 + global_batch_size=64. GenAcc 73.67 vs 72.4 (+1.27pp, significant). PerAcc 82.34 vs 82.76 (-0.42pp, within 1% tolerance). Key improvement: 63 negatives/query (vs 7) provides stronger contrastive signal for global semantic prototypes.
