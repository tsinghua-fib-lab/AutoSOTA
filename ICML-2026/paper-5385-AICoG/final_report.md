# Final Report: paper-5385

- Title: Aitchison Embeddings for Learning Compositional Graph Representations
- Primary metric: `AUC-ROC` (higher)
- Records: 11
- Generated: 2026-07-13T02:17:10Z

## Best Result

- Iteration: 10
- Idea: PARAM-03 — Scaling epochs=1500 with ratio=2.5 (best config)
- Primary metric: 0.8488
- Commit: `22a7e1701410b692979ad969ec7d0f1216e5da06`
- Notes: Found optimal config: neg_ratio=2.5, scaling_epochs=1500. AUC-ROC=0.8488 (+0.0081 vs baseline 0.8407), PR-AUC=0.8775 (+0.0062 vs baseline 0.8713). Both metrics show clear improvement. The combination of fewer negatives (ratio 2.5) and longer gamma calibration (1500 epochs) allows the model to better learn per-node degree effects before distance-based logits dominate.
