# Final Report: paper-3720

- Title: Leveraging Lineage Barcodes as Natural Augmentations for Contrastive Learning of Cell Fate in scRNA-seq Data
- Primary metric: `KNN_Test_Error` (lower)
- Records: 8
- Generated: 2026-07-10T19:00:46Z

## Best Result

- Iteration: 6
- Idea: ALGO-04 — Expression-Weighted Contrastive Loss
- Primary metric: 0.2538
- Commit: `65891e0eff407026a7ac9d0aa4c249a3b36e7cb0`
- Notes: Weighted positive pairs by expression dissimilarity (weight=1-cos_sim, clipped [0.1,1.0]). Hard positives weighted higher. Combined with dropout+gene_dropout+grad_accum. KNN: 0.4009->0.2538 (-36.7%). KL: 0.4558->0.5066 (within 0.595). Both metrics improved.
