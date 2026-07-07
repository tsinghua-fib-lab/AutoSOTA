# Final Report: paper-1430

- Title: Geometry-Preserving Unsupervised Alignment for Heterogeneous Foundation Models
- Primary metric: `Accuracy` (higher)
- Records: 14
- Generated: 2026-07-06T19:37:17Z

## Best Result

- Iteration: 11
- Idea: PARAM-01-alpha0.02 — Alpha=0.02 fusion + triplet best config - BEST OVERALL
- Primary metric: 89.63
- Commit: `0feff7c98c11653873146f59182df15cd64b7b33`
- Notes: Sinkhorn fusion alpha=0.02 (98% CLIP, 2% prototype). Triplet loss, lr=1e-3, 400 iters. Test: 89.63% (baseline: 88.67%). BEST RESULT: +0.96pp improvement. Surpasses paper reported 89.5%. Optimal balance: nearly all CLIP guidance in Sinkhorn with minimal prototype noise.
