# Final Report: paper-5195

- Title: InfoGlobe: Local-and-Global Information-Preserving Statistical Manifold Learning for Single-Cell Transcriptomics
- Primary metric: `Spearman Correlation` (higher)
- Records: 13
- Generated: 2026-07-13T04:43:42Z

## Best Result

- Iteration: 11
- Idea: PARAM-05 — l2_ratio=0.1 with MAE loss
- Primary metric: 0.9852
- Commit: `d8ad46be083e7d6164dc94e164412e8c321711e7`
- Notes: Reduced l2_ratio to 0.1 with MAE loss. Spearman=0.9852 (+2.36% vs baseline, +0.09% vs l2=0.3). Trust=0.9841 (-0.01% vs baseline), Contin=0.9844 (-0.07%). Trend: lower l2_ratio consistently improves Spearman. The reconstruction loss (fisher_rao_dis) already captures structure; minimal MDS guidance preserves rank ordering best.
