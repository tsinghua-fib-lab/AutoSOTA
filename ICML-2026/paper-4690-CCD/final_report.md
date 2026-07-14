# Final Report: paper-4690

- Title: Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences
- Primary metric: `IoU` (higher)
- Records: 9
- Generated: 2026-07-13T07:43:23Z

## Best Result

- Iteration: 8
- Idea: IDEA-12 — K=32 Hutchinson + per-sample norm + sigma=4.0
- Primary metric: 0.8935
- Commit: `d9041baeb431901f06e1fd3d486c9b0c2d62e775`
- Notes: K=32 Hutchinson (up from K=24) with optimal per-sample [2,98] percentile normalization and Gaussian smoothing sigma=4.0. IoU: 0.8933->0.8935 (+0.02%). Diminishing returns from K increase confirmed: K=16->24 = +0.03%, K=24->32 = +0.02%.
