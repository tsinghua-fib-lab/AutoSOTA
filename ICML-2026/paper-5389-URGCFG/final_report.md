# Final Report: paper-5389

- Title: UNIVERSAL REPRESENTATION OF GENERALIZED CONVEX FUNCTIONS AND THEIR GRADIENTS
- Primary metric: `Profit per Good` (higher)
- Records: 13
- Generated: 2026-07-13T02:41:13Z

## Best Result

- Iteration: 9
- Idea: IDEALIB-5389-16 — Grid init + npoints=100 + max_samples=5000
- Primary metric: 0.3072
- Commit: `f3dce5886f8868c23d5e79715a91cab9004e1136`
- Notes: BEST YET: Profit per Good 0.307185. +0.00369 over baseline. npoints=100 (4^4=256 grid points, 100 selected) with max_samples=5000. The combination of rich candidate set (100 support points with uniform grid init) and good Monte Carlo estimation (5000 type samples) enables the mechanism to learn a significantly better bundling strategy. Now 0.00218 above SJA bound (0.305). The GCF parameterization is proving more flexible than expected for this setting.
