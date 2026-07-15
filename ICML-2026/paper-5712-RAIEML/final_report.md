# Final Report: paper-5712

- Title: Robust AI Evaluation through Maximal Lotteries
- Primary metric: `win_rate_held_out_pct` (higher)
- Records: 7
- Generated: 2026-07-15T05:54:46Z

## Best Result

- Iteration: 5
- Idea: IDEA-5712-01-only — Sig filter (p<0.001) only, no adaptive smoothing
- Primary metric: 49.58
- Commit: `3a326f7d927b5916cf97dacbded59987fb8fd6f5`
- Notes: Significance filter p<0.001 only (no adaptive smoothing). Rho=0.4: 49.58% (baseline: 44.48%, +5.10pp). Adaptive smoothing adds negligible benefit — significance filtering alone drives the improvement by removing only the noisiest pairwise comparisons while preserving genuine signal.
