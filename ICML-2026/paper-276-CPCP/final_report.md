# Final Report: paper-276

- Title: Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction
- Primary metric: `MSCE` (lower)
- Records: 13
- Generated: 2026-07-05T11:42:05Z

## Best Result

- Iteration: 8
- Idea: PARAM-02 — Delta bandwidth sweep: 0.02 to 0.03
- Primary metric: 0.001356
- Commit: `c187bce3aedb6a25a597b4ae2b094b060b1deb3d`
- Notes: Changed delta from 0.02 to 0.03 within known robust range [0.01,0.05]. MSCE ↓18.2% from baseline (0.001658→0.001356), L2-ERT ↓13.0% (0.000820→0.000713). WSC, L1-ERT, Volume all maintained near best levels. Wider bandwidth → more discriminative density weights. All metrics from eval stdout.
