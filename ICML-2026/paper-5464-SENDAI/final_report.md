# Final Report: paper-5464

- Title: SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework
- Primary metric: `SSIM` (higher)
- Records: 9
- Generated: 2026-07-13T07:26:25Z

## Best Result

- Iteration: 7
- Idea: I01-TUNE — Tune SSIM loss lambda: 0.2 -> 0.5 (stronger SSIM emphasis)
- Primary metric: 0.6857
- Commit: `71362e2c0fc29ff5bd2f75ccab62a3e182ce9bb7`
- Notes: Increased SSIM loss weight lambda from 0.2 to 0.5. Massive improvement: SSIM 0.6857 (+11.7% vs baseline 0.614, +3.1% vs previous best 0.6652). RMSE 0.0952 (-2.4% vs baseline). Both metrics improved — unconditional win. Stronger SSIM emphasis in loss combined with INR decoder + SSL pretraining is highly effective.
