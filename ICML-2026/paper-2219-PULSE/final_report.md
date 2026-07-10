# Final Report: paper-2219

- Title: PULSE: Generative Phase Evolution for Non-Stationary Time Series Forecasting
- Primary metric: `MSE` (lower)
- Records: 4
- Generated: 2026-07-09T17:28:26Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — Add MSE loss weight (rec_lambda=0.5) alongside FFT auxiliary loss
- Primary metric: 0.159433
- Commit: `199a4756b2d42127a318fe7b9b4962a02091d5bc`
- Notes: Added --rec_lambda 0.5 to evaluate_electricity.sh, adding MSE supervision alongside FFT auxiliary loss. Baseline only used FFT loss (rec_lambda=0). MSE improved from 0.159931 to 0.159433 (-0.31%). MAE slightly increased from 0.255389 to 0.255750 (+0.14%, well within 2.5% tolerance). Per-horizon: 96=0.133580, 192=0.150401, 336=0.166233, 720=0.187518.
