# Final Report: paper-811

- Title: ReNF: Rethinking the Design of Neural Long-Term Time Series Forecasters
- Primary metric: `MSE` (lower)
- Records: 12
- Generated: 2026-07-05T07:36:56Z

## Best Result

- Iteration: 8
- Idea: PARAM-DLAYERS — d_layers=4 with d_ff=3072/dropout=0.92
- Primary metric: 0.35006
- Commit: `22e8a278ec258be01988c0bad00758c713d10f0d`
- Notes: d_layers=4 with d_ff=3072 and dropout=0.92 achieves new best MSE=0.35006 (-0.20% vs baseline 0.35076). Extra cascade stage gives finer BDO granularity (24 steps/stage vs 32). MAE=0.38377 within guardrail. Training stable, test_loss=0.35040 at epoch 10.
