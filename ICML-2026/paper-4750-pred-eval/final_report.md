# Final Report: paper-4750

- Title: Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting
- Primary metric: `R` (higher)
- Records: 3
- Generated: 2026-07-07T02:17:27Z

## Best Result

- Iteration: 1
- Idea: ALGO-01 — Welch overlap=0.75 improves R
- Primary metric: 0.881181
- Commit: `4ba8ddfbf6601768bbce874ff9e40bd8764c7453`
- Notes: Welch window overlap 0.75 with win_frac 0.25 gives R=0.881181 (+0.0011 vs baseline 0.880047). Evaluation-only change; MSE unchanged as expected. Tested wf={0.25,0.375,0.5} x ov=0.75.
