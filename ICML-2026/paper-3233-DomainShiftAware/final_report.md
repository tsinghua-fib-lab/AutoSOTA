# Final Report: paper-3233

- Title: Domain-Shift-Aware Conformal Prediction for Large Language Models
- Primary metric: `Empirical Coverage` (higher)
- Records: 7
- Generated: 2026-07-10T03:41:41Z

## Best Result

- Iteration: 6
- Idea: IDEA-01+IDEA-09 — APS score + best XGBoost config (IDEAs 01+09)
- Primary metric: 0.9556
- Commit: `97c4160af61076334fb9447161bbb70d0953545c`
- Notes: APS nonconformity score with XGBoost depth=2,n=80,lr=0.2. Highest coverage achieved: 0.9556 (+0.0139 vs baseline). Set size 3.3632 within tolerance. APS captures richer distributional info from 6-option MMLU.
