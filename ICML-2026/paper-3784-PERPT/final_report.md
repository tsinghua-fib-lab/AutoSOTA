# Final Report: paper-3784

- Title: Predicting evolutionary rate as a pretraining task improves genome language model representations
- Primary metric: `AUROC (Log-likelihood)` (higher)
- Records: 9
- Generated: 2026-07-10T22:00:45Z

## Best Result

- Iteration: 6
- Idea: ALGO-NEW-cons — Multi-position conservation aggregation (radius=1, mean)
- Primary metric: 0.547
- Commit: `ff58c557ac8d4e53b094d49c52e8dbad77e31ef2`
- Notes: Multi-position conservation aggregation with radius=1, mean pooling. Combined with best LLR settings (radius=3, mean, max strand). LLR AUROC=0.547 (baseline 0.498, +9.9%). ERP AUROC=0.655 (baseline 0.635, +3.1%). Both core metrics improved per multi_metric_tradeoff objective. The conservation head benefits from averaging across neighboring positions (±1bp), smoothing noise in the predicted evolutionary rate.
