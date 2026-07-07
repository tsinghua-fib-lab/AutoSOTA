# Final Report: paper-988

- Title: RACER: Risk-Aware Calibrated Efficient Routing for Large Language Models
- Primary metric: `Accuracy` (higher)
- Records: 13
- Generated: 2026-07-05T20:35:11Z

## Best Result

- Iteration: 1
- Idea: ALGO-1 — Focal Loss for Router Training (gamma=2.0, alpha=0.25)
- Primary metric: 77.84
- Commit: `cdba26154259e8554184233ff83ab775d27eee13`
- Notes: Replaced BCEWithLogitsLoss with Focal Loss (gamma=2.0, alpha=0.25) in MLP router training. +0.27pp over baseline. Risk stays at 2.93% (<=0.03). Set size 2.82. Loss function focuses on hard borderline cases and down-weights easy negatives.
