# Final Report: paper-1518

- Title: SORA: Free Second-Order Attacks in Fast Adversarial Training
- Primary metric: `Clean` (higher)
- Records: 8
- Generated: 2026-07-07T23:14:05Z

## Best Result

- Iteration: 5
- Idea: LABEL-SMOOTHING — Label Smoothing 0.1 for Cross-Entropy Loss
- Primary metric: 80.33
- Commit: `790385c220b053835e569755301c4da4403a8f3e`
- Notes: Added label_smoothing=0.1 to F.cross_entropy. ALL metrics improved: Clean +0.29pp (80.04→80.33), FGSM +0.89pp (53.39→54.28), PGD-10 +0.60pp (48.63→49.23). Simple one-line change with consistent improvement. Prior SOTA context suggested +1.07pp potential. AA eval pending.
