# Final Report: paper-741

- Title: Robust Linear Dueling Bandits with Post-serving Context under Unknown Delays and Adversarial Corruptions
- Primary metric: `Cumulative Regret` (lower)
- Records: 13
- Generated: 2026-07-15T00:11:54Z

## Best Result

- Iteration: 11
- Idea: ALGO-4b — Tuned MLP Warm-Start (100 pretrain rounds)
- Primary metric: 8022.8
- Commit: `14e3582009e9385479fb121fab4891e1522158a8`
- Notes: Increased pretrain_rounds from 50 to 100. Random exploration for first 100 rounds (200 random samples) builds a better initial MLP buffer. Combines with adaptive UCB alpha decay. Result 8022.8, 14.4% below baseline 9375.0. More pretraining rounds further improve early MLP predictions.
