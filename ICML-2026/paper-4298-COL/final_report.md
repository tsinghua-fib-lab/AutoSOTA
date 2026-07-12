# Final Report: paper-4298

- Title: Contrastive Order Learning: A General Framework for Ordinal Regression
- Primary metric: `SRCC` (higher)
- Records: 7
- Generated: 2026-07-11T19:44:49Z

## Best Result

- Iteration: 6
- Idea: IDEA-09 — Use all training data (drop_last=False)
- Primary metric: 0.8963
- Commit: `61b63767a4223b1f045bfb4df53a2c025fca1363`
- Notes: Changed drop_last=True to False in training DataLoader to use all 472 training samples (was dropping ~24/epoch = 5.1%). Combined with warmup cosine LR + Gaussian noise. SRCC improved to 0.8963 (+0.0016 over iter-5). PCC at 0.9159 (near baseline, within guardrail). Best split 1 result (0.8700 vs 0.8578 in iter-5).
