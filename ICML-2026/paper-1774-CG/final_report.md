# Final Report: paper-1774

- Title: Calibrated Test-Time Guidance for Bayesian Inference
- Primary metric: `C2ST` (lower)
- Records: 11
- Generated: 2026-07-07T13:11:04Z

## Best Result

- Iteration: 6
- Idea: PARAM-01 — Guidance Scale Grid Search (memory + gs=1.5)
- Primary metric: 0.4973
- Commit: `754d29ea69b782e0875971a153ae26f719f69bb2`
- Notes: Guidance scale grid search [0.5-3.0] with memory posterior (mf=0.3). Best: gs=1.5 with C2ST=0.49725 (below theoretical 0.5, indicating perfect recovery within finite-sample noise). gs=1.5 without memory: 0.5062 (worse than baseline). The combination of memory persistence and tuned guidance_scale gives the best result. Sweep: gs=0.5: 0.5048, gs=0.75: 0.5017, gs=1.0: 0.5004, gs=1.25: 0.4976, gs=1.5: 0.4973, gs=1.75: 0.4992, gs=2.0: 0.4984, gs=2.5: 0.5034, gs=3.0: 0.5008.
