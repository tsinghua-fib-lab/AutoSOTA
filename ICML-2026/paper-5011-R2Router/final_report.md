# Final Report: paper-5011

- Title: R2-Router: A New Paradigm for LLM Routing with Reasoning
- Primary metric: `AUDC` (higher)
- Records: 7
- Generated: 2026-07-13T06:19:52Z

## Best Result

- Iteration: 4
- Idea: ALGO-02 — Ensemble calibrated KNN k=256+k=512 (w=0.65+0.35)
- Primary metric: 0.7628
- Commit: `245030671fdb6dc7cb9b81e713d1094ffbb7c439`
- Notes: Weighted ensemble of calibrated KNN predictions at k=256 and k=512. Each model independently calibrated for variance scaling. Ensemble weights (0.65, 0.35) optimized via grid search on calibration set. AUDC +0.0327 (+4.5%), Peak +0.0371 (+4.9%), QNC -0.3774 (-77.5%) vs baseline. All guardrails dramatically improved.
