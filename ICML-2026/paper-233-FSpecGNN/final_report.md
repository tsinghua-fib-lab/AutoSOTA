# Final Report: paper-233

- Title: Full-Spectrum Graph Neural Networks: Expressive and Scalable
- Primary metric: `Test Accuracy` (higher)
- Records: 12
- Generated: 2026-07-04T17:30:26Z

## Best Result

- Iteration: 3
- Idea: ALGO-1 — SGDR Cosine Annealing (T_0=200)
- Primary metric: 39.48
- Commit: `48043a06c588b4606ff612ce6da59a12e24cc0ba`
- Notes: Added CosineAnnealingWarmRestarts scheduler with T_0=200, T_mult=1, eta_min=1e-6. Test Accuracy improved from 39.35% (iter-1) to 39.48% (+0.13%). T_0=50 and T_0=100 both performed worse, suggesting the model benefits from a longer, gentler LR cycle on this small dataset. Runtime 4.49s (within 5.43s guardrail).
