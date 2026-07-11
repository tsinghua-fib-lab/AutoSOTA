# Final Report: paper-3815

- Title: “Do Diffusion Models Dream of Electric Planes?” Discrete and Continuous Simulation-Based Inference for Aircraft Design
- Primary metric: `MMD Value` (lower)
- Records: 3
- Generated: 2026-07-11T00:03:34Z

## Best Result

- Iteration: 2
- Idea: CODE-01 — Explicit weight_decay=1e-4 in AdamW optimizer
- Primary metric: 0.003619
- Commit: `b19e158dad483dd9370b914864baaae45c4812c6`
- Notes: CODE-01: Explicit weight_decay=1e-4 in AdamW optimizer. MMD reduced 8.3% (0.003948→0.003619). Joint C2ST improved 0.8%. All C2ST guardrails within regression tolerances: Mean +0.3% (<5%), Max +1.4% (<15%). Training was stable (train_loss 0.28, val_loss 0.30). Best candidate so far.
