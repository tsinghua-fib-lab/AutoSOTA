# Final Report: paper-2592

- Title: A Diffusive Classification Loss for Learning Energy-based Generative Models
- Primary metric: `L_clf` (lower)
- Records: 9
- Generated: 2026-07-08T17:22:05Z

## Best Result

- Iteration: 3
- Idea: ALGO-03-04-05 — EMA + CosineLR + AdamW (seed 0)
- Primary metric: 4.285
- Commit: `7396abf2815fdffd03931f4d420f54ec2607db0a`
- Notes: Combined EMA (decay=0.999), Cosine Annealing LR (5% warmup, cosine decay to 0), and AdamW (weight_decay=1e-4). Seed 0. L_clf improved from 4.388 to 4.285 (-2.4%). FD improved from 2.162 to 0.533. MMD improved from 0.871 to 0.481. All metrics improved simultaneously.
