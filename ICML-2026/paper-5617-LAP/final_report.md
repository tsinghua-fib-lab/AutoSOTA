# Final Report: paper-5617

- Title: A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots
- Primary metric: `W1_distance` (lower)
- Records: 8
- Generated: 2026-07-15T00:49:00Z

## Best Result

- Iteration: 3
- Idea: IDEA-01 — WSD LR schedule (warmup=500, plateau=6000, decay to 1e-6) - no gradient clipping
- Primary metric: 0.7077
- Commit: `e5291ac29a9db321f329bdf0d86d760d7abf1b7c`
- Notes: WSD schedule alone (no gradient clipping). Final W1=0.7077 beats baseline 0.7168 by 1.3%. Best intermediate W1=0.6959 at step 4000 (2.9% improvement). Model overfits during plateau (W1 rises 0.696→0.722), then LR decay recovers to 0.708. Suggests shorter plateau + early stopping would capture the 0.696 result.
