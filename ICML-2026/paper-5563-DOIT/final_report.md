# Final Report: paper-5563

- Title: Training-Free Adaptation of Diffusion Models via Doob's $h$-Transform
- Primary metric: `Normalized Score` (higher)
- Records: 13
- Generated: 2026-07-13T18:24:36Z

## Best Result

- Iteration: 10
- Idea: PARAM-particles-64 — Increase particles from 32 to 64
- Primary metric: 56.17
- Commit: `f20fcb26c2bad16e1c0768559b4efae328674563`
- Notes: Per-seed: [56.05, 56.15, 56.09, 56.16, 56.40]. Overall 56.17 (+0.92 vs baseline 55.25). Particles: 4->8 (+0.38), 8->16 (+0.16), 16->32 (+0.14), 32->64 (+0.24). Best result so far. More particles improve both MC estimation and Best-of-K selection.
