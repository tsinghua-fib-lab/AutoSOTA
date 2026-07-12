# Final Report: paper-4000

- Title: Inverting Data Transformations via Diffusion Sampling
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-11T13:58:17Z

## Best Result

- Iteration: 3
- Idea: ALGO-03 — Temperature annealing + Antithetic MC sampling
- Primary metric: 83.12
- Commit: `88cb5f192874024b70d0b28565ba1a3414386b5a`
- Notes: Antithetic MC on top of temperature annealing: replaced independent noise with antithetic pairs (z, -z) in forward diffusion. Accuracy +0.59pp over baseline (82.53→83.12), FID -0.32 (5.04→4.72). Zero additional compute cost. Cumulative: T=2.0→0.5 annealing + antithetic pairs.
