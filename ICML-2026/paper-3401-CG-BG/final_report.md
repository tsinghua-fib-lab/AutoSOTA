# Final Report: paper-3401

- Title: Coarse-Grained Boltzmann Generators
- Primary metric: `JS` (lower)
- Records: 6
- Generated: 2026-07-11T00:37:52Z

## Best Result

- Iteration: 5
- Idea: IDEA-02-SOFT — clip=90 α=0.1: combined clip+soft optimization
- Primary metric: 0.004484
- Commit: `45ed38794c1ea6c1aa6888a30c301b3b03a89099`
- Notes: Combined clip sweep + soft clipping. clip=90 with alpha=0.1 gives JS=0.004484 (↓14.4% vs baseline 0.005236), matching clip=93 alpha=0.05 on JS but with better PMF=0.213476 (↓3.4%). ESS=0.786037 (↑42.2%). All three metrics Pareto-dominate baseline dramatically.
