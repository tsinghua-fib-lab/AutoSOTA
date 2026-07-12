# Final Report: paper-4029

- Title: Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution
- Primary metric: `DiffID` (higher)
- Records: 6
- Generated: 2026-07-11T17:57:13Z

## Best Result

- Iteration: 4
- Idea: ALGO-04 — IDGI + Cosine fraction schedule
- Primary metric: 0.3716
- Commit: `85ff7751a359f2a4831c0845d5c0da1a7ce2219e`
- Notes: IDEA-04: Stacked cosine fraction annealing (0.10→0.02) on top of IDGI. DiffID +0.0156 (0.356→0.3716, +4.4%), Ins +0.0075 (0.4409→0.4484), Del -0.008 (0.0849→0.0769). All three metrics improved. Cosine schedule enables broader early exploration, IDGI maintains clean direction decomposition.
