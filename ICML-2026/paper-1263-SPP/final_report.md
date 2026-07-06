# Final Report: paper-1263

- Title: SubspacePath Pruner: Inference-time Pruning via Probe-based Representation–Parameter Coupling
- Primary metric: `Token-level Recall` (higher)
- Records: 8
- Generated: 2026-07-06T11:33:09Z

## Best Result

- Iteration: 7
- Idea: PARAM-01d — rho_min=0.25 + pruning_strength=0.7, plateau reached
- Primary metric: 26.14
- Commit: `5506c2609497b5c682f429b1c0d80304f51d57a7`
- Notes: Combined rho_min=0.25 with pruning_strength=0.7. Result: 26.14% (+1.47pp over baseline). Head retention 95.8%. Improvement plateaued — only +0.01pp from iter-6. Only 1.56pp below dense ceiling 27.70%. Binary mask mechanism is the likely bottleneck for remaining gap.
