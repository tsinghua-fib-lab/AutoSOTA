# Final Report: paper-2744

- Title: LOTTERY: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry
- Primary metric: `Test Power` (higher)
- Records: 11
- Generated: 2026-07-09T19:32:59Z

## Best Result

- Iteration: 8
- Idea: PARAM-01 — Fine-tuned: k_lof=10 + perturbation_scale=0.20
- Primary metric: 0.615
- Commit: `666a1d7095f2e8a5eede5387c91b84f761f4c051`
- Notes: PARAM-01 fine-tuning: with k_lof=10, the optimal perturbation_scale shifts to 0.20. Results: ps=0.15->0.609, ps=0.20->0.615, ps=0.25->0.609, ps=0.30->0.611, ps=0.35->0.608. The interaction between LOF k and sensitivity perturbation scale is important: smaller k gives more localized LOF scores which benefit from a slightly smaller perturbation scale. Config: sensitivity_weight, statistic_formulation=hybrid, train=0.5, calib=0.15.
