# Final Report: paper-3287

- Title: A Two-Layer Framework for Joint Online Configuration Selection and Admission Control
- Primary metric: `competitive_ratio` (higher)
- Records: 7
- Generated: 2026-07-10T11:07:40Z

## Best Result

- Iteration: 6
- Idea: PARAM-01 — PARAM-01: alpha=0.017 + noise_sigma=0.05 (refined optimum)
- Primary metric: 97.63
- Commit: `2dfac9a7e367891bdc2d6d26ec55a4eec4fe6455`
- Notes: alpha=0.017 + noise_sigma=0.05. CR=97.63% BEATS paper reported 97.38% (+0.25pp). Std=0.61% vs baseline 3.11% (80% reduction). Min=95.12% vs baseline 75.89%. Further refinement of the alpha-noise_sigma joint optimum. Optimal region found: alpha in [0.017, 0.020], noise_sigma=0.05. 50-seed full evaluation.
