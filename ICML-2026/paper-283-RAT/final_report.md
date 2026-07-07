# Final Report: paper-283

- Title: Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation
- Primary metric: `Ep. Returns` (higher)
- Records: 7
- Generated: 2026-07-05T16:54:13Z

## Best Result

- Iteration: 3
- Idea: CODE-P0-01 — gamma configurable (gamma=0.99) + ent_coef=0.005
- Primary metric: 5531.23
- Commit: `ea461419044700df7ea793ca38f6e3e406218c8b`
- Notes: Made gamma configurable via YAML/CLI. Set gamma=0.99 (paper value vs code default 0.999). Combined with ent_coef=0.005 from Iter 1. BREAKTHROUGH: final eprewmean 5531.23 (+32.8% over baseline). Peak 5575.30. Entropy went slightly negative (-0.23) but above -2.0 guardrail. gamma=0.99 dramatically improves value estimation stability.
