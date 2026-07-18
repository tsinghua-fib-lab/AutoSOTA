# Final Report: paper-2258

- Title: Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models
- Primary metric: `AbsRel` (lower)
- Records: 13
- Generated: 2026-07-10T12:03:40Z

## Best Result

- Iteration: 11
- Idea: PARAM-01b — TSR sigma=0.5 with ensemble_size=10
- Primary metric: 5.29
- Commit: `f08d2cec46ae4b2e8e27b66b4b2633efba1abbb6`
- Notes: Tested sigma=0.5 with ensemble=10, k=1.5 fixed. DDIM: AbsRel=5.27% d1=96.69%. TSR: AbsRel=5.29% d1=96.69%. Slightly better AbsRel than sigma=1.0 (5.29 vs 5.30), but d1 slightly lower (96.69 vs 96.79). sigma=0.5 concentrates TSR effect on fewer timesteps.
