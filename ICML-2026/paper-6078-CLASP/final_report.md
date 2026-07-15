# Final Report: paper-6078

- Title: CLASP: Online learning algorithms for Convex Losses And Squared Penalties
- Primary metric: `CCVT_2` (lower)
- Records: 12
- Generated: 2026-07-14T11:25:00Z

## Best Result

- Iteration: 9
- Idea: step-c-0.25-s50 — step_c=0.25 validated with S=50 full run
- Primary metric: 47.03
- Commit: `f743020e8f7e0bf4dc4cb725b5e3246fd114ffae`
- Notes: step_c=0.25, S=50. CCVT2=47.0 (-45% vs baseline 85), CCVT1=25.2 (-52% vs baseline 52), loss=687.7 (+3.5% vs baseline 664, within 10% tolerance). Best constraint metrics achieved. Trade-off: stronger CCVT improvement at modest loss cost.
