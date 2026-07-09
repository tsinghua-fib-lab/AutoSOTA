# Final Report: paper-2760

- Title: On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective
- Primary metric: `AUROC` (higher)
- Records: 11
- Generated: 2026-07-09T07:56:22Z

## Best Result

- Iteration: 10
- Idea: CODE-04 — MAX_LENGTH=1024 with length normalization
- Primary metric: 89.67
- Commit: `661202c2491304106a8be0fbe00dec8a60c457d3`
- Notes: MAX_LENGTH doubled from 512 to 1024 with CHUNK_SIZE halved to 16. Added length normalization. AUROC jumped from 88.03% to 89.67% (+1.64%). Cumulative gain from baseline: +4.79%. The longer sequence provides more tail tokens and richer entropy estimates. Built on ALGO-02 (windowed entropy), ALGO-05 (entropy shape), PARAM-01 (weight tuning), and ALGO-06 (adaptive percentile).
