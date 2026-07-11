# Final Report: paper-3825

- Title: Spike Camera Autofocus via Frequency-Domain Spectral-Centroid Migration
- Primary metric: `AbsErr` (lower)
- Records: 8
- Generated: 2026-07-10T19:27:47Z

## Best Result

- Iteration: 11
- Idea: PARAM-01c — alpha_multi=3.00 with dense r2 grid (0.005 step)
- Primary metric: 2.73
- Commit: `8aa959034435230c55d609c34022e3990c0fdc1e`
- Notes: alpha_multi=3.00 with dense r2 grid (0.005 step, 66 candidates). USAF_static_decrease improved 7->4 (r2 0.105->0.090). USAF overall 3.78->3.44. MEAN AbsErr 2.73 (-33.9% baseline 4.13, BEATS paper 2.80). 5/15 scenes perfect (abs_err=0). RelErr 0.30% (baseline 0.40%). All guardrail metrics improved.
