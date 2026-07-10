# Final Report: paper-3290

- Title: Learning Treatment Allocations with Risk Control Under Partial Identifiability
- Primary metric: `Treatment Risk` (lower)
- Records: 14
- Generated: 2026-07-10T03:08:04Z

## Best Result

- Iteration: 5
- Idea: 3290-012-split — More calibration data 0.4/0.4/0.2
- Primary metric: 0.246
- Commit: `4bc80215f3622d1022bf432bb8513f2ec5d34eed`
- Notes: Pareto improvement: PopRisk 0.521 (-3.5pct vs 0.540 baseline), TreatRisk 0.246 (still well below 0.35). More calibration data tightens Hoeffding-Bentkus bound, allowing policies that treat more patients.
