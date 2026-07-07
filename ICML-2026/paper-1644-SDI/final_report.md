# Final Report: paper-1644

- Title: Step-resolved data attribution for looped transformers
- Primary metric: `Relative SDI Error` (lower)
- Records: 8
- Generated: 2026-07-07T09:44:35Z

## Best Result

- Iteration: 6
- Idea: PARAM-1b — Sketch dimension m=32768 (extended grid search)
- Primary metric: 0.006391
- Commit: `9b7e67493f664d6364522949ca26d606267f02b4`
- Notes: Extended m sweep to 32768. m=40960 showed no further improvement (0.00663). Sweet spot at m=32768. SDI Error: 0.00743->0.00639 (-14.0% from m=20480; -79.5% total from baseline 0.0311). Theoretical O(1/sqrt(2048/32768))=0.25 vs actual 0.205 reduction factor. Combined stack: ALGO-1 (FP16) + ALGO-3 (CW hash) + ALGO-2 (antithetic) + m=32768.
