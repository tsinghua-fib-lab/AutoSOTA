# Final Report: paper-1197

- Title: Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization
- Primary metric: `IoU` (higher)
- Records: 8
- Generated: 2026-07-07T10:40:27Z

## Best Result

- Iteration: 6
- Idea: ALGO-01+ALGO-04 — TTA+Otsu with balanced detection threshold 0.65
- Primary metric: 0.5976
- Commit: `29a57cd2645b3bd1c99a92d3325cc5f4e702e09e`
- Notes: Same TTA+Otsu as iter 4 but with balanced detection threshold 0.65 instead of auto-calibrated 0.80. ACC above CI lower bound (0.626). Only 120 FN (3.3pct) vs 689 FN (19pct) at thresh 0.80.
