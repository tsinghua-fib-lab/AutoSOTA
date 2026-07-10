# Final Report: paper-1755

- Title: Real-Time Monitoring and Calibration of Chain-of-Thought Sycophancy in Large Reasoning Models
- Primary metric: `RR` (higher)
- Records: 7
- Generated: 2026-07-09T12:58:33Z

## Best Result

- Iteration: 3
- Idea: CODE-01 — Per-layer steering scales [0.7, 0.85, 1.0, 1.5]
- Primary metric: 0.5333
- Commit: `6c308682fcd2013b6d5bf61a3e039c929ef4d4a8`
- Notes: CODE-01: Per-layer independent steering scales. Applied weights [0.7, 0.85, 1.0, 1.5] to steer layers [16, 17, 18, 19] respectively. Weaker steering on early layers, stronger on later layers (closer to output). RR improved from 0.3667 to 0.5333 (+45.4%), SR unchanged at 0.2000. Converted 2-3 wrong answers to correct while maintaining sycophancy suppression.
