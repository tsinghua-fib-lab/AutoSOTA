# Final Report: paper-4636

- Title: Flow Matching Calibration for Simulation-Based Inference under Model Misspecification
- Primary metric: `jMMD` (lower)
- Records: 13
- Generated: 2026-07-12T11:22:47Z

## Best Result

- Iteration: 11
- Idea: PARAM-03 — z_score + epochs=300 patience=40
- Primary metric: 4.2e-05
- Commit: `60942814d305a5477c501828f96656dbaa77b442`
- Notes: z_score + weight_decay=1e-5 + epochs=300 patience=40. Further improvement: jC2ST 0.512 (was 0.515), jMMD 0.000042 (was 0.000065, 35% better), jWass 0.0043 (was 0.0058). All seeds stable (std 0.006). More epochs with z_score consistently improves.
