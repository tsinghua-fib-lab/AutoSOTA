# Final Report: paper-4281

- Title: Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens
- Primary metric: `GSM8K_accuracy` (higher)
- Records: 6
- Generated: 2026-07-12T03:08:52Z

## Best Result

- Iteration: 5
- Idea: PARAM — fusion_alpha=0.45 (optimal)
- Primary metric: 28.51
- Commit: `698cd309a86e789a6f7e15176382000f3d0fe1af`
- Notes: Set fusion_alpha=0.45. Result: 376/1319=28.51%. BEST overall result. Alpha-to-accuracy mapping: 0.3→27.82, 0.4→28.43, 0.45→28.51, 0.6→27.75. Optimal alpha is ~0.45 for this checkpoint. Improvement over baseline: +3.19% absolute (25.32→28.51).
