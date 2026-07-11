# Final Report: paper-3690

- Title: Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation
- Primary metric: `CHAIRs` (lower)
- Records: 4
- Generated: 2026-07-10T19:00:50Z

## Best Result

- Iteration: 2
- Idea: PARAM-01 — alpha=7.0 (higher visual grounding)
- Primary metric: 34.7
- Commit: `58170f150fa6aba55718036cd0a91e4510d8f2c2`
- Notes: Alpha increased from 5.0 to 7.0. Massive improvement: CHAIRs from 43.6 to 34.7 (-8.9, -20.4%). CHAIRi from 25.1 to 19.8 (-5.3, -21.1%). Captions with hallucinations reduced from 217 to 173. Higher alpha=7.0 significantly improves visual grounding. This beats the paper reported CHAIRs of 39.8.
