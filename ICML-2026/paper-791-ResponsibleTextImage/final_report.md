# Final Report: paper-791

- Title: Responsible Text-to-Image Diffusion: Interpretable and Linearly Controllable Semantics for Fair and Safe Generation
- Primary metric: `delta` (lower)
- Records: 7
- Generated: 2026-07-17T12:53:38Z

## Best Result

- Iteration: 1
- Idea: ALGO-02 — coefficient=100 for optimal delta-CLIP tradeoff
- Primary metric: 0.0
- Commit: `1eb182a3983f5351320665655daa80c7ef89ef5c`
- Notes: Increased coefficient from 10.0 (baseline) to 100.0. Achieved perfect gender balance (150M/150F, delta=0.0000). CLIP dropped from 29.25 to 28.40 (-2.9%, within 5% tolerance). Higher coefficient strengthens concept vector influence on generation, pushing gender distribution toward balance.
