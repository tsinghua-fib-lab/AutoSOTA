# Final Report: paper-3583

- Title: RBCBF: Decoding Time Safety Alignment via Risk Guided Rollback and Barrier Control
- Primary metric: `Dterm` (lower)
- Records: 9
- Generated: 2026-07-10T11:31:26Z

## Best Result

- Iteration: 1
- Idea: 3583-ALGO-02 — EMA smoothing of h-score (alpha=0.5)
- Primary metric: 0.2206
- Commit: `e4a8c4fcb2703e5ea99d3b18e5ccaf87b89352c7`
- Notes: EMA smoothing (alpha=0.5) applied to h-score gate decisions. Dterm reduced from 0.597 to 0.2206 (63% improvement). Same 10 prompts triggered as baseline. Smoothing stabilizes gate decisions against per-step scorer noise, leading to more precise trigger timing and better post-rollback recovery.
