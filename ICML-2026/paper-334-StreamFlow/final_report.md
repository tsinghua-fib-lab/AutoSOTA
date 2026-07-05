# Final Report: paper-334

- Title: StreamFlow: Theory, Algorithm, and Implementation for High-Efficiency Rectified Flow Generation
- Primary metric: `FPS` (higher)
- Records: 11
- Generated: 2026-07-04T18:23:24Z

## Best Result

- Iteration: 7
- Idea: CODE-1+ALGO-2 — TAESD + VAE batch decode (batch=4)
- Primary metric: 22.14
- Commit: `32e75711cc0698cc6360c1d587d1b336d0d76c02`
- Notes: TAESD VAE + VAE_BATCH_SIZE=4. FPS: +82.5% over baseline (12.14->22.14). Memory: -7.8% (3326->3066 MB). Power: +0.8% (393->397W). All guardrails satisfied. TAESD enables both faster decode AND memory headroom for batching.
