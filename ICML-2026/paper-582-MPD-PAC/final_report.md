# Final Report: paper-582

- Title: Mitigating Mask Prior Drift and Positional Attention Collapse in Large Diffusion Vision-Language Models
- Primary metric: `CIDEr` (higher)
- Records: 13
- Generated: 2026-07-05T09:07:20Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — Param sweep: prior=0.5 step_per_block=8 (max steps)
- Primary metric: 46.9
- Commit: `67b60fa468460a82c72fbc818d55fe56bdfa2fa4`
- Notes: Further improvement with step_per_block=8: CIDEr 46.90 (+5.18 vs baseline, +12.4%). Trend: more denoising steps consistently improve quality. step_per_block=8 is the max (equals block_length=8).
