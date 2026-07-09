# Final Report: paper-2579

- Title: Mantis: Lightweight Foundation Model for Time Series Classification
- Primary metric: `UCR_Accuracy` (higher)
- Records: 11
- Generated: 2026-07-09T07:49:37Z

## Best Result

- Iteration: 6
- Idea: IDEA-01+07 — Multi-scale concat (raw+diff, 4096-dim) + multi-classifier CV
- Primary metric: 0.8526
- Commit: `aaecde7c4f00b62d3a002994b90d0c1407094d1a`
- Notes: Multi-scale [128,256,512,1024] on both raw signal and first-order difference, concatenated -> 4096-dim. Multi-classifier CV. 108/108 evaluated. +0.0007 over multi-scale alone (0.8519). +0.0336 over baseline. Diminishing returns from diff features.
