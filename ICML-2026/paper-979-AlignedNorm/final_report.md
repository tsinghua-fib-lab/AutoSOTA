# Final Report: paper-979

- Title: AlignedNorm: Prompting Vision–Language Models via Coupled Prompt Field
- Primary metric: `HM` (higher)
- Records: 10
- Generated: 2026-07-05T19:48:54Z

## Best Result

- Iteration: 6
- Idea: CODE-03-T12 — Temperature calibration T=1.2 for novel classes
- Primary metric: 90.34
- Commit: `57fb92d61c58044f24c49e25189394174fba01e7`
- Notes: Temperature T=1.2 applied to novel-class logits (logits/1.2). New improved from 84.87 to 85.20 (+0.33pp). Base unchanged at 96.20. HM=90.34 > baseline 90.15. Softening novel-class logits reduces overconfidence. First successful optimization.
