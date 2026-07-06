# Final Report: paper-512

- Title: VIP: Visual-guided Prompt Evolution for Efficient Dense Vision-Language Inference
- Primary metric: `mIoU` (higher)
- Records: 9
- Generated: 2026-07-05T16:19:43Z

## Best Result

- Iteration: 7
- Idea: idea-07 — PAMR steps=3 + Gaussian + slide_stride=84 (more overlap)
- Primary metric: 74.36
- Commit: `7890654d937649c1f9400b8d010828dba7e9f8bb`
- Notes: slide_stride 112→84 (75% overlap vs 67%): mIoU 74.29→74.36 (+0.07). More overlap improves spatial consistency at crop boundaries. Combined with PAMR steps=3 and Gaussian blending. Cumulative improvement: 73.26→74.36 (+1.10).
