# Final Report: paper-5373

- Title: Convex Low-resource Accent-Robust Language Detection in Speech Recognition
- Primary metric: `Detection Accuracy` (higher)
- Records: 7
- Generated: 2026-07-13T11:21:34Z

## Best Result

- Iteration: 3
- Idea: IDEA-06 — Increased Nystrom rank from 20 to 32
- Primary metric: 0.9774
- Commit: `ac708b049a969e9b68ddd982d1fcf64f898fa734`
- Notes: Increased Nyström preconditioner rank from 20 to 32. Detection Accuracy improved from 0.9739 to 0.9774 (+0.35%). WER: 48.55 (baseline 48.25, within 5% tolerance). CER: 27.95 (baseline 27.45, within 5% tolerance). Classification report shows accuracy 0.98 vs baseline 0.97. Rank=32 provides better preconditioner for 768-dim Whisper features. The rank=20 default was inherited from ImageNet experiments and was suboptimal for this data distribution. Best model so far.
