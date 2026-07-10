# Final Report: paper-2947

- Title: Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors
- Primary metric: `E2E-ASR` (higher)
- Records: 8
- Generated: 2026-07-09T15:59:39Z

## Best Result

- Iteration: 7
- Idea: IDEA-09 — Sink head pre-filtering
- Primary metric: 81.55
- Commit: `b54209acee7e88e83a19bd807c0a12e9bfc3ffc6`
- Notes: IDEA-09: Attention sink head pre-filtering (threshold=0.5). E2E-ASR=81.55% — IMPROVEMENT of +1.94pp over baseline 79.61%. Above upper CI bound 81.011%. Excluding sink-dominated heads improved steering effectiveness by removing heads that allocate >50% attention to special tokens (CLS/SEP/PAD/UNK). This is a genuine, statistically significant improvement.
