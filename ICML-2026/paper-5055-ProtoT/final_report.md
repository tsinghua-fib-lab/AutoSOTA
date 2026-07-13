# Final Report: paper-5055

- Title: Prototype Transformer: Towards Language Model Architectures Interpretable by Design
- Primary metric: `test_perplexity` (lower)
- Records: 9
- Generated: 2026-07-12T16:00:28Z

## Best Result

- Iteration: 8
- Idea: PARAM-01 — WSD (Warmup-Stable-Decay) LR Schedule
- Primary metric: 85.74
- Commit: `3d883deadc8abd14458b2341719e7bddfaba0d6e`
- Notes: Replaced cosine decay with WSD schedule (80% stable at peak LR, 20% linear decay). test_ppl=85.74 vs baseline 91.27 (-5.53, -6.1%!). Sustained high LR allowed better prototype space exploration. Epoch 8 was still behind but epochs 9-10 showed dramatic improvement. Best result by far.
