# Final Report: paper-2301

- Title: Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs
- Primary metric: `perplexity` (lower)
- Records: 13
- Generated: 2026-07-08T23:18:44Z

## Best Result

- Iteration: 11
- Idea: PARAM-01 — rank=96 + block_size=4 + iter=2
- Primary metric: 15.7284
- Commit: `4606d75fd6764621533bc2938841bbaabae32761`
- Notes: Increased rank to 96 with block_size=4, iter=2. Result: 15.7284 perplexity (-3.915 vs baseline 19.643, -19.93%). Rank=96 at finest block size gives significant additional improvement over rank=64 (16.05). Adaptive k* selection now has room to allocate up to 96 per layer; some layers use ~50-65 of the budget. Total correction capacity: 50% more than rank=64 at same block_size.
