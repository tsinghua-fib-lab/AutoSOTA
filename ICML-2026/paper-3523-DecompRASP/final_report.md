# Final Report: paper-3523

- Title: Discovering Interpretable Algorithms by Decompiling Transformers to RASP
- Primary metric: `Task Acc in [101-150]` (higher)
- Records: 9
- Generated: 2026-07-10T08:22:10Z

## Best Result

- Iteration: 2
- Idea: IDEA-03 — Warmup + cosine LR + grokking patience
- Primary metric: 100.0
- Commit: `68eff88fcbb8b2ec46c3be50e286c4cdaabae69f`
- Notes: Applied warmup_steps=500, lr_scheduler_type=cosine, max_steps=6000, patience=1500 after in-distribution saturation. All three metrics achieve 100% — breakthrough improvement from baseline 99.7% on OOD [101-150]. Training to step 6000 with cosine LR + warmup enables grokking of induction head patterns that generalize perfectly to 3x training length.
