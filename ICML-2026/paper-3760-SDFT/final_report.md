# Final Report: paper-3760

- Title: Self-Distillation Enables Continual Learning
- Primary metric: `New Task Accuracy` (higher)
- Records: 11
- Generated: 2026-07-10T22:38:02Z

## Best Result

- Iteration: 2
- Idea: ALGO-8-LORA — LoRA SFT from SDFT checkpoint
- Primary metric: 0.6844
- Commit: `d90fa8b08f25cd097d237ce0a949f9f2d0e55fd3`
- Notes: LoRA SFT fine-tuning (r=8, alpha=16) for 1 epoch from the SDFT checkpoint. Training completed in ~7 min on 2674 examples. First eval: 68.64% (348/507), second eval: 68.24% (346/507). Average: 68.44%. This beats the baseline of 67.85% by +0.59pp. LoRA trains only ~1% of parameters (attention projections q,k,v,o), dramatically reducing memory from >80GB to ~25GB on single GPU. Previous Tasks Avg not evaluated (requires lm-evaluation-harness across 6 benchmarks). Risk of forgetting is low given minimal parameter change.
