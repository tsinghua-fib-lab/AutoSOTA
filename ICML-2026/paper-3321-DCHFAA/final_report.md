# Final Report: paper-3321

- Title: Detecting Contextual Hallucinations in Large Language Models with Frequency-Aware Attention
- Primary metric: `AUROC` (higher)
- Records: 10
- Generated: 2026-07-10T03:40:06Z

## Best Result

- Iteration: 7
- Idea: CODE-01+PARAM-01 — L1 C=0.5 + sliding_window=9
- Primary metric: 0.975
- Commit: `be1cf105c03ee5478fbdb280484fbe416fa325b7`
- Notes: BEST RESULT: L1 C=0.5 + SW=9. AUROC=0.9750 (+6.0% over observed baseline, +11.7% over manifest baseline 0.8584). F1=0.8495 (+6.4% over baseline). Both metrics dramatically improved. L1 sparsity + larger context window both contribute.
