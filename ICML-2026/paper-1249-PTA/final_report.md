# Final Report: paper-1249

- Title: Prototype-Based Test-Time Adaptation of Vision-Language Models
- Primary metric: `top1_accuracy` (higher)
- Records: 9
- Generated: 2026-07-06T08:04:45Z

## Best Result

- Iteration: 5
- Idea: ALGO-3 — Prototype repulsion regularization
- Primary metric: 74.23
- Commit: `2925ad8ca153245ef471d49980cb2064e3afcb8c`
- Notes: Added periodic prototype repulsion (every 10 steps, threshold=0.7, strength=0.001-0.003). Pushes apart class prototypes with cosine similarity > 0.7. Small +0.03 gain over alpha=0.001 alone (74.20->74.23). Repulsion prevents prototype collapse for fine-grained action classes.
