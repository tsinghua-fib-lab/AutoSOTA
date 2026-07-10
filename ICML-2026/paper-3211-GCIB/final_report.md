# Final Report: paper-3211

- Title: GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation
- Primary metric: `HR@10` (higher)
- Records: 8
- Generated: 2026-07-10T02:01:55Z

## Best Result

- Iteration: 6
- Idea: IDEA-07+IDEA-10 — Learnable layer weights + dropout + temperature annealing
- Primary metric: 0.1698
- Commit: `364f85d4d8298c54a46d0ac13893efc724810d10`
- Notes: IDEA-07 (learnable layer weights + layer dropout) combined with IDEA-10 (temperature1 annealing from 1.0→0.2 over 20 epochs). Best at epoch 20, early stop at epoch 40. HR@10=0.1698 (+2.8% vs baseline 0.1651). Marginal +0.0002 improvement over IDEA-07 alone. All guardrail metrics preserved (NDCG@10=0.0979, +3.7% vs baseline). Temperature annealing provides only tiny additional benefit on top of learnable layer weights.
