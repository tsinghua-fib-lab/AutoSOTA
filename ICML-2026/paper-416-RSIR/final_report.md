# Final Report: paper-416

- Title: Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control
- Primary metric: `ndcg@10` (higher)
- Records: 14
- Generated: 2026-07-04T19:49:05Z

## Best Result

- Iteration: 8
- Idea: ALGO-03 — Gradient clipping (max_norm=1.0) + neg_num=512
- Primary metric: 0.0296
- Commit: `e83396f48e6d8cc3501dc343cd29c03f9f0a5c7a`
- Notes: ALGO-03: gradient clipping + neg_num=512 with anomaly detection ON. First improvement! +1.0% ndcg@10, all 5 metrics improved. Anomaly detection MUST be ON.
