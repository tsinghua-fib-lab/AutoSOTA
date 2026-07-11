# Final Report: paper-3837

- Title: Identifying Common Hubs in Multiple Gaussian Graphical Models
- Primary metric: `F-score` (higher)
- Records: 13
- Generated: 2026-07-10T19:46:01Z

## Best Result

- Iteration: 1
- Idea: IDEA-01 — Softmax gradient relaxation in .objective.der()
- Primary metric: 0.7373
- Commit: `b4118969614eb8c60ce9b045d5cd8f46b9a81da8`
- Notes: Replaced hard argmax gradient selection with softmax-weighted sum of all K=3 group gradients (tau=5.0). F-score: 0.7177->0.7373 (+2.7%). Precision: 0.8765 (was 0.8898). Recall: 0.6580 (was 0.6260). Trade-off: slightly lower precision for much better recall.
