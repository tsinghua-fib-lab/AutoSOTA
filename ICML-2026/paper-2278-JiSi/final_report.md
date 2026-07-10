# Final Report: paper-2278

- Title: Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale
- Primary metric: `Accuracy` (higher)
- Records: 10
- Generated: 2026-07-09T09:21:09Z

## Best Result

- Iteration: 3
- Idea: idea-3 — Idea 2+3: Discriminative Filtering + Per-Dataset Prior Bias
- Primary metric: 87.78
- Commit: `df0286cad89ebaea5356888164d82eec2b475c94`
- Notes: Ideas 2+3 combined. Discriminative support filtering + per-dataset per-model accuracy prior (alpha=0.10). MMLU-Pro 87.78% (+0.89pp vs baseline 86.89%, +0.22pp vs Idea-2-alone 87.56%). MMLU-Pro cost $3.26 (vs $3.54 baseline). Overall 67.73%. Dataset prior provides small but consistent gain.
