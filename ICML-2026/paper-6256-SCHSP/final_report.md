# Final Report: paper-6256

- Title: Supervised Classification Heads as Semantic Prototypes: Unlocking Vision-Language Alignment via Weight Recycling
- Primary metric: `Accuracy` (higher)
- Records: 8
- Generated: 2026-07-18T07:41:29Z

## Best Result

- Iteration: 3
- Idea: ALGO-1+CODE-10+CODE-9 — InfoNCE + Prompt Ensemble + I0Tpost Standardization
- Primary metric: 79.16
- Commit: `7f185a18e0f7dd87d2b6223146d25334950e69be`
- Notes: InfoNCE (tau=0.07) + 3-template prompt ensemble + I0Tpost post-hoc embedding standardization. +0.83pp over without I0Tpost, +2.77pp over baseline 76.39%. Zero-cost inference-only change. Both text and image embeddings zero-centered per batch.
