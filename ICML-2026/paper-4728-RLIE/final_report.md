# Final Report: paper-4728

- Title: RLIE: Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation for Large Language Models
- Primary metric: `Accuracy` (higher)
- Records: 3
- Generated: 2026-07-12T22:06:15Z

## Best Result

- Iteration: 2
- Idea: ALGO-P0-02 — Diverse stratified hard example mining with TF-IDF clustering
- Primary metric: 80.87
- Commit: `89ecff74a296c922cde48123115dbc8a05d12ff3`
- Notes: Added TF-IDF + k-means clustering to select_hard_samples for diverse failure modes. Stratified by true class, equal allocation, picks hardest per cluster. Mean of 3 repeats: Acc=80.87% (80.0, 81.2, 81.4), F1=80.69%. +0.77% Accuracy over baseline. Reduced std from 1.62 to 0.76 — lower variance across repeats.
