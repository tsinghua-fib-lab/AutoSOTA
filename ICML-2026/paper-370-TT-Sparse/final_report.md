# Final Report: paper-370

- Title: TT-Sparse: Learning Sparse Rule Models with Differentiable Truth Tables
- Primary metric: `AUC` (higher)
- Records: 13
- Generated: 2026-07-04T17:14:43Z

## Best Result

- Iteration: 5
- Idea: CODE-02 — Multi-Seed Rule Ensemble
- Primary metric: 0.8152
- Commit: `a7e095fadb693f97f89a763c37b1a9effe82fbee`
- Notes: Multi-seed ensemble: average probability from 5 independently trained models. Ensemble AUC=0.8152 (+0.0076 vs baseline), Complexity=5 (well within 50 threshold). Reduces variance by averaging out bad seeds.
