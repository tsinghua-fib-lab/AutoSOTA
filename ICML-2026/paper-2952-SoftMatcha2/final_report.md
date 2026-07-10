# Final Report: paper-2952

- Title: SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora
- Primary metric: `P@20` (higher)
- Records: 13
- Generated: 2026-07-09T15:56:39Z

## Best Result

- Iteration: 2
- Idea: IDEA-01 — Tuned LAMBDA interpolation weight to 0.7
- Primary metric: 42.8
- Commit: `87b0f99b45ffa05148562c7c4147e3fa1ef2d365`
- Notes: Swept LAMBDA in {0.3,0.5,0.6,0.7,0.8}. Best P@20=42.8 at LAMBDA=0.7 vs baseline 36.0. R@1000=27.9 still above guardrail (>25.46). Combination of min-max normalization + BM25-heavy interpolation (0.7) works best.
