# Final Report: paper-5213

- Title: Formalizing and Falsifying Causal Pathways of Rare Events
- Primary metric: `Pathway Explanation Score` (higher)
- Records: 10
- Generated: 2026-07-13T05:26:02Z

## Best Result

- Iteration: 6
- Idea: ALGO-04 — Refined A->D->E: P(D|A)=0.85, P(E|D)=0.22
- Primary metric: 0.782
- Commit: `bfd252528221db2982c71ee6f0c44e201140c6ab`
- Notes: Sensitivity-weighted refinement on best pathway structure A->D->E. P(D|A)=0.85 from Marwaha & Johnson (2004) meta-analysis (80-90% schizophrenia unemployment). P(E|D)=0.22: +10 percent adjustment to LLM estimate based on long-term unemployment-homelessness association. Product=0.187. HUD 2024 P(E)=0.000457.
