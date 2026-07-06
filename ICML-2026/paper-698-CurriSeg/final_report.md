# Final Report: paper-698

- Title: Refining Context-Entangled Content Segmentation via Curriculum Selection and Anti-Curriculum Promotion
- Primary metric: `M` (lower)
- Records: 7
- Generated: 2026-07-06T11:34:44Z

## Best Result

- Iteration: 2
- Idea: ALGO-01b — H-flip TTA + sigmoid temperature T=0.45
- Primary metric: 0.029003
- Commit: `2741cb61f87d9a0e0b9607a788f1d245ed52d93e`
- Notes: H-flip TTA ensemble with sigmoid temperature 0.45. M improved from 0.029308 (TTA only) to 0.029003 (-1.0% further). Temperature sweep tested 0.35-2.0; T=0.45 optimal. Fbeta: 0.737 (+2.3% from baseline 0.721), Ephi: 0.906 (+1.6%), Salpha: 0.835 (+0.4%). Cumulative M improvement: -5.57% from baseline 0.030715.
