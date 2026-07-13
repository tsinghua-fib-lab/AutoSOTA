# Final Report: paper-4980

- Title: Search Space Synthesis for Parametric Functions
- Primary metric: `NMSE` (lower)
- Records: 7
- Generated: 2026-07-12T15:46:09Z

## Best Result

- Iteration: 6
- Idea: PARAM-refined — log1p_clip + alpha=1e-3 + no-normalize-y fine-tuned
- Primary metric: 0.180963
- Commit: `aec9f251521fc4b1b40687a80ca5326432c031b7`
- Notes: Refined: log1p_clip + no-normalize-y + alpha=1e-3. NMSE=0.181 (-31.7% vs baseline 0.265). R2=0.818 (+6.4% vs baseline 0.769). Optimal config found: --y-transform log1p_clip --no-normalize-y --alpha 1e-3.
