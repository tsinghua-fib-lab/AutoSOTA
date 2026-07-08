# Final Report: paper-1887

- Title: Causal Preference Elicitation
- Primary metric: `SHD` (lower)
- Records: 7
- Generated: 2026-07-07T13:38:06Z

## Best Result

- Iteration: 6
- Idea: GFN-REJUV4-T50 — DAG-GFN prior + rejuvenate_steps=4 + T=50 queries
- Primary metric: 4.6
- Commit: `9a60666ca42388a07c074ff57bcff23401c28014`
- Notes: GFN prior with 4 rejuvenation steps and 50 oracle queries (vs 40 in iter 4). 5 seeds. SHD improved from 6.16 to 4.60 (25% further). OrientF1 improved from 0.777 to 0.838. Total improvement from baseline: SHD 15.09→4.60 (70%), OrientF1 0.600→0.838 (40%). More queries provide additional oracle feedback that further refines the posterior.
