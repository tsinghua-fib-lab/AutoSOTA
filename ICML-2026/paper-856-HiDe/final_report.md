# Final Report: paper-856

- Title: HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
- Primary metric: `Attr` (higher)
- Records: 9
- Generated: 2026-07-05T21:42:43Z

## Best Result

- Iteration: 6
- Idea: PARAM-1 — sigma=2 (sharper attention) with [8,12,20]
- Primary metric: 87.0
- Commit: `61bdc6a8c2c01b03c349bac429efb57cc268088b`
- Notes: Reducing sigma from 3 to 2 (less Gaussian smoothing) with [8,12,20] improves Attr from 86.1%% to 87.0%% (+0.9pp) while maintaining Spatial at 72.4%%. Avg 81.2%% (+0.6pp vs sigma=3). Sharper attention maps provide more precise localization for attribute recognition. NEW BEST.
