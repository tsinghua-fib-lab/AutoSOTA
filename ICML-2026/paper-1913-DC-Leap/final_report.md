# Final Report: paper-1913

- Title: DC-Leap: Training-Free Acceleration of dLLMs via Draft-Guided Contiguous Leaping Decoding
- Primary metric: `Accuracy` (higher)
- Records: 10
- Generated: 2026-07-07T18:45:33Z

## Best Result

- Iteration: 7
- Idea: PARAM-01 — Moderate commit_thres=0.68 draft_thres=0.975
- Primary metric: 43.0
- Commit: `b3aebb0e85c677b0e1fd98d3182aeeac182c4fa8`
- Notes: PARAM: commit_thres=0.68, draft_thres=0.975. Good tradeoff: TPS +4.2% with only -0.6% accuracy regression. On the Pareto frontier between baseline (best accuracy) and iter-2 (best TPS).
