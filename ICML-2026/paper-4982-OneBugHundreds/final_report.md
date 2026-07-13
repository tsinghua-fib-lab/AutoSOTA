# Final Report: paper-4982

- Title: One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery
- Primary metric: `Precision` (higher)
- Records: 7
- Generated: 2026-07-13T02:55:47Z

## Best Result

- Iteration: 2
- Idea: I-04-v2 — Refined parser: balanced extract_final_verdict with first-word priority
- Primary metric: 0.897
- Commit: `iter-2-I-04-v2`
- Notes: I-04-v2: Refined extract_final_verdict() with first-word priority, JSON parsing, bold markers, and last-line extraction. Precision +1.27%, Recall unchanged at baseline, PAcc +1.30%. Clean improvement over baseline with no regression. Replaces v1 which had Recall -1.56%.
