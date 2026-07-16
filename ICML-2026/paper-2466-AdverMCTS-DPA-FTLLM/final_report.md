# Final Report: paper-2466

- Title: Data Provenance Auditing of Fine-Tuned Large Language Models with a Text-Preserving Technique
- Primary metric: `Chunk Hit Probability (p)` (higher)
- Records: 12
- Generated: 2026-07-15T14:55:09Z

## Best Result

- Iteration: 7
- Idea: GEN-04 — max_new_tokens=300, temp=0.1
- Primary metric: 0.958333
- Commit: `5775f7ccb818dbec7caf8fb88798f991b16ced9c`
- Notes: Increased max_new_tokens from 200 to 300 with temp=0.1. Best adapter (model_A_d0_e5). p=0.958 (+0.300 from baseline). More generation budget helps complete additional watermark patterns.
