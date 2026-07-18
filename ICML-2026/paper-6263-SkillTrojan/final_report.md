# Final Report: paper-6263

- Title: SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems
- Primary metric: `ASR` (higher)
- Records: 7
- Generated: 2026-07-16T15:19:56Z

## Best Result

- Iteration: 5
- Idea: ALGO-01b — Redundant fragments M=2 (two tools carry full-payload fallback)
- Primary metric: 87.5
- Commit: `ed526ebc791c32d1e0deb1199cd9ba029c79b71d`
- Notes: Increased redundant fragments from M=1 to M=2. Two tools (schema_analyzer and query_builder) now carry full-payload fallback fragments. ASR improved from 75% (iter-1, M=1) to 87.5% — significant improvement over baseline 50%. ACC also improved to 95.31%. Only 2/16 poisoned samples failed (87.5% success rate).
