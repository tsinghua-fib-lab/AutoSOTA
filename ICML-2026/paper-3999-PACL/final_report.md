# Final Report: paper-3999

- Title: Probably Approximately Correct Labels
- Primary metric: `Budget save (%)` (higher)
- Records: 10
- Generated: 2026-07-11T09:32:53Z

## Best Result

- Iteration: 5
- Idea: IDEA-08b — Class-enhanced uncertainty with alpha=0.5 blend
- Primary metric: 62.73
- Commit: `1cf3004320f7c314b255084240d66c4f2a45957c`
- Notes: Same as IDEA-08 but with alpha=0.5 (equal weight to confidence and class-difficulty uncertainty). Budget save +0.20% vs iter 4 (62.53 to 62.73), Error +0.198% (4.572 to 4.770) but still within 5.003% guardrail. Correlation between signals only 0.377 so equal weighting captures more complementary information. Metrics parsed from stdout.
