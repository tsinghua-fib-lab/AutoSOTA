# Final Report: paper-4906

- Title: Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm That Provably Exploits Model Similarity
- Primary metric: `Identification Accuracy` (higher)
- Records: 9
- Generated: 2026-07-12T17:42:17Z

## Best Result

- Iteration: 3
- Idea: IDEA-09 — Budget rounding fix: round up instead of down
- Primary metric: 95.5
- Commit: `a65cd68f22f9c3bc72383a5ce52f76dec1e8d1ea`
- Notes: Fixed budget calculation to round up instead of down. Recovered up to K-1 wasted pulls per dataset. 95.5% vs 95.2% baseline (+0.3%). Biggest gains on narrative_qa (+2.0%, 82.9→84.9) and mmlu (+0.3%, 96.9→97.2). Small task-pool datasets benefit most from recovered budget.
