# Final Report: paper-2356

- Title: LLM4Branch: Large Language Model for Discovering Efficient Branching Policies of Integer Programs
- Primary metric: `Time` (lower)
- Records: 7
- Generated: 2026-07-09T14:40:14Z

## Best Result

- Iteration: 1
- Idea: CODE-02 — Add pseudocost_product feature (feat 43) with 0.01 weight
- Primary metric: 8.3
- Commit: `2dd71e798e8896edd71ce73acf7ed938ac0fa66a`
- Notes: Added feature 43 (pseudocost_product) from cross-benchmark analysis. Initial weight 0.01. Geometric mean time improved 4.2% (8.66→8.30), nodes improved 2.2% (419.96→410.77). Gap 0.0 for all instances. Score function remains linear to avoid SCIP segfault issues.
