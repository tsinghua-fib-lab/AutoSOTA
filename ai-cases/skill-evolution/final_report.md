# Final Report: AutoSOTA Skill Evolution case

- Title: Failure attribution and cross-trajectory generalization for self-evolving agent skills
- Primary metric: task accuracy for SearchQA, SpreadsheetBench, and DocVQA; episode success for ALFWorld (higher is better)
- Research rounds: 2
- Documented rejected development candidates: 5
- Report prepared: 2026-11-07

## Best Result

Figure 9 reports the following four-stage comparison.

| Benchmark        | Initial | SkillOpt | Round 1 | Round 2 |
| ---------------- | -------:| --------:| -------:| -------:|
| SearchQA         | 0.7300  | 0.7807   | 0.8041  | 0.8193  |
| SpreadsheetBench | 0.4321  | 0.5179   | 0.6631  | 0.7359  |
| DocVQA           | 0.8289  | 0.8690   | 0.8847  | 0.8930  |
| ALFWorld         | 0.7090  | 0.8060   | 0.8351  | 0.8582  |

Round 2 evaluation files:

| Benchmark | Skill in `best_code/final_skills/` | Held-out correct / total | Recomputed score | Original summary                                |
| --------- | ---------------------------------- | ------------------------:| ----------------:| ----------------------------------------------- |
| SearchQA  | `searchqa_round2.md`               | 1147 / 1400              | 0.8192857        | `best_code/evidence/searchqa/eval_summary.json` |
| DocVQA    | `docvqa_round2.md`                 | 334 / 374                | 0.8930481        | `best_code/evidence/docvqa/eval_summary.json`   |
| ALFWorld  | `alfworld_round2.md`               | 115 / 134                | 0.8582090        | `best_code/evidence/alfworld/eval_summary.json` |

## Research trajectory

Round 1, Failure-Attributed Skill Evolution, separates skill defects from failures to execute an existing rule. It applies bounded core-skill edits for the former, reinforces applicable existing instructions for the latter, and validates candidate revisions before adoption. The implementation is in `best_code/skillopt/optimizer/skill_aware.py` and the training engine.

Round 2, Cross-Trajectory Skill Generalization, compares residual failures with successful trajectories, clusters recurring behavioral deficiencies, compiles evidence-supported constraints, and validates them against the source skill. The implementation is in `best_code/skillopt/optimizer/algo03.py` and `best_code/scripts/consolidate_algo03.py`.

The located development records include these rejected attempts. Their scores are **development validation scores**, separate from the held-out Figure 9 endpoints:

| Attempt                              | Source validation hard | Candidate validation hard | Decision and evidence                                                                              |
| ------------------------------------ | ----------------------:| -------------------------:| -------------------------------------------------------------------------------------------------- |
| SearchQA trace compiler plus         | 0.7950                 | 0.7650                    | Rejected; `best_code/evidence/searchqa_rejected_candidate_summary.json` records `accepted: false`. |
| DocVQA evolution iteration 1 delta rule | 0.8868              | 0.8491                    | Rejected after validation regression.                                                              |
| DocVQA evolution iteration 1 LLM rewrite | 0.8868            | 0.8302                    | Rejected after validation regression.                                                              |
| DocVQA evolution iteration 2 contrastive candidate | 0.8868  | 0.8113                    | Rejected after validation regression.                                                              |
| DocVQA evolution iteration 2 delta rule | 0.8868              | 0.8679                    | Rejected after validation regression.                                                              |

The four DocVQA candidate records come from `best_code/evidence/docvqa_evolve_summary.json`. These are candidate-search iterations within Round 2. The iteration 1 contrastive candidate matched its source on validation and reached the DocVQA Round 2 endpoint on the held-out set.

## Archive evidence coverage

The SpreadsheetBench stage values above are reported values from the supplied
Figure 9 comparison. A final SpreadsheetBench skill and per-example evaluation
record were not included in the handoff. The saved Round 2 skills and individual
score records cover SearchQA, DocVQA, and ALFWorld. These records identify several
historical source revisions; `best_code/` is the latest supplied implementation.
No new model or benchmark run was performed during repository integration.
