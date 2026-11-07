# AutoSOTA Skill Evolution

This case studies **failure attribution and cross-trajectory generalization** for
agent skills. Round 1 distinguishes missing or defective skill instructions from
failures to follow an existing rule. Round 2 uses successful and unsuccessful
trajectories to propose reusable constraints, then validates candidate revisions.

[Research report](final_report.md) · [Implementation](best_code/) · [Final skills](best_code/final_skills/) · [Evaluation evidence](best_code/evidence/) · [Integration record](INTEGRATION.md)

## Results and evidence

The supplied report attributes the following four-stage comparison to Figure 9.
Scores are task accuracy or episode success, with higher values better.

| Benchmark | Initial | SkillOpt | Round 1 | Round 2 | Archived Round 2 evidence |
|---|---:|---:|---:|---:|---|
| SearchQA | 0.7300 | 0.7807 | 0.8041 | **0.8193** | 1,147 / 1,400; [summary](best_code/evidence/searchqa/eval_summary.json) |
| SpreadsheetBench | 0.4321 | 0.5179 | 0.6631 | **0.7359** | Reported endpoint; final skill and per-example evaluation not supplied |
| DocVQA | 0.8289 | 0.8690 | 0.8847 | **0.8930** | 334 / 374; [summary](best_code/evidence/docvqa/eval_summary.json) |
| ALFWorld | 0.7090 | 0.8060 | 0.8351 | **0.8582** | 115 / 134; [summary](best_code/evidence/alfworld/eval_summary.json) |

The three saved per-example score files support recalculating the corresponding
Round 2 endpoints. The other stage values and the SpreadsheetBench endpoint are
reported values from the supplied comparison, rather than newly reproduced
measurements. The evaluation summaries identify their split as `valid_unseen`.
Rejected-candidate scores in the report refer to development validation and are
separate from those endpoints.

## Code and reproduction

[`best_code/`](best_code/) preserves the latest supplied SkillOpt implementation,
including both optimization rounds, benchmark adapters, configurations, final
skills, and evaluation records. Historical records identify multiple code
revisions; the latest snapshot should not be assumed to match every historical
source hash. Per-record hashes are retained for traceability.

Install the dependencies in `best_code/requirements.txt` in a separate environment.
From `best_code/`, the supplied entry points are:

```bash
python -m scripts.train --help
python -m scripts.consolidate_algo03 --help
python -m scripts.eval_only --help
```

Training and new evaluations require external benchmark data and a configured
model provider. This archive does not include model credentials or benchmark
payloads. No provider calls or full benchmark runs were made during integration.
The original MIT license is retained in [`best_code/LICENSE`](best_code/LICENSE).
