# AI research cases

This directory archives three AutoSOTA research cases released on November 7,
2026. Each case includes an English Markdown report, the supplied implementation,
and recorded evaluation evidence. The experiments predate this repository update;
the integration does not represent a new training or benchmark campaign.

| Case | Evaluation | Selected result | Explore |
|---|---|---|---|
| **Causal-Ripple** | Frozen LLaDA-8B-Base on HumanEval, MBPP, MBPP+, DS-1000, and BigCodeBench | 40 / 146 / 84 / 55 / 73 successful tasks, respectively; the separate Visible-Cascade extension uses additional compute | [Report](causal-ripple/REPORT.md) · [Code](causal-ripple/) · [Recorded results](causal-ripple/results/paper_results.json) |
| **GRPO / PDA-Iso** | Qwen2.5-0.5B on Countdown; matched seed-0, 500-step runs, 1,024 evaluation examples | 523 correct and 53.92 mean output tokens, versus DrGRPO's 532 and 125.13: 56.91% shorter outputs with nine fewer correct answers | [Report](grpo/REPORT.md) · [Code overlay](grpo/overlay/) · [Recorded results](grpo/results/matched500.csv) |
| **Skill Evolution** | SearchQA, SpreadsheetBench, DocVQA, and ALFWorld | Reported Round 2 scores: 0.8193 / 0.7359 / 0.8930 / 0.8582; saved final skills and per-example records are available for three of the four tasks | [Report](skill-evolution/final_report.md) · [Code](skill-evolution/best_code/) · [Evidence coverage](skill-evolution/README.md#results-and-evidence) |

The reports retain unsuccessful candidates and evaluation qualifications. These
cases use different tasks and metrics and should not be combined into a single
ranking. Refer to each report for repeated benchmark exposure, compute accounting,
baseline definitions, and external reproduction requirements.

Source ZIP files and duplicate PDF figures are kept outside the repository.
Reports are published as Markdown; supplied PNG/SVG figures, licenses, result
tables, and source provenance are retained. See each case's `INTEGRATION.md` and
the [source manifest](source_manifest.json) for the archive mapping and file hashes.
