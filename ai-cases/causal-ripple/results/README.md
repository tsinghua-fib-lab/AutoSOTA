# Recorded results

- `paper_results.json`: complete-task counts, selected settings, and SHA-256 hashes of the archived score files.
- `outcomes.json`: per-task pass flags for the methods shown in the README, including the corresponding Visible-Cascade inputs.
- `release_validation.json`: CPU compatibility cases, A100 replays, public-input comparisons, official scorer checks, and the installed CLI smoke run.
- `bigcodebench_groundtruth_times.jsonl`: reference execution timings used by the pinned BigCodeBench scorer. The file contains task IDs and timings only.

Historical HumanEval and MBPP outcomes use harness document IDs. The other benchmarks use official task IDs. Every method within a benchmark covers the same complete task set. DS-1000 CORE scores combine the disjoint 80-task and 920-task archived partitions.

The main DS-1000 Causal-Ripple configuration solves 55 tasks with 16 verification candidates. Its separately recorded cascade input uses 32 candidates and solves 52 tasks; Visible-Cascade increases that input to 53. The main and auxiliary configurations are recorded separately in `paper_results.json`.
