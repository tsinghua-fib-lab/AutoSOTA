# Evidence and Provenance

[Research report](../REPORT.md) · [Machine-readable source index](../evidence/source_index.json)

The report is based on the local experiment archive and selected full-task outcomes. It is not a new literature survey. AutoSOTA's research contribution is described as a project-level finding; priority claims against all prior work are not inferred from these records.

## Evidence map

| ID | Evidence | Availability in this package |
|---|---|---|
| E01 | August exploration ledger: initial baseline reproduction, candidate matrix, graph/safety experiments, local repair, portfolios, and budget-neutral variants | English synthesis in REPORT.md and idea_catalog.md; original ledger checksum retained |
| E02 | September transfer diagnosis: prompt/scoring parity, unchanged edits, and transfer truncation | English synthesis in REPORT.md; original diagnostic checksum retained |
| E03 | DS-1000, BigCodeBench, and APPS full-set engineering runs | [Original English receipt](../evidence/E03.md) |
| E04 | Changed/ranked variants on BigCodeBench and APPS | [Original English receipt](../evidence/E04.md) |
| E05 | P24, global probes, APPS K2 and late revision | [Original English receipt](../evidence/E05.md) |
| E06 | Final BigCodeBench P48 experiment, paired outcomes, independent recheck | [Original English receipt](../evidence/E06.md) |
| E07 | DS-1000 recheck across selected candidates and baselines | [JSON receipt](../evidence/E07.json) |
| E08 | DS-1000 and BigCodeBench Cascade evaluation | [Original English receipt](../evidence/E08.md) |
| E09 | Complete GSM8K comparison | [Original English receipt](../evidence/E09.md) |
| E10 | Main archived full-task outcomes | [Updated result registry](../results/paper_results.json) and [Boolean outcomes](../results/outcomes.json); original registry checksum retained |
| E11 | Complete later baseline evaluation and audit | Full-task counts and source-score hashes merged into the result registry; audit checksum retained |
| E12 | Mechanism controls and inference-cost analysis | [Portable analysis data](../results/analysis_results.json); original data checksum retained |
| E13 | Recorded BigCodeBench/939 paired-probe measurements | [Full diagnostic JSON](../evidence/E13.json) |

Source receipts retain historical experiment names. An older receipt may have a stale batch heading; the source index records its actual archive location. Those headings do not determine which score is the final selected result.

## What can be checked without model inference?

`python scripts/verify_results.py` checks:

- Every main result against a full per-task Boolean outcome vector.
- Identical task IDs across methods within each benchmark.
- The strongest listed baseline, net task gains, and the 5.0% macro-average relative gain.
- Visible-Cascade counts against its corresponding input profile.
- Figure-analysis task counts against the final main table.
- Byte identity of selected runtime source files against the handoff snapshot manifest.

The historical exploration tables contain both full and subset results. Their denominators and evidence sources are stated explicitly. The compact package does not contain every historical generated program or rejected implementation; it contains the selected runnable implementation and enough structured outcomes to independently verify the final count arithmetic.

## Historical validation versus handoff validation

[release_validation.json](../results/release_validation.json) records the earlier release checks: deterministic sampler compatibility, A100 token/cost replays, public-input comparison, scorer checks, and CLI validation. These are historical receipts, not reruns made during packaging.

[handoff_validation.json](../evidence/handoff_validation.json) records the checks actually performed for this handoff. Full GPU benchmark reruns are not required to package archived results and were not performed.

## Data, licensing, and attribution

Model weights and benchmark payloads are obtained from their upstream sources using the preparation/reproduction workflow. Per-task outcome vectors contain task identifiers and Boolean correctness flags, not hidden tests. Code retains its MIT license, and external methods, datasets, and evaluators retain their original attribution in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

AutoSOTA is the automated experimentation system used for the project exploration. It is not presented as the originator of upstream CORE, LLaDA, ReMDM, or benchmark datasets.
