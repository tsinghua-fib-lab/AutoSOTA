# Open Problems: Single-Cell Perturbations — AutoSOTA case

[Research report](REPORT.md) · [Source code](best_code/)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| MRRMSE ↓ | 0.521 | **0.721** | `57126892` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

Consensus-relative residual transfer with B-cell correction and Myeloid abstention (RAMRT).

## Reproduction

The retained entry point is `best_code/experiments/build_ramrt_b_abstention_candidate.py`. Prepare competition training data, a TabPFN v2 checkpoint, test-row mappings, and the frozen base submission externally. The code retains its original /workspace/c09-single-cell path assumptions.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
