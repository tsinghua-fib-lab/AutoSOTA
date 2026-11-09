# Novozymes Enzyme Stability Prediction — AutoSOTA case

[Research report](REPORT.md) · [Source code](code/)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| Spearman ↑ | 0.60035 | **0.58370** | `57143817` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

An 11-component structural ensemble with normal-score pooling and a distribution-corrected LightGBM twin.

## Reproduction

The retained entry point is `code/pipeline/run_pipeline.py`. Prepare external sequence/structure inputs and public component predictions. Released full-test feedback informed the search; see the report for evaluation exposure.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
