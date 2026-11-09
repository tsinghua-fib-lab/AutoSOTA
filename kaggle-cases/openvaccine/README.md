# OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction — AutoSOTA case

[Research report](REPORT.md) · [Source code](code/)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| MCRMSE ↓ | 0.22859 | **0.33917** | `57170018` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

Cross-family pseudo-labeling between Keras and structure-aware Transformer models, masked-node pretraining, and equal final fusion.

## Reproduction

The retained entry point is `code/scripts/train.py`. Reconstruct the external dataset with code/scripts/build_data.py and prepare EternaFold and the separate TensorFlow/PyTorch runtimes. The selected eight-experiment dependency chain is documented in the report.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
