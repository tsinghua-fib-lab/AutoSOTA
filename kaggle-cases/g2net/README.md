# G2Net Gravitational Wave Detection — AutoSOTA case

[Research report](REPORT.md) · [Source code](./)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| ROC-AUC ↑ | 0.88581 | **0.88540** | `57159613` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

Complementary 1D detector models, physical-signal training, targeted continuation, and calibrated logit ensembling.

## Reproduction

The retained entry point is `scripts`. The scripts and upstream implementations are retained; waveform data, whitening arrays, trained checkpoints, fitted calibration files, and prediction caches must be prepared externally.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
