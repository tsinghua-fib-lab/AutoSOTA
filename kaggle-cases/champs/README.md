# CHAMPS: Predicting Molecular Properties — AutoSOTA case

[Research report](REPORT.md) · [Source code](./)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| Mean per-type log MAE ↓ | −3.32551 | **−3.32019** | `57036892` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

An equal blend of the reproduced winning ensemble output and a released runner-up single-model prediction.

## Reproduction

The retained entry point is `reproduce_best.py`. Supply the two external component prediction files. Replay validates their hashes before blending. Full model inference also requires external weights, molecular inputs, and the upstream manual bond-order corrections, which are excluded as data.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
