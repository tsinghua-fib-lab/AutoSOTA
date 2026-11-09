# LEAP / ClimSim — AutoSOTA case

[Research report](REPORT.md) · [Source code](./)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| Weighted R² ↑ | 0.79431 | **0.79165** | `57180922` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

Constrained humidity-conditioned ensembling, three small-learning-rate fine-tuned members, and state-dependent residual calibration.

## Reproduction

The retained entry point is `run_best.py`. The external asset manifest assets.json lists inputs required by cached/inference/training profiles; it contains paths and hashes, not payloads. Learned calibration and normalization files must be supplied externally.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
