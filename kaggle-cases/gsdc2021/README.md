# Google Smartphone Decimeter Challenge 2021 — AutoSOTA case

[Research report](REPORT.md) · [Source code](./)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| Position error (m) ↓ | 2.53616 | **1.12943** | `56804334` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

Shared multi-phone trajectories with 1 s/5 s TDCP, base-station differential code, road constraints, and date-grouped bias calibration.

## Reproduction

The retained entry point is `run_best.py`. Prepare raw GNSS/reference data, run prepare_best.py to regenerate observations and fitted bias coefficients, then run run_best.py. Linux build/runtime assumptions remain in the supplied scripts.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
