# GeoHab 2026 MLWG — AutoSOTA case

[Research report](REPORT.md) · [Source code](code/)

## Recorded result

| Metric | Public | Private | Submission reference |
|---|---:|---:|---|
| Weighted F1 ↑ | 0.88296 | **0.88691** | `57130319` |

These are archived scored submissions, not new evaluations performed during
repository integration. Post-competition scores do not change frozen official
standings or award a new competition rank. See the report for comparison scope,
baseline provenance, and development feedback.

## Method

LightGBM/CNN fusion with class reliability, tabular sharpening, and fold-aware spatial smoothing.

## Reproduction

The retained entry point is `code/pipeline/run_pipeline.py`. The source expects external raster features, frozen folds, and CNN outputs. See the retained reproduction notes in code/.

This archive includes source code, necessary runtime configuration, dependency
specifications, licenses, and English Markdown reports. It includes no competition
data, test labels, trained model weights, fitted parameter arrays, stored
predictions, submission files, raw run logs, or report figures. Notebook outputs,
execution counts, attachments, and transient metadata are cleared. Historical
artifact paths in the code and reports refer to external files.

No training, model inference, Kaggle scoring, or submission was performed while
preparing this archive. See integration and cleanup notes (external reference: `../INTEGRATION.md`).
