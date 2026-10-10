# Medical benchmark studies

This directory archives AutoSOTA experiments on medical imaging benchmarks.
Each study includes an English Markdown report, its evaluation protocol, aggregate
results, and the supplied source code.

| Benchmark | Task | Released baseline | AutoSOTA | Study |
|---|---|---:|---:|---|
| ODELIA Breast MRI Challenge 2025 | Healthy / benign / malignant breast-ROI classification | 0.876448 | 0.899644 | [Report](ODELIA-AutoSOTA/docs/research_report.md) · [Code and results](ODELIA-AutoSOTA/) |

Scores are the mean of micro-AUC and two micro-ROC operating-point metrics.
The ODELIA result is a same-cohort comparison on the published public test set.
Public-test feedback informed candidate selection, and released encoder training
may overlap that cohort. It is not a new official hidden-test leaderboard result.
See the report for the full scope and limitations.
