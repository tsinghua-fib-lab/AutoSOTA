# Biological benchmark studies

This directory archives AutoSOTA experiments on biological benchmarks. Each study
includes an English Markdown report, the evaluation protocol, aggregate results,
and the supplied source code.

| Benchmark | Task | Published baseline | AutoSOTA | Study |
|---|---|---:|---:|---|
| ProteinGym v1.3 · S2F | Protein fitness prediction across 217 substitution assays | 0.45560865 | 0.46633717 | [Report](S2F-AutoSOTA/docs/research_report.md) · [Code and results](S2F-AutoSOTA/) |

The scores use ProteinGym's function/UniProt-balanced Spearman aggregation.
Benchmark subset labels informed method selection; the study does not establish
performance on an untouched test set or an overall leaderboard rank. See the
report for baseline provenance, subset analyses, and reproduction requirements.
