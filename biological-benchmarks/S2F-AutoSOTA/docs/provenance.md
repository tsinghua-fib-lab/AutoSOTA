# Source provenance and integration record

**Integrated into AutoSOTA: October 12, 2026.**

## Supplied evidence

The supplied `S2F_AutoSOTA` directory contained a Chinese Markdown research report
dated October 11, 2026 and **114 source files** under `code/`, totaling less than
1 MB. It contained no datasets, model weights, per-mutation predictions, reference
assay CSVs, or run logs.

The original report SHA256 is:

```text
43140c59f1bf807347025e2d12a0d80422ef2f3830f9c04b0aba84c173a960c8
```

The Chinese original is retained outside this repository. The published report
is an English Markdown translation; no Word or binary report files are included.
The [source manifest](../results/source_manifest.json) records original and
integrated SHA256 values for every supplied code file.

## Integration changes

- Preserved the original source directory structure, scripts, configurations,
  vendor implementations, environment exports, and third-party license notices.
- Added the English research report, a seven-round search summary, the study and
  benchmark-directory READMEs, aggregate-result JSON, and archive checksums.
- Edited three imported READMEs only: the historical development README now
  identifies the snapshot and points missing figure/result assets to upstream;
  both scoring-package READMEs link the report and clarify that the checkpoint
  and assay metadata CSVs are external.
- Updated both main-project READMEs with one biological benchmark row and an
  Experiment News entry. The primary row uses the full-217 grouped Spearman,
  rounded to six decimal places, rather than the non-Binding arithmetic mean.

All **61 Python files and seven shell scripts** retain their supplied bytes.
The numerical implementation, model weights, and inference configuration were
not changed. No benchmark run or model training was performed during integration.

## Result provenance

[`summary.json`](../results/summary.json) is a structured transcription of the
supplied report. It is not a new metric output file. Raw prediction and official
performance tables were not supplied, so the scores and win/tie/loss counts have
not been independently recomputed here.

The report distinguishes published original S2F from a historical local
checkpoint reevaluation. The authoritative README comparison is the published
full-217 grouped Spearman, **0.45560865 → 0.46633717**. The report also preserves
the non-Binding subset, historical baselines, negative category results, seed
variation, and selection on benchmark labels.

The archived vendor-source manifest names S3F commit
`8a05a9a6e32a656e9d2747dcf944c8f68a7a084d`, a ProteinGym archive hash, and hashes
for 16 vendor files. Those 16 hashes match the supplied files. The exact local
checkpoint identity is recorded in the [weight manifest](../code/s2f_217_scoring/code/reference/weights.json);
the checkpoint itself and its hosting URL are absent.

## Verify the archive

From this study directory:

```bash
shasum -a 256 -c SHA256SUMS
```

The checksum file covers every study file except itself. The source manifest
separately makes the three documentation-only changes explicit.
