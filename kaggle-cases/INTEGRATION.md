# Integration and cleanup record

**Date: November 9, 2026.** Eight supplied Kaggle case packages were organized
into this source-only archive. The original Desktop materials were not modified.
This update publishes existing research; it does not perform new training,
benchmark evaluation, leaderboard queries, or competition submissions.

## Included material

- Model, training, inference, preprocessing, and reproduction source code.
- Necessary runtime configuration, selected recipe constants, external-input
  manifests, dependency specifications, and original license notices.
- English Markdown case guides and research reports.
- Source notebooks with outputs, execution counts, attachments, and transient
  metadata removed.

Small scalar recipe settings, architecture definitions, and external-file hashes
are configuration, not checkpoint payloads. LEAP's asset manifest describes
external inputs but contains none of those inputs. CHAMPS replay metadata is
reduced to the row counts, expected hashes, and constants consumed by its code.

## Excluded material

Competition datasets and labels, row-mapping tables, generated pseudo-label
sequences, pretrained/trained checkpoints, serialized normalization or fitted
parameter arrays, OOF matrices, cached predictions, final submission files and
ZIPs, run logs, raw result JSON/CSV, figures, PDF/LaTeX exports, installed
environments, and compiled binaries are not published here.

Specific removals include C09's three OOF NPZ arrays, both submission ZIPs and
the test-row mapping CSV; OpenVaccine's four dataset/label JSON files and compressed
submission; and the raw metrics, run receipts, prediction metadata, and duplicate
snapshots throughout the handoff. CHAMPS' molecule-specific manual bond-order
corrections are treated as data and must be obtained externally when required.
The Novozymes fitted-weight dump is omitted; its selected scalar ensemble recipe
remains part of the runtime configuration.

The approximately 19 MB GeoHab notebook is retained as source cells only. All
43 published notebooks have empty outputs and no stored execution counts or
attachments. The complete source archive is approximately 8 MB, compared with
approximately 154 MB in the supplied directory, which also contains compressed
artifacts. These are file-size summaries, not a benchmark-compute comparison.

## Reports and evidence boundaries

The supplied C09 English report is retained with an archive-scope note. The LEAP,
CHAMPS, G2Net, and GSDC reports retain their existing full English sections;
duplicate Chinese text and bilingual appendices are not copied. GeoHab,
Novozymes, and OpenVaccine have English research summaries prepared from their
supplied Chinese reports. OpenVaccine's 18 experiment groups and 31 blend records
are transcribed into Markdown tables rather than uploaded as raw result files.

Historical artifact links are converted into external references when their
targets are deliberately excluded. Statements about assets retained in the
original environment do not imply that those assets are in this GitHub archive.
Each case guide identifies the external prerequisites for reproduction.

Late-submission scores do not change historical official ranks. Novozymes used
released full-test feedback during search; OpenVaccine's source protocol used
public-set selection and notes pseudo-label CV leakage. Other reports preserve
their own baseline, compute, and selection qualifications. These cases should
not be interpreted as eight new official competition wins or independent tests.

## Integration checks

| Check | Result |
|---|---|
| Numerical Python, shell, C, and header source compared with the supplied files | Unchanged |
| Plain Python syntax | 404 files parsed |
| Preserved upstream IPython export | One file; Python body parsed after excluding cell/shell magic lines |
| Source notebooks | 43; no outputs, execution counts, attachments, or transient metadata |
| JSON runtime configurations | Parsed and reviewed as configuration/external-file metadata |
| Shell scripts | Syntax checked |
| `leap/run_best.py --help` | Passed |
| `leap/prepare_workspace.py --help` | Passed |
| `champs/reproduce_best.py --help` | Passed |
| English-only text, local documentation links, and README mirror paths | Checked |
| Datasets, prediction/submission files, model binaries, embedded payloads, and common credential patterns | Excluded or absent in retained files |

The help checks do not load weights, download data, or submit to Kaggle. Legacy
CUDA frameworks, Linux build chains, model inference, and full training were not
run on the integration machine. Runtime paths in the original source may need
adaptation after external inputs and dependencies are prepared.
