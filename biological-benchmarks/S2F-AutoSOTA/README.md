# AutoSOTA S2F protein fitness study

An archive of AutoSOTA experiments extending **S2F**, the sequence/structure model
from *Multi-Scale Representation Learning for Protein Fitness Prediction*
(NeurIPS 2024), on **ProteinGym v1.3**. The working method identifier is
`S2F_ESM1v_PSSM`; the final publication name and author metadata remain pending.

[Research report](docs/research_report.md) · [Search history summary](docs/search_history.md) · [Aggregate results](results/summary.json) · [Source provenance](docs/provenance.md) · [Integration checks](docs/validation.md)

## Results

| Evaluation | Published S2F | AutoSOTA | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| All 217 substitution assays · function/UniProt-balanced Spearman | 0.45560865 | **0.46633717** | +0.01072852 | **+2.35%** |
| 204 non-Binding assays · arithmetic mean of per-assay Spearman | 0.4802745 | **0.5040833** | +0.0238088 | **+4.96%** |

The full run covered **217 assays and 2,465,767 mutants** on September 24–25,
2026. At the published per-assay table's three-decimal precision, 145 assays
improved, two tied, and 70 declined. The grouped primary score and the arithmetic
mean have different aggregation rules and must be reported separately.

Settings were selected using experimental labels from ProteinGym subsets, and
the non-Binding full-set results were inspected during development. This is
benchmark-adaptive development, even though the portable inference entry point
discards experimental labels. The comparison is against **original S2F**; it does
not establish overall ProteinGym SOTA, a global rank, or superiority to S2F-MSA.

## Method

The final method combines a locally trained S2F structure checkpoint, an ensemble
of five published ESM-1v models, and a Henikoff-weighted MSA profile. The effective
S2F / masked ESM-1v / wild-type ESM-1v weights are **0.56 / 0.14 / 0.30**.
An MSA-depth-dependent gate controls the profile contribution. The structure
branch continues to use frozen ESM-2-650M features.

The checkpoint is the local `model_best.pth`, **not** the original author's
`s2f.pth`. Its expected SHA256 is
`4942dea7cff7290b30f0ee21f8936ea790ba07d7cc87a6003eab3959f7a5759e`.
See the [model card](code/s2f_217_scoring/code/MODEL_CARD.md) for the fixed recipe
and development-data exposure.

## Code and reproduction

The recommended inference entry point is
[`code/s2f_217_scoring/code/scripts/score.py`](code/s2f_217_scoring/code/scripts/score.py).
It accepts explicit data, structure, MSA, checkpoint, and output paths. Official
metrics are computed separately by
[`scripts/run_official.py`](code/s2f_217_scoring/code/scripts/run_official.py),
using the preserved ProteinGym evaluation scripts.

This is a **source-only archive**. To reproduce the experiment, obtain the
external ProteinGym v1.3 assay CSVs, structures, alignments, matching assay
reference metadata, and all eight weight files listed in
[`reference/weights.json`](code/s2f_217_scoring/code/reference/weights.json).
The local trained checkpoint has no public download URL in the supplied package.
It must be supplied separately and match the recorded hash.

The supplied package also omits the two metadata CSVs required by the launchers:
`reference/DMS_substitutions.csv` for inference and
`reference/DMS_substitutions_official.csv` for official evaluation. Obtain the
matching v1.3 reference metadata from the original run or upstream resources and
place them at those paths. Metadata must preserve the assay identifiers,
filenames, sequences, structure/MSA mappings, and grouping fields used by the run.
Do not assume a later benchmark snapshot gives identical results.

The recorded inference environment is Linux, Python 3.8, PyTorch 1.8.1+cu111,
and CUDA/PyG extensions. Exact environment exports and an installation recipe
are included. Fresh installation of these older dependencies has not been
validated during this integration.

```bash
cd biological-benchmarks/S2F-AutoSOTA/code/s2f_217_scoring/code
python scripts/score.py --help

# After preparing the recorded environment, metadata, data, and weights:
export DMS_DATA=/path/to/DMS_ProteinGym_substitutions
export STRUCTURES=/path/to/ProteinGym_AF2_structures
export MSA_DATA=/path/to/DMS_msa_files
export WEIGHTS=/path/to/weights
export SCORES=/path/to/scores/S2F_ESM1v_PSSM
export GPU=0
sh scripts/score_all.sh

python scripts/run_official.py --scores "$SCORES" --data "$DMS_DATA" \
    --output /path/to/official_metrics
```

The [scoring package guide](code/s2f_217_scoring/code/README.md) provides the full
workflow. Its publication and checkpoint-hosting placeholders remain unresolved.
The source report records four representative GPU checks of the portable entry
point; it does not claim that this entry point reran all 217 assays. The historical
full benchmark was not rerun during repository integration.

## Archive layout

```text
README.md                       study overview and reproduction requirements
docs/research_report.md          English translation of the supplied report
docs/search_history.md           seven-round summary; individual logs not supplied
docs/provenance.md               source identity and integration changes
results/summary.json             aggregate values transcribed from the report
code/s2f_204_experiments/         historical training and inference experiments
code/s2f_217_run/                 original full-run launcher and environment
code/s2f_217_scoring/code/        portable scoring package and vendored evaluation
code/s2f_217_scoring/integration/ supplied ProteinGym contribution overlay
code/plot_scripts/              historical plotting scripts
SHA256SUMS                      checksums for the integrated archive
```

Historical launchers and plotting/packaging helpers retain assumptions about the
original machine and external files. They are provided for traceability; use the
portable scoring package for new runs. The contribution overlay is an archived
draft, not evidence of ProteinGym acceptance or a submitted pull request.

## Attribution and licensing

The source derives from [DeepGraphLearning/S3F](https://github.com/DeepGraphLearning/S3F).
Original MIT notices, DeepMind Apache-2.0 notices, and the ProteinGym license are
preserved. See [third-party notices](code/s2f_217_scoring/code/THIRD_PARTY_NOTICES.md).
External data and weights retain their upstream terms. Researchers supplied the
idea list and acceptance constraints; AutoSOTA implemented, searched, and
evaluated the candidates. The report does not attribute all ideas to autonomous
discovery.
