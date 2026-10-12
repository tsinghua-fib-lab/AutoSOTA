# S2F inference extension — publication draft

The working identifier is `S2F_ESM1v_PSSM`; the final method name, authors and
hosting URLs are pending in `publication.json`.

This package reproduces the scoring method used for the September 24–25, 2026
ProteinGym v1.3 run: 217 substitution assays, including Binding, and 2,465,767
mutants. It combines an S2F structure branch, five published ESM-1v models and
an MSA-derived frequency profile. Read [MODEL_CARD.md](MODEL_CARD.md) for the
development protocol and limitations.

## Results and evaluation scope

The independently recomputed function/UniProt-balanced Spearman is approximately
0.46634 (official three-decimal display: 0.466). The 204 non-Binding assay simple
mean is 0.50408; published S2F scores on those same assay IDs average 0.48027.
The 204-assay arithmetic mean is a separate analysis, not the leaderboard metric.
See the [archived study report](../../../docs/research_report.md) for all five metrics.

The development process used experimental labels from ProteinGym subsets to
select inference settings. These results do not constitute evaluation on a fully
untouched zero-shot test set. The requested benchmark placement must be agreed
with the ProteinGym maintainers. No leaderboard inclusion or overall SOTA is claimed.

## Environment

Linux, Python 3.8, PyTorch 1.8.1+cu111 and a CUDA-capable GPU are required for
inference. The original run used four A100 80 GB GPUs; the launcher below uses
one GPU. The package has been checked in the preserved run environment; fresh
installation of the following recipe still depends on availability of historical
CUDA/PyG wheels. The exact installed versions are in `environment/pip_freeze.txt`.

```sh
conda create -n s2f-release python=3.8 cudatoolkit=11.1.1 -c conda-forge
conda activate s2f-release
sh environment/install.sh
```

If a newer host driver needs the conda CUDA runtime explicitly, set
`LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`. Use the real PyG scatter
extensions; no substitute scatter implementations are included in this package.

## Weights

All eight required files, sizes and SHA256 values are listed in
`reference/weights.json`. ESM weights are downloaded from the original FAIR URLs.
`model_best.pth` is the exact local trained checkpoint used for this run, and
must be hosted with the release. **Do not substitute the original author's
`s2f.pth`: it is a different checkpoint.**

```sh
python scripts/download_weights.py --destination /path/to/weights \
    --checkpoint-url 'PUBLIC_CHECKPOINT_URL'
python scripts/download_weights.py --destination /path/to/weights --verify-only
```

The trained `model_best.pth` is not included in this source-only archive. Supply
the exact matching checkpoint separately; the downloader verifies it and downloads
the remaining published files. Its missing URL is a publication placeholder.

The required `reference/DMS_substitutions.csv` and
`reference/DMS_substitutions_official.csv` are also external. Prepare matching
ProteinGym v1.3 assay reference metadata before inference or official evaluation.

## Data

Obtain these resources from the [ProteinGym v1.3 resources](https://github.com/OATML-Markslab/ProteinGym#resources):

- `DMS_ProteinGym_substitutions.zip`
- `ProteinGym_AF2_structures.zip`
- `DMS_msa_files.zip`

Download URL pattern:
`https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/FILENAME`.
Point the launcher at the extracted folders containing the CSV, PDB and a2m files.
No surface data or precomputed MSA sequence weights are required. The profile
weights are recomputed from alignments. An empty cache is used for every new run,
so a cache from an earlier experiment cannot silently change predictions.

## Scoring

```sh
export DMS_DATA=/path/to/DMS_ProteinGym_substitutions
export STRUCTURES=/path/to/ProteinGym_AF2_structures
export MSA_DATA=/path/to/DMS_msa_files
export WEIGHTS=/path/to/weights
export SCORES=/path/to/scores/S2F_ESM1v_PSSM
export GPU=0
sh scripts/score_all.sh
```

For one assay, use `python scripts/score.py --help` and select `--DMS-index N`
(zero-based reference row) or `--assay-id DMS_ID` instead of `--all`. Repeated
`--assay-id` arguments permit sharding across GPUs. Output files are named by
DMS ID with columns `mutant,model_score`; higher score means greater predicted
fitness. Predictions follow the input mutant order. Logs and exact configurations
are kept under the output folder's `_runs/` directory.

The launcher discards experimental labels before inference and supplies dummy
zeros to the legacy diagnostic evaluator. Diagnostic correlations inside
`inference.log` are consequently undefined and must not be reported. Use the
separate official performance pipeline with the original labelled assay CSVs.

## Official metrics

CPU execution is sufficient; no model inference is repeated:

```sh
python scripts/run_official.py --scores "$SCORES" --data "$DMS_DATA" \
    --output /path/to/official_metrics
```

This executes archived, unmodified ProteinGym `merge.py` and
`performance_DMS_benchmarks.py`, including `--performance_by_depth`, and validates
all 217 inputs and outputs. Source hashes are recorded in
`reference/source_provenance.json`. The run includes only this model: a local
rank of 1 and a bootstrap standard error of 0 in the resulting summary refer
to comparison with itself, not a global rank or evidence of zero uncertainty.

## ProteinGym contribution

```sh
python scripts/prepare_proteingym.py --proteingym-root /path/to/ProteinGym \
    --output /path/to/integration
cd /path/to/ProteinGym
git apply --check /path/to/integration/proteingym.patch
git apply /path/to/integration/proteingym.patch
```

The overlay contains a self-contained baseline folder, scoring bash script and
config/constants updates. Fill `publication.json` and regenerate the overlay
before posting the PR and `new model` issue. The archived overlay is a contribution draft and does not establish benchmark
acceptance. The model card explicitly discloses selection on benchmark labels.

## Attribution

Core implementation derives from [DeepGraphLearning/S3F](https://github.com/DeepGraphLearning/S3F),
the code for *Multi-Scale Representation Learning for Protein Fitness Prediction*
(NeurIPS 2024). Original MIT notices are retained. ESM weights and ProteinGym
resources retain their respective upstream terms and attribution. Author/affiliation
metadata for this extension will be provided before publication.
