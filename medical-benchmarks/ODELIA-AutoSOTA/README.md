# ODELIA AutoSOTA

Seven independent lightweight classification heads improve the readout of frozen
MeisenMeister breast MRI encoders. Each fold keeps its released encoder and original
classification output, learns a constrained correction, and contributes equally to the
final probability average. This repository packages the final method, its reproducible
released-model baseline, and aggregate evaluation evidence.

**Evaluation scope:** the reported numbers are a same-cohort comparison on the
published Fold 0 test set, not independent hidden-challenge results. Public-test
aggregate feedback influenced candidate selection. The released seven-fold weights
were trained with a leave-one-source-out protocol on collected ODELIA/AMBL data and
may already have seen public Fold 0 test cases in several ensemble members. We do
not claim leakage-free generalization, a new official leaderboard result, or exact
retraining of the original champion. See [fairness and limitations](#fairness-and-limitations).

## Research report

Read the [full research report](docs/research_report.md) for the evaluation protocol,
results, six research phases, and limitations. The [search-history appendix](docs/search_history.md)
preserves all 198 iteration records from the supplied report. Exact aggregate values
are in [historical_metrics.json](results/historical_metrics.json).

## Data and task

- Task: three-class classification of each breast ROI: `0=healthy`, `1=benign`, `2=malignant`.
- [ODELIA Challenge dataset](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025):
  741 examinations from six centers; public split: 472 train, 119 validation, 150 test cases.
- [BreastDividerDataset V2](https://huggingface.co/datasets/Bubenpo/BreastDividerDataset):
  the local preparation used 196 AMBL cases and 511 non-RSH ODELIA cases from this resource.
- Inputs: pre-contrast, first post-contrast, second post-contrast DCE-MRI; left and right ROIs.
- Common preprocessing: released `mmPlans.json`, foreground per-channel z-score,
  3 mm margin, spacing `[2.0, 0.7, 0.7]`, ROI shape `[3, 128, 320, 320]`.

| Local adaptation partition | Cases | Breast ROIs | Composition |
|---|---:|---:|---|
| Training | 604 | 1,208 | 408 non-RSH ODELIA public-training + 196 AMBL |
| Validation | 103 | 206 | Non-RSH ODELIA public-validation |
| Fixed public test | 150 | 300 | Published Fold 0 test, including 20 RSH cases |

Train/validation partitioning is by examination, so both breasts of one examination
stay together. The 80 RSH public train/validation cases are excluded from adaptation
training and validation. The public-test cohort contains 130 non-RSH and 20 RSH cases;
it is not the original organizer-held regular/OOD evaluation cohort.

The larger prepared image pool had 937 cases (741 ODELIA + 196 AMBL), but only the
604/103 cases above trained/selected the adaptation heads. Dataset and weight revisions
are pinned in [provenance.json](provenance.json). Download resources from their providers
and follow their access terms. Images, labels, exact case-membership lists, cached features,
and pretrained weights are not bundled. Matching the historical cohort requires those
external artifacts; the sample counts alone are insufficient for exact reproduction.

## How we reproduced the baseline

The baseline is **inference reproduction of the released seven-fold MeisenMeister**,
not training from scratch. Its exact source snapshot is bundled in
[`third_party/meisenmeister`](third_party/meisenmeister), commit
`a9b8ded20d59a7f6df8b03f24a6ec28d9d8ede0e`.

1. Download the pinned [MeisenMeister release](https://huggingface.co/Bubenpo/MeisenMeister)
   and [BreastDivider weights](https://huggingface.co/ykirchhoff/BreastDividerModel).
2. Convert public Parquet image arrays to NIfTI using the published dataset conversion
   procedure; retain all three channels and image geometry.
3. Run the released ROI/inference pipeline with `fold_0` through `fold_6`, checkpoint
   `best`, and the released default test-time augmentation and compilation settings.
   Historical execution processed disjoint batches and merged all 150 cases without
   missing or duplicate ROIs.
4. Keep the seven original per-fold probabilities. Identity adaptation heads reproduce
   those values; their fixed arithmetic mean is the baseline. Do not drop folds or tune weights.
5. Score baseline and adapted predictions on the **same 300 ROI identifiers and labels**
   with the same micro-ROC metric implementation in [`metrics.py`](src/odelia_autosota/metrics.py).

Upstream installation and raw-image inference:

```bash
python -m pip install -e third_party/meisenmeister
mm_predict_from_modelfolder weights/MeisenMeister \
  -i data/nifti -o artifacts/released_predictions -f 0 1 2 3 4 5 6 --checkpoint best
```

The raw-image pipeline needs separately acquired data and weights. For the historical
BreastDivider release, `SimpleITKIOWithReorient` was unavailable in the installed reader
registry. An isolated runtime copy of its plan used `NibabelIOWithReorient`; checkpoint
tensors, dataset, and evaluator were unchanged. See [baseline reproduction](docs/baseline_reproduction.md).

## Final optimization

For each fold, Stage 3/4/5 channel-wise mean, maximum and population standard deviation
form a 2,688-dimensional descriptor (768 + 960 + 960). Five folds use independent
`LayerNorm → Linear(16) → GELU → Linear(3)` stage probes; folds 1 and 5 use bias-only heads.
Stage outputs start at zero, are bounded by `tanh`, and are anchored to the released logits
with KL regularization. The encoders are frozen; no cross-fold features or shared head
parameters are used. Final ensemble weights remain exactly `1/7`.

| Fold | Head | Per-stage residual bound | KL weight | Temperature |
|---:|---|---:|---:|---|
| 0 | Stage probes, no bias | 0.25 | 1.0 | Fixed 1 |
| 1 | Bias only, bound 0.2 | — | 0 | Fixed 1 |
| 2 | Stage probes, bias bound 0.2 | 0.25 | 1.0 | Fixed 1 |
| 3 | Stage probes, bias bound 0.4 | 1.00 | 2.0 | Learned, bounded |
| 4 | Stage probes, bias bound 0.4 | 1.50 | 2.0 | Learned, bounded |
| 5 | Bias only, bound 0.2 | — | 0 | Fixed 1 |
| 6 | Stage probes, bias bound 0.2 | 0.40 | 1.0 | Fixed 1 |

Each head trains for 300 epochs with batch size 128, AdamW `lr=3e-4`,
`weight_decay=1e-3`, fold seed `3107 + fold`, and cosine decay over 3,000 steps
for residual heads. Best epoch is selected on validation Challenge score, with lower
validation NLL as the tie breaker. See [method](docs/method.md).

## Reported results

Historical run `run_20260817_060026`; best iteration 29; final source commit
`83b4bce929d78b9f159736adc649ac30cf922e9e`.

| Metric on 300 public-test ROIs | Released baseline | Final method |
|---|---:|---:|
| Challenge score | 0.876448 | **0.899644** |
| Micro-AUC | 0.951011 | **0.963933** |
| Macro OVR AUC | 0.931091 | **0.954975** |
| Specificity at 90% sensitivity | 0.841667 | **0.868333** |
| Sensitivity at 90% specificity | 0.836667 | **0.866667** |
| Balanced accuracy | 0.675725 | **0.683102** |
| Macro F1 | 0.710471 | **0.721422** |
| NLL ↓ | 0.429921 | **0.397167** |
| Benign recall | 0.312500 | 0.312500 |
| Malignant recall | 0.745763 | **0.762712** |

All seven folds improve **Challenge score**, not every individual fold submetric.
Validation Challenge score increases from 0.842685 to 0.844181, while validation
Micro-AUC, macro F1, balanced accuracy and benign recall have trade-offs.
Full numerical results and artifact hashes are in
[`results/historical_metrics.json`](results/historical_metrics.json).

Challenge score is the mean of micro-AUC, specificity at 90% sensitivity and sensitivity
at 90% specificity, computed after flattening three-class one-hot labels and probabilities.
Older baseline reports used additional macro/binary operating-point summaries; those
numbers must not be substituted for the fixed micro-ROC components in this comparison.

## Installation and quick start

From the AutoSOTA repository root, enter this archive first:

```bash
cd medical-benchmarks/ODELIA-AutoSOTA
```

Python 3.12 or newer. Install a PyTorch build appropriate for your device, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
odelia-autosota --help
```

The core toolkit requires only PyTorch, NumPy and scikit-learn. The optional vendored
raw-image baseline has its own additional dependencies. Historical numerical versions
are in `requirements-historical.txt`; portability tests do not guarantee identical
clinical scores across hardware or library versions.

### Train and save the final heads

```bash
odelia-autosota pack --cache-root data/seven_encoder_cache --output data/training.npz
odelia-autosota train --packed-cache data/training.npz \
  --output-dir artifacts/final_heads --device cuda:0 --strict-historical
```

If only preprocessed ROIs and published weights are available, first generate the cache:

```bash
python examples/extract_frozen_features.py --dataset data/preprocessed \
  --split data/splits_challenge_fair_fold0.json --checkpoint-root weights/MeisenMeister \
  --output data/seven_encoder_cache --device cuda:0 --amp-dtype float16
```

This optional helper requires the separately installed vendored baseline dependencies.
It preserves stage indices, population standard deviation, float32 final-mean readout,
and examination-level split membership. It has been checked structurally but has not
been re-run on the full clinical cohort during this packaging session.

The training archive contains train/validation only; test labels are absent.
The command saves seven `fold_*/head_best.pth` files and validation reports.
For new research cohorts, omit `--strict-historical`; the fixed final method and its
3,000-step schedule remain unchanged. See [data formats](docs/data_formats.md).

### Compare the released baseline and final heads

Modern input: one NPZ with `sample_ids`, `features` and `released_logits`.
Labels live in a separate archive used only by `evaluate`:

```bash
odelia-autosota baseline --inputs data/test_inputs.npz --output-dir artifacts/baseline
odelia-autosota infer --inputs data/test_inputs.npz --run-dir artifacts/final_heads \
  --output-dir artifacts/adapted --device cuda:0
odelia-autosota evaluate --prediction-archive artifacts/baseline/predictions.npz \
  --labels data/test_labels.npz --output artifacts/baseline_metrics.json
odelia-autosota evaluate --prediction-archive artifacts/adapted/predictions.npz \
  --labels data/test_labels.npz --output artifacts/adapted_metrics.json
```

For the historical test path, use **the original seven-fold TTA probabilities** and
the frozen test descriptor archive for both commands:

```bash
odelia-autosota baseline --predictions data/released_predictions.json \
  --block-features data/block_features.npz --output-dir artifacts/baseline_legacy
odelia-autosota infer --predictions data/released_predictions.json \
  --block-features data/block_features.npz --run-dir artifacts/final_heads \
  --output-dir artifacts/adapted_legacy --device cuda:0
```

This legacy route forms test logits as `log(clip(released_probability, 1e-8, 1))`,
matching the experiment. Unaugmented logits from a new extraction are a different
inference protocol and must not be presented as an exact reproduction of these scores.

`examples/synthetic_demo.py` runs the full baseline/train/save/load/infer/evaluate
workflow on artificial data; its metrics are **not** clinical results. No trained
final-head checkpoints are supplied by this source-only release.

## Fairness and limitations

The same-cohort comparison holds encoder weights, preprocessing, sample identifiers,
labels, metrics, and equal ensemble weights constant. Each head uses only its own
fold's features/logits. Adaptation gradients use training labels only; epoch selection
uses validation only. These controls make the reported **relative comparison** auditable.

They do not remove two limitations:

1. Released encoder training may overlap the public-test cases. The released converter
   uses the collected 937-case pool with leave-one-source-out folds, rather than strictly
   following the public split column. Several models may therefore have seen each test case.
   The bundle does not establish the full checkpoint training membership independently.
2. Public-test aggregate scores guided architecture/hyperparameter search over 32 candidates.
   This is public-test-adaptive development. Test labels did not enter head-gradient training,
   but the outer selection is not blind testing.

The competition-time Docker image and original hidden evaluation cohorts are unavailable.
The exact MAMA-MIA ResEncUNetL `fold_all` initialization used to train the released classifier
is not publicly available in the audited segmentation weight package. Thus this release
does not claim a from-scratch, bit-identical reconstruction of champion training.
No confidence interval or statistical-significance claim is made for the reported gains.

## Code organization

The public entry point is `odelia-autosota` (or `python -m odelia_autosota`).
`cli.py` parses arguments and dispatches to dedicated modules; it contains no
training loop or metric implementation. Data handling is shared by training and
inference, and evaluation consumes saved predictions plus separate labels.
`heads.py` is the authoritative definition of the final model and fold settings;
`configs/final_heads.json` is a tested read-only audit mirror, not a competing
runtime configuration. Model and metric computations are preserved.

The vendored baseline lives entirely under `third_party/meisenmeister`; its
upstream tests and docs are retained there. The root `tests/` covers our toolkit.
Raw candidate-search workspaces, AutoSOTA service code, API configuration, clinical
artifacts, build outputs and virtual environments are excluded. The research report
and Markdown search-history appendix retain the supplied aggregate research evidence.

## Repository layout

```text
src/odelia_autosota/
  cli.py                    arguments and command dispatch
  heads.py                  final model, fold settings and optimization rules
  data.py                   archive validation, packing and identifier alignment
  training.py               train and save validation-selected heads
  inference.py              adapted and identity-baseline prediction
  evaluation.py             score saved predictions against separate labels
  features.py               exact Stage 3/4/5 descriptor construction
  metrics.py                fixed evaluation formulas
third_party/meisenmeister/ fixed upstream baseline source and license
configs/                   final fold configuration audit mirror
results/                   aggregate historical results only
docs/                      research report, search history and reproduction guides
examples/                  artificial-data demonstration and extraction helper
tests/                     protocol, independence, metrics and round-trip tests
.github/                   CPU test workflow and bug report template
```

## License and citation

Source code is distributed under Apache-2.0; upstream attribution is in `NOTICE`.
Published MeisenMeister weights use [CC BY-NC-SA 4.0](https://huggingface.co/Bubenpo/MeisenMeister/blob/main/README.md).
Data and weights retain their own terms and are not redistributed here.
For research use, cite [MeisenMeister](https://arxiv.org/abs/2510.27326) and describe
the independent-head adaptation, this release version, and the public-test selection protocol.
Research software; no clinical deployment validation is claimed.
