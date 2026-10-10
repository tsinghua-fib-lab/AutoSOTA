# Overview

Meisenmeister is a command-line workflow for turning raw breast MRI datasets into preprocessed ROI crops and then training classification models on those crops.

At a high level, the project does three things:

1. It validates and fingerprints a dataset.
2. It plans and writes preprocessed ROI data into the project’s own format.
3. It trains a model with fold-based splits, experiment folders, checkpoints, plots, and resume support.

## The Mental Model

There are three important storage roots:

- `mm_raw`: where the source dataset folders live
- `mm_preprocessed`: where Meisenmeister writes fingerprints, plans, splits, labels, and `.b2nd` ROI data
- `mm_results`: where training experiments and checkpoints live

Each command takes a dataset id like `-d 1`. Internally that becomes `001`, and Meisenmeister resolves a dataset folder with a name like `Dataset_001_Something`.

## The Normal Workflow

For a new dataset, the usual sequence is:

1. `mm_create_breast_segmentations -d <id>` if segmentations do not exist yet.
2. `mm_extract_dataset_fingerprint -d <id>`
3. `mm_plan_experiment -d <id>` or `mm_plan_and_preprocess -d <id>`
4. `mm_preprocess -d <id>` if planning and preprocessing are done separately
5. `mm_create_5fold -d <id>`
6. `mm_train -d <id> -f <fold>`

That is the path the codebase is built around.

## What The CLI Actually Gives You

The repository installs these console commands:

- `mm_extract_dataset_fingerprint`
- `mm_create_breast_segmentations`
- `mm_homogenize`
- `mm_plan_experiment`
- `mm_preprocess`
- `mm_plan_and_preprocess`
- `mm_create_5fold`
- `mm_train`

If you only want one page to orient yourself before using the project, read [CLI Guide](cli.md).

## Training In One Paragraph

Training is fold-based and experiment-based.

Each run writes into:

`<mm_results>/<dataset_name>/<Trainer>_<Architecture>[/_<postfix>]/fold_<n>/`

All-data training via `mm_train -f all` writes into:

`<mm_results>/<dataset_name>/<Trainer>_<Architecture>[/_<postfix>]/fold_all/`

Inside that folder you get:

- `train.log`
- `model_last.pt`
- `model_best.pt`
- `training_curves.png`

You can start from scratch, fine-tune from external weights with `-w`, or resume an existing experiment with `-c`. Resume and external weights are intentionally not allowed together in the same command because they mean two different sources of truth.

## Where To Go Next

- For first-time setup, go to [Setup and Paths](setup.md)
- For command-by-command usage, go to [CLI Guide](cli.md)
- For checkpoints, fine-tuning, postfix naming, and resume behavior, go to [Training and Experiments](training.md)
- For what is written during planning and preprocessing, go to [Data and Pipeline Notes](pipeline.md)
