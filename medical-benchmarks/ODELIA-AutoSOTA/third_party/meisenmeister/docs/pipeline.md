# Data and Pipeline Notes

This page explains what the preprocessing side of the project produces and how the pieces fit together.

## Raw Dataset Expectations

The code resolves datasets from `MM_RAW` using folder names like:

`Dataset_001_Something`

Within a dataset folder, Meisenmeister expects the project-specific raw layout, including:

- `imagesTr`
- `masksTr`
- `dataset.json`

The utility code validates that the expected training files are present and that the basic dataset metadata is complete before doing heavier work.

## What Fingerprinting Produces

`mm_extract_dataset_fingerprint` reads the dataset and writes:

- `dataset_fingerprint.json`

This file captures ROI-level geometry statistics such as:

- spacings
- cropped shapes
- full shapes
- median spacing
- median shape after crop

That fingerprint is what the planning step uses.

## What Planning Produces

`mm_plan_experiment` writes:

- `mmPlans.json`

The plan currently includes values such as:

- normalization mode
- ROI label mapping
- margin in millimeters
- target spacing
- target shape
- output format
- output folder name

In practice, `mmPlans.json` is the bridge between “what the dataset looks like” and “how preprocessing should write the trainable ROI data.”

## What Preprocessing Produces

`mm_preprocess` reads the raw data and `mmPlans.json`, then writes the preprocessed ROI dataset under the dataset’s preprocessed directory.

This includes:

- the actual ROI data in the configured output folder
- labels and split-related metadata already used by the training code

The training dataset class reads this preprocessed structure directly, rather than going back to the raw NIfTI files.

## Combined Planning and Preprocessing

`mm_plan_and_preprocess` is a convenience wrapper that does:

1. planning
2. preprocessing

It is the right command when you do not need a pause between those stages.

## Five-Fold Split Creation

`mm_create_5fold` writes:

- `splits.json`

The split logic is case-aware and checks for train/validation leakage at the case level. That matters because the preprocessed dataset is ROI-based, and multiple ROIs can belong to the same case.

## How Training Uses Preprocessed Data

`mm_train` does not read the raw dataset directly.

Instead it relies on:

- the resolved dataset folder name from `MM_RAW`
- the matching dataset directory under `MM_PREPROCESSED`
- `splits.json`
- `labelsTr.json`
- the preprocessed ROI output folder named in `mmPlans.json`

That design keeps training focused on the model loop rather than on raw-image preprocessing.

## Useful Practical Sequence

If you want a straightforward “just get me to training” sequence:

```bash
mm_create_breast_segmentations -d 1
mm_extract_dataset_fingerprint -d 1
mm_plan_and_preprocess -d 1 --num-workers 4
mm_create_5fold -d 1
mm_train -d 1 -f 0
```

If you want more control, split planning and preprocessing apart and inspect the intermediate files before training.

## Related Reading

- [High-Level Overview](overview.md)
- [Setup and Paths](setup.md)
- [CLI Guide](cli.md)
