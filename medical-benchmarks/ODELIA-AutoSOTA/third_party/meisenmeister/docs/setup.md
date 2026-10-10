# Setup and Paths

## Installation

Meisenmeister is packaged as a normal Python project with console-script entrypoints.

```bash
conda create -n meisenmeister python=3.12 -y
conda activate meisenmeister
pip install -e .
```

`pip install -e .` is the most practical choice during active development because the CLI immediately reflects local code changes.

## Required Paths

Most commands need three configured roots:

- `MM_RAW`
- `MM_PREPROCESSED`
- `MM_RESULTS`

The code also accepts lowercase variants:

- `mm_raw`
- `mm_preprocessed`
- `mm_results`

These are read by `meisenmeister.utils.path_utils`.

## Recommended Shell Setup

Add these to your shell profile:

```bash
export MM_RAW=/path/to/mm_raw
export MM_PREPROCESSED=/path/to/mm_preprocessed
export MM_RESULTS=/path/to/mm_results
```

Then start a new shell or source the profile before using the CLI.

## What Each Path Is For

### `MM_RAW`

The source dataset root.

This directory should contain dataset folders named like:

`Dataset_001_Name`

Inside a dataset folder, the code expects the usual project-specific raw structure such as:

- `imagesTr`
- `masksTr`
- `dataset.json`

### `MM_PREPROCESSED`

The derived-data root.

For each dataset, Meisenmeister writes files such as:

- `dataset_fingerprint.json`
- `mmPlans.json`
- `splits.json`
- `labelsTr.json`
- the preprocessed ROI output folder named by `mmPlans.json`

### `MM_RESULTS`

The training-results root.

Training experiments are created underneath this path and hold checkpoints, logs, and plots.

## Dataset Resolution

Most commands take a short integer dataset id:

```bash
mm_train -d 1 -f 0
```

Internally that becomes `001`, and Meisenmeister searches `MM_RAW` for exactly one folder matching:

`Dataset_001_*`

If there is no match, or more than one match, the command fails early and loudly.

## One Good Sanity Check

Before doing real work, confirm your paths resolve cleanly by running a lightweight command on a known dataset, for example:

```bash
mm_extract_dataset_fingerprint -d 1
```

If path setup is wrong, this usually fails quickly with a useful error.

## Related Reading

- [High-Level Overview](overview.md)
- [CLI Guide](cli.md)
