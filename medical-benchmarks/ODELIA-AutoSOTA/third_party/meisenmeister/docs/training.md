# Training and Experiments

This page explains what `mm_train` actually does and what gets written to disk.

## The Default Trainer

The default trainer is `mmTrainer`.

Today it:

- builds fold-specific training and validation datasets from the preprocessed ROI data
- also supports `mm_train -f all`, which uses the full dataset for both training and validation
- instantiates the selected architecture, currently `ResNet3D18` by default
- trains with `CrossEntropyLoss`
- writes a timestamped training log to both stdout and file
- saves both last and best checkpoints
- writes a training-curve PNG after each epoch
- can optionally export Grad-CAM++ masks during final validation when `--grad-cam` is used

## Experiment Folder Layout

Training outputs always live under `MM_RESULTS`.

The path shape is:

`<MM_RESULTS>/<dataset_name>/<Trainer>_<Architecture>/fold_<n>/`

For all-data training, the path uses:

`<MM_RESULTS>/<dataset_name>/<Trainer>_<Architecture>/fold_all/`

If a postfix is supplied, it is appended to the experiment name:

`<MM_RESULTS>/<dataset_name>/<Trainer>_<Architecture>_<postfix>/fold_<n>/`

Examples:

- `.../Dataset_001_OdeliaTest/mmTrainer_ResNet3D18/fold_0/`
- `.../Dataset_001_OdeliaTest/mmTrainer_ResNet3D18/fold_all/`
- `.../Dataset_001_OdeliaTest/mmTrainer_ResNet3D18_finetuningNNSSL/fold_0/`

## Files Written Per Fold

Each fold directory contains:

- `train.log`
- `model_last.pt`
- `model_best.pt`
- `training_curves.png`

If `--grad-cam` is enabled, final validation also writes:

- `grad_cam_last/` after the standard final validation pass
- `grad_cam_best/` when using `mm_train --val best --grad-cam`

### `train.log`

A timestamped record of the run, including:

- trainer and dataset information
- resume or weight-initialization messages
- epoch summaries
- best-checkpoint updates

### `model_last.pt`

The latest checkpoint. This is what resume uses first.

### `model_best.pt`

The checkpoint selected by the trainer’s “best model” rule.

### `training_curves.png`

A three-panel plot with:

- losses and validation metrics
- epoch duration
- learning rate

## Best Model Logic

The trainer does not save the best model based on a single raw spike.

Instead it tracks:

- validation balanced accuracy
- an EMA of validation balanced accuracy

`model_best.pt` is selected using:

1. EMA validation balanced accuracy as the primary signal
2. raw validation loss as the tie-breaker

That is intentionally conservative. It avoids treating one lucky noisy epoch as “the best run so far.”

## Resume Behavior

Resume is explicit:

```bash
mm_train -d 1 -f 0 -c
```

The same applies to all-data training:

```bash
mm_train -d 1 -f all -c
```

If the experiment was created with a postfix, the same postfix must be provided on resume:

```bash
mm_train -d 1 -f 0 -c --postfix finetuningNNSSL
```

Resume restores:

- model weights
- optimizer state
- scheduler state
- metric history
- best-model tracking state
- RNG state where available

If `model_last.pt` is unreadable, the trainer will try `model_best.pt` as a fallback. This is there to make interrupted runs less fragile.

## Fine-Tuning From External Weights

You can start a fresh run from a weights file:

```bash
mm_train -d 1 -f 0 -w /path/to/weights.pt
```

You can also give that run its own experiment name:

```bash
mm_train -d 1 -f 0 -w /path/to/weights.pt --postfix finetuningNNSSL
```

In this mode, the weights file is only used to initialize the model.

The trainer does not import:

- optimizer state
- scheduler state
- previous epoch counters
- previous metric history

So this is a fresh run with inherited model weights, not a resume.

## Allowed Weight File Formats

The fine-tuning loader accepts:

- a plain model `state_dict`
- a trainer checkpoint containing `model_state_dict`

That makes it easy to initialize from either a raw exported model or a prior Meisenmeister checkpoint.

## One Important Constraint

You cannot combine `-c` and `-w`.

That combination is rejected on purpose because it is ambiguous.

- `-c` means “resume this experiment”
- `-w` means “start a new experiment from external weights”

The code fails fast rather than guessing.

## Overwrite Behavior

If you start a non-resume run in an experiment folder that already exists, the trainer logs a clear overwrite warning and continues.

That is intentional. The code does not ask for interactive confirmation during training. The user is expected to choose a postfix if they want a separate experiment namespace.

## A Good Rule Of Thumb

Use:

- no postfix for the canonical base run
- a postfix when you want to preserve a variant as its own experiment
- `-c` only when you want to continue the exact same experiment
- `-w` only when you want a fresh run initialized from another model

## Related Reading

- [CLI Guide](cli.md)
- [Data and Pipeline Notes](pipeline.md)
