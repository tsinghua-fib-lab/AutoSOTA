# Release Inventory

This clean release was curated from the working research repository for the camera-ready paper. It intentionally excludes files that are not needed to reproduce the experiments.

## Excluded

- Raw datasets.
- Model checkpoints.
- Generated result grids and figures.
- Historical drafts, review-response notes, screenshots, notebooks, local logs, archives, and old submission folders.
- Cluster-specific account names and absolute paths.
- Development-only certifier variants that do not correspond to the paper experiments.

## Expected User-Provided Paths

- MNIST data root, downloaded by `torchvision`.
- Rotated MNIST model checkpoint after training.
- UTKFace image directory.
- MiVOLO-v2 Hugging Face checkpoint directory.
