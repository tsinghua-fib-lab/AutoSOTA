# ORB GFFMerge Experiments

This repository contains scripts and notebooks used to train, merge, and evaluate ORB models.

## Repository layout
- `notebooks/` — main experiment notebooks
- `scripts/` — training, merging, and evaluation utilities
- `data/` — data location (not included in the repo)
- `models/` — generated checkpoints (created by notebooks)
- `results/` — evaluation outputs (created by notebooks)
- `logs/` — training/merge logs (created by notebooks)

## Setup
1. Create a Python environment (3.10+ recommended).
2. Dependencies are installed by the notebooks’ `pip install` cell; if you skip that cell, install the same packages manually.

## Reproducibility
- All experiments were run from the notebooks in `notebooks/`.
- The notebooks assume they are opened from the `notebooks/` directory (default Jupyter behavior) and write outputs to `../data`, `../models`, `../results`, and `../logs`.
- Experiment seeds and hyperparameters are defined in the configuration cells near the top of each notebook.
- The notebooks include a lightweight `pip install` cell for extra packages; this list can vary by environment. If you skip that cell, install the additional packages manually.

### Recommended execution order
1. `notebooks/Experiments_pipeline.ipynb`
2. `notebooks/Baselines_pipeline.ipynb`

### Outputs
- Training checkpoints: `models/`
- Merged checkpoints: `models/merged/`
- Evaluation results: `results/`
- Logs: `logs/`

## Data
Raw datasets are not included. See `data/README.md` for expected structure and preparation steps.
