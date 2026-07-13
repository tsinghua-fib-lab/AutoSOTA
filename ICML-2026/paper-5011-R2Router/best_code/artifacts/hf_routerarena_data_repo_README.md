# R2-Router RouterArena Data

This repository hosts the released data artifacts for the **RouterArena branch** of **R2-Router**.

## Contents

```text
budget_sweep/
category_router/
  training_data.pkl
embeddings/
  routerarena_embeddings.pkl
  routerarena_robustness_embeddings.pkl
routerarena_meta/
  router_data.json
  router_data_10.json
  model_cost.json
LICENSE_NOTE.md
```

## Description

This release contains the derived data used by the RouterArena pipeline of R2-Router, including:

- per-model, per-budget sweep outputs
- consolidated training artifacts
- cached query embeddings
- RouterArena metadata used by the training and evaluation scripts

These files are intended to support:

- offline RouterArena evaluation
- retraining the released RouterArena predictors
- reproducing exported RouterArena submissions

## Code Repository

Please use this data package together with the main code repository:

- code: `R2-Router`
- reproduction entrypoints:
  - `reproduce/routerarena_train.sh`
  - `reproduce/routerarena_eval.sh`

## Expected Environment Variables

```bash
export R2_SWEEP_ROOT=/path/to/budget_sweep
export R2_TRAINING_DATA_PATH=/path/to/category_router/training_data.pkl
export R2_EMBEDDINGS_PATH=/path/to/embeddings/routerarena_embeddings.pkl
export R2_ROBUSTNESS_EMBEDDINGS_PATH=/path/to/embeddings/routerarena_robustness_embeddings.pkl
export R2_ROUTER_DATA_PATH=/path/to/routerarena_meta/router_data.json
export R2_ROUTER_DATA_10_PATH=/path/to/routerarena_meta/router_data_10.json
export R2_MODEL_COST_PATH=/path/to/routerarena_meta/model_cost.json
```

## Notes

- This release focuses on the RouterArena branch only.
- Some contents are derived from RouterArena and upstream benchmark/task sources.
- Please see `LICENSE_NOTE.md` for redistribution notes.

## Citation

If you use this release, please cite the R2-Router paper/code release.
