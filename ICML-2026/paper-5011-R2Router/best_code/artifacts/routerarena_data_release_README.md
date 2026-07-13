# R2-Router RouterArena Data Release

This directory documents the data package used for the RouterArena track of R2-Router.

## Contents

Recommended package layout:

```text
routerarena_data_release/
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
  README.md
  LICENSE_NOTE.md
```

## Files

### `budget_sweep/`

Per-model, per-budget sweep results used by the RouterArena pipeline.

Typical structure:

```text
budget_sweep/
  235b/
    budget_10.json
    budget_20.json
    ...
    concise.json
  80b/
  30b/
  Qwen3-Coder-Next/
  ...
```

These files are used by the routing and evaluation scripts to:

- build consolidated training data
- evaluate routing decisions
- export RouterArena-format submissions

### `category_router/training_data.pkl`

Consolidated training artifact built from sweep outputs and cached embeddings.

Expected fields:

- `embeddings`
- `global_indices`
- `categories`
- `models`

This is the main derived dataset consumed by the category-aware RouterArena scripts.

### `embeddings/routerarena_embeddings.pkl`

Cached embeddings aligned with the main RouterArena query set.

### `embeddings/routerarena_robustness_embeddings.pkl`

Cached embeddings aligned with the robustness split.

### `routerarena_meta/router_data.json`

Main RouterArena query metadata used for global index ordering and export logic.

### `routerarena_meta/router_data_10.json`

RouterArena `sub_10` split metadata used for training/evaluation protocol and export logic.

### `routerarena_meta/model_cost.json`

Model pricing table used to compute routing costs and Arena-style tradeoffs.

## How To Use With The Code Repository

Set the following environment variables before running RouterArena-related scripts:

```bash
export R2_SWEEP_ROOT=/path/to/routerarena_data_release/budget_sweep
export R2_TRAINING_DATA_PATH=/path/to/routerarena_data_release/category_router/training_data.pkl
export R2_EMBEDDINGS_PATH=/path/to/routerarena_data_release/embeddings/routerarena_embeddings.pkl
export R2_ROBUSTNESS_EMBEDDINGS_PATH=/path/to/routerarena_data_release/embeddings/routerarena_robustness_embeddings.pkl
export R2_ROUTER_DATA_PATH=/path/to/routerarena_data_release/routerarena_meta/router_data.json
export R2_ROUTER_DATA_10_PATH=/path/to/routerarena_data_release/routerarena_meta/router_data_10.json
export R2_MODEL_COST_PATH=/path/to/routerarena_data_release/routerarena_meta/model_cost.json
```

Then the main RouterArena workflow is:

```bash
python scripts/train_category_classifier.py
python scripts/train_category_predictors.py
python scripts/route_and_eval.py --lambda_val 0.999 --shrinkage_k 0
python scripts/route_knn_export.py --lambda_val 0.999 --export output.json
```

## Source And Scope

This release contains derived artifacts used by the RouterArena branch of R2-Router.
It may include content derived from RouterArena and upstream benchmark sources.
Please refer to `LICENSE_NOTE.md` for redistribution notes.

## Contact

If any file in this package should be corrected, replaced, or removed for licensing or attribution reasons, please contact the maintainers of the R2-Router release.
