# RouterArena Checkpoints Release

This document describes the pretrained checkpoint package for the RouterArena branch of R2-Router.

## Recommended Package Layout

```text
routerarena_checkpoints_release/
  category_router/
    classifier.joblib
    classifier_cv_results.json
    category_means.pkl
    predictor_metrics.pkl
    predictor_results.json
    routing_results.json
    train_test_split.pkl
    predictors/
      Code/
      Math/
      Knowledge/
      NLU/
      Translation/
      Trivia/
      Domain/
  README.md
```

## What `category_router/` Contains

The `category_router/` directory is the full pretrained checkpoint bundle for the category-aware RouterArena pipeline.

It includes:

1. Query category classifier
   - `classifier.joblib`

2. Per-category quality predictors
   - `predictors/<Category>/*_quality.joblib`
   - one predictor for each `(category, model, budget)` combination

3. Per-category token predictors
   - `predictors/<Category>/*_token.joblib`
   - one predictor for each `(category, model)` combination

4. Metadata and statistics
   - `*_quality_meta.json`
   - `predictor_results.json`
   - `predictor_metrics.pkl`
   - `category_means.pkl`
   - `train_test_split.pkl`

## How The Code Uses These Checkpoints

The main loading path is in:

- `scripts/route_and_eval.py`

At evaluation time, the code:

1. loads per-category predictors from `predictors/<Category>/`
2. loads category-level means and CV statistics
3. predicts quality and token usage for each model and budget
4. routes each query using the token-aware risk objective

## Recommended Environment Variable

Point the code to this checkpoint directory with:

```bash
export R2_CHECKPOINT_DIR=/path/to/routerarena_checkpoints_release/category_router
```

## When Users Should Download This Package

Users should download this checkpoint package if they want to:

- run RouterArena evaluation without retraining
- reproduce routing behavior more quickly
- inspect pretrained predictors directly

Users do not strictly need this package if they are willing to retrain from the released RouterArena data package using:

```bash
bash reproduce/routerarena_train.sh
```

## Compatibility Note

These checkpoints are stored using `joblib` and depend on compatible Python and scikit-learn versions.

If checkpoint loading fails due to environment mismatch, users should retrain predictors from the released data package instead of relying on binary compatibility.
