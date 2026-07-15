# Architecture

TSL is a Cargo workspace plus two Python packages:

| Component | Path | What it is |
|-----------|------|------------|
| `tsl_rust` (lib `tsl`) | `src/` | the Rust core — the model and fitting |
| `tsl-py` | `tsl-py/` | PyO3/maturin wrapper + scikit-learn API |
| `tslviz` | `tsl-split-evolution-dashboard/` | FastAPI + D3 app to replay a fit |

The model is a **three-level hierarchy**, and the `src/` module tree mirrors it exactly.
Read these three files first: `src/grid_tensor.rs`, `src/stage_predictor.rs`,
`src/forest.rs`.

```
TSL (forest)                    Vec<StagePredictor>            src/forest.rs
└── StagePredictor              bag of GridTensors + OLS        src/stage_predictor.rs
    └── GridTensor              one separable component         src/grid_tensor.rs
```

1. **`GridTensor`** ([page](grid-tensor.md)) — one fitted separable component, stored in
   two-tensor form (`backbone_values`, `tilt_values`, `lambda_plus`, `lambda_minus`).
2. **`StagePredictor`** ([page](stage-predictor.md)) — one boosting stage: a bag of
   `n_trees` `GridTensor`s aggregated into one `primary_grid_tensor`, plus OLS
   `scaling_plus`/`scaling_minus`.
3. **`TSL`** ([page](forest.md)) — the boosted model, a `Vec<StagePredictor>`; `predict`
   sums stage predictions.

`src/lib.rs` re-exports the four modules (`stage_predictor`, `forest`, `grid_tensor`,
`logging`) and defines `FitResult { err, residuals, y_hat }`, the common training-result
struct.

## Data flow of a fit

`fit_boosted` (`src/forest/fitter.rs`) drives the whole thing:

1. Initialize residuals $R = y$.
2. For each epoch: fit a `StagePredictor` (`fit_ensemble`) on the current residuals.
   Internally this fits `n_trees` `GridTensor`s (in parallel under `use-rayon`) via the
   grid-refinement loop, then aggregates them.
3. **Backfit:** refit `scaling_plus`/`scaling_minus` for *all* stages so far by incremental
   OLS over the per-stage $[\tilde{m}_+, -\tilde{m}_-]$ columns.
4. Update residuals; `n_iter` decays by `decay` after the first epoch.

See [Fitting](../math/fitting.md) for the math.

## Two critical invariants

These two rules are load-bearing throughout the codebase — get them wrong and predictions
double-count scale or lose the sign structure.

1. **Scaling is applied exactly once**, at the `StagePredictor` level, via
   `scaling_plus`/`scaling_minus` from the OLS backfit. `GridTensor::predict_unscaled` and
   `extract_two_tensor_predictions_unscaled` deliberately return **unscaled** $\tilde{m}_+$/$\tilde{m}_-$;
   the legacy `GridTensor::scaling` field is ignored in two-tensor mode (set to `1.0`). Do
   not multiply by it.

2. **Positivity:** $\lambda_+,\lambda_-\ge 0$ and $b\ge 0$, so $\tilde{m}_+,\tilde{m}_-\ge 0$. This removes
   the sign ambiguity of unconstrained tensor decompositions; signed effects come only from
   the $\tilde{m}_+ - \tilde{m}_-$ difference. Enforced by clamping in the solver
   (`solve_two_tensor`); checked in `tests/forest.rs` and `tests/stage1_positive_only.rs`.

## The action/reducer pattern

Grid-tensor fitting (`src/grid_tensor/`) is structured as an action/reducer loop over a
mutable `FittingState`:

- a `SplitStrategy` (`splitting.rs`: `Random` / `Best` / `TopK`) **proposes** a
  `FittingAction` (`ApplySplit`, `ApplyResplit`, `ApplyMerge`, `ApplyScaling`, `Terminate`,
  `Composite`);
- `fitting_reducer` (`reducer.rs`) **applies** it, calling the `RefinementStrategy`
  (`refinement.rs`: `L2Refinement` / `HuberRefinement`) which solves the per-node $2\times2$
  problem via `two_tensor_solver.rs`.

This separation keeps the fit loop testable and modular. Details on
[GridTensor](grid-tensor.md).

## The builder pattern

Hyperparameters use fluent builders throughout, nested to mirror the hierarchy:

```
TSLBoostedParamsBuilder        (src/forest/params.rs)        — epochs, decay, seed, visualdb
  └── StagePredictorParamsBuilder   (src/stage_predictor/params.rs)  — n_trees, aggregation, similarity
        └── GridTensorParamsBuilder (src/grid_tensor/params.rs)      — n_iter, split + refinement params
              ├── SplitStrategyParamsBuilder
              └── RefinementStrategyParamsBuilder
```

The Python `TSL.fit(...)` classmethod takes flat hyperparameters and maps them onto these
builders; see [Python API](python-api.md) and the
[Hyperparameters](../guides/hyperparameters.md) reference.

## Feature flags

Default features are `use-rayon` (parallel bag fitting) and `evo-logging` (SQLite split
logging — see [Logging](logging.md)). The core is pure Rust and needs no system libraries
to build. See [Getting started](../guides/getting-started.md).
