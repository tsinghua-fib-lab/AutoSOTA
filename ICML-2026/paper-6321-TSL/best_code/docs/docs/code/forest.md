# TSL

`src/forest.rs` and `src/forest/` implement the boosted model — a `Vec<StagePredictor>` —
and the stagewise fitting loop.

## The `TSL` struct

```rust
pub struct TSL {
    stage_predictors: Vec<StagePredictor>,
}
```

`TSL::predict(x)` sums the per-stage predictions: $\hat{m}(\mathbf{x}) = \sum_\ell \hat{m}^{(\ell)}(\mathbf{x})$,
each stage already carrying its OLS scaling.

## The boosting loop (`forest/fitter.rs`)

`fit_boosted(x, y, &TSLBoostedParams) -> (FitResult, TSL)` runs the forward-stagewise loop
(`fit_boosted_with_test_error` is the same but also tracks held-out error per epoch):

1. initialize the outer residuals $R = y$;
2. for each epoch:
    - fit a `StagePredictor` on the current residuals (`fit_ensemble`);
    - extract the per-stage unscaled $\tilde{m}_+, \tilde{m}_-$;
    - **incremental OLS backfit:** refit `scaling_plus`/`scaling_minus` for *all* stages so
      far simultaneously, regressing $y$ on the stacked $[\tilde{m}_+^{(k)}, -\tilde{m}_-^{(k)}]$ columns
      (solved with `LeastSquaresSvd`);
    - recompute residuals $R = y - \sum_k \hat{m}^{(k)}$;
    - after the first epoch, decay the split budget: `n_iter ← n_iter * decay`.

This **orthogonal-greedy** refit (each stage fit greedily on residuals, then *all* stage
coefficients re-optimized together) is what makes later stages correct earlier ones rather
than merely stacking. The math is in [Fitting](../math/fitting.md#coefficient-backfitting).

If a `visualdb_path` is set (and `evo-logging` is on), per-stage snapshots — grid JSON,
$\tilde{m}_+/\tilde{m}_-$ arrays, scaling — are streamed to SQLite; see [Logging](logging.md).

## Parameters (`forest/params.rs`)

`TSLBoostedParams` is the top-level config, flattened by `TSLBoostedParamsBuilder`:

| Field | Meaning |
|-------|---------|
| `epochs` | number of boosting rounds |
| `decay` | multiplicative `n_iter` decay after epoch 1 (`1.0` = none) |
| `sp_params` | nested `StagePredictorParams` (n_trees, aggregation, …) |
| `seed` | RNG seed for reproducibility |
| `visualdb_path` | optional SQLite path for evo-logging |

The builder nests `StagePredictorParamsBuilder` → `GridTensorParamsBuilder`, so a full
configuration is assembled fluently. The Python `TSL.fit(...)` classmethod maps its flat
keyword arguments onto this tree — see [Python API](python-api.md) and the
[Hyperparameters](../guides/hyperparameters.md) reference.

## Tests

- `tests/forest.rs` — boosted-fit reproducibility, seed consistency, the positivity
  invariants, error computation.
- `tests/openml_tuned.rs` — tuned-hyperparameter fits on OpenML datasets.
- `tests/serialization.rs` — Serde round-trips for `GridTensor`, `StagePredictor`, `TSL`.

Run them with `cargo test -p tsl_rust --release` (the suite is numerically heavy — always
`--release`).
