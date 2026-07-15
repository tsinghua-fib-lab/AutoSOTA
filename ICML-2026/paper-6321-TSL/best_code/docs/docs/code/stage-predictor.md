# StagePredictor

`src/stage_predictor.rs` and `src/stage_predictor/` implement one **boosting stage**: a bag
of `GridTensor`s aggregated into one representative grid, plus the OLS scaling that is the
*only* place stage scaling is applied.

## The `StagePredictor` struct

```rust
pub struct StagePredictor {
    pub grid_tensors: Vec<GridTensor>,        // the bag of fitted components (n_trees)
    pub primary_grid_tensor: GridTensor,      // the aggregated representative grid
    pub candidate_indices: Option<Vec<usize>>,// indices kept after similarity filtering
    pub aggregation_method: Aggregation,      // Mean | GeometricMean | Combined
    pub scaling_plus: Option<f64>,            // OLS coefficient for m̃₊
    pub scaling_minus: Option<f64>,           // OLS coefficient for −m̃₋
    pub energy: Option<f64>,                  // optional unscaled prediction energy
}
```

### Prediction and the scaling invariant

`StagePredictor::predict` is where the OLS scaling is applied — exactly once:

$$
\text{stage prediction} = \texttt{scaling_plus}\cdot \tilde{m}_+ \;+\; \texttt{scaling_minus}\cdot(-\tilde{m}_-).
$$

The free function `extract_two_tensor_predictions_unscaled(grid, x)` pulls $\tilde{m}_+$ and $\tilde{m}_-$
straight from the two-tensor fields with **no** scaling — this is what the
[backfit](../math/fitting.md#coefficient-backfitting) regresses on, and what enforces the
[scaling invariant](architecture.md#two-critical-invariants). `predict_unscaled` returns the
stage prediction before OLS scaling.

### The `Aggregation` enum

```rust
pub enum Aggregation { Mean, GeometricMean, Combined }
```

| Variant | How the bag is reduced |
|---------|------------------------|
| `Mean` | arithmetic mean of the unscaled per-grid predictions |
| `GeometricMean` | sign-preserving geometric mean |
| `Combined` | take $\tilde{m}_+, \tilde{m}_-$ from the aggregated two-tensor `primary_grid_tensor` and apply `scaling_plus`/`scaling_minus` |

## Fitting a stage (`fitter.rs`)

`fit_ensemble` builds a stage:

1. derive `n_trees` seeds and fit that many `GridTensor`s independently — in parallel under
   the `use-rayon` feature;
2. align them to a common grid (`refine_grids_to_union_two_tensor`) and canonicalize each
   with `l2_identify`;
3. similarity-filter and combine them into `primary_grid_tensor`
   (`aggregate_bagged_two_tensor`, gated by `similarity_threshold`);
4. solve OLS for `scaling_plus`/`scaling_minus` where the aggregation mode needs it;
5. return the `StagePredictor` and a `FitResult`.

## Combining grids

- **`combine_grids.rs`** — `refine_grids_to_union_two_tensor` resamples every grid onto the
  union of all split points per axis (so components are comparable interval-by-interval);
  `geometric_mean_combiner` is the sign-preserving geometric mean used internally.
- **`aggregate_bagged.rs`** — `aggregate_bagged_two_tensor` runs the full
  [align → normalize → reference → similarity-filter → log-space average](../math/bagging-aggregation.md)
  pipeline and returns the kept `candidate_indices` plus the combined grid.

## Parameters (`params.rs`)

`StagePredictorParams` carries `n_trees`, the nested `GridTensorParams`,
`similarity_threshold` ($\xi$), and `aggregation_method`, with a fluent
`StagePredictorParamsBuilder` that delegates to `GridTensorParamsBuilder`. See
[Architecture → builder pattern](architecture.md#the-builder-pattern).
