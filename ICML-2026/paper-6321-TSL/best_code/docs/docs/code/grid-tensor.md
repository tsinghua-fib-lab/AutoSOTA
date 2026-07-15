# GridTensor

`src/grid_tensor.rs` and the `src/grid_tensor/` module tree implement one fitted separable
component — the rank-1 product in [two-tensor form](../math/model.md#backbone-and-exponential-tilt).
This is the heart of the codebase.

## The `GridTensor` struct

```rust
pub struct GridTensor {
    pub splits: Vec<Vec<f64>>,              // split thresholds per axis
    pub observation_counts: Vec<Vec<usize>>,// samples per interval per axis (for normalization)
    pub intervals: Vec<Vec<(f64, f64)>>,    // interval bounds per axis (±inf at the ends)
    pub scaling: f64,                       // legacy; ignored in two-tensor mode (= 1.0)
    pub backbone_values: Vec<Vec<f64>>,     // b_j ≥ 0  per interval per axis
    pub tilt_values: Vec<Vec<f64>>,         // d_j ∈ ℝ  per interval per axis
    pub lambda_plus: f64,                   // λ₊ ≥ 0
    pub lambda_minus: f64,                  // λ₋ ≥ 0
}
```

Prediction follows the model directly — the difference of the two non-negative branches,
$\hat{m}_+ - \hat{m}_- = \lambda_+\prod_j b_j e^{d_j} - \lambda_-\prod_j b_j e^{-d_j}$:

| Method | Returns |
|--------|---------|
| `new_two_tensor(...)` | construct from the two-tensor fields (sets `scaling = 1.0`) |
| `predict_single_unscaled(x)` / `predict_unscaled(X)` | **unscaled** $\tilde{m}_+ - \tilde{m}_-$ (one point / batch) |
| `predict_single_backbone_and_tilt(x)` / `predict_backbone_and_tilt(X)` | decompose into $(\prod_j b_j,\ \sum_j d_j)$ |

!!! warning "Use the unscaled path"
    `predict_unscaled` returns $\tilde{m}_+ - \tilde{m}_-$ **without** $\lambda$ scaling beyond what is
    baked into the grid; the OLS stage scaling lives on `StagePredictor`. See the
    [scaling invariant](architecture.md#two-critical-invariants).

## The fitting loop

`grid_tensor::fit` (`src/grid_tensor/fit.rs`) iterates the action/reducer pattern:

1. build a `FittingState` from `(x, y)`;
2. initialize the `RefinementStrategy` (precomputes all candidate scores) and the
   `SplitStrategy` (builds allowed intervals);
3. optionally apply the histogram-binning mask (`max_bins`);
4. loop until the fineness budget `n_iter` is hit: the strategy **proposes** an action, the
   **reducer** applies it (mutating the grid, the per-point caches, and the cached
   statistics);
5. normalize with [`l2_identify`](#identification) and return the `GridTensor`.

### `FittingState` (`state.rs`)

The mutable per-grid context threaded through the loop: the in-progress two-tensor grid
(`backbone_values`, `tilt_values`, `lambda_plus`, `lambda_minus`); per-point caches
(`f_plus`, `f_minus`, `f`, `r_tilde` = within-stage residuals); which interval each point
falls in per axis; and a `PrecomputedStatistics` cache of prefix sums and per-interval
sufficient statistics enabling $O(1)$ candidate scoring. Holds an optional `LoggingState`.

### Actions (`action.rs`) and splitting (`splitting.rs`)

`FittingAction` enumerates the state mutations: `ApplySplit`, `ApplyResplit`, `ApplyMerge`,
`ApplyScaling`, `Terminate`, `Composite`. The `SplitStrategy` enum proposes them:

- **`Random`** — sample up to `split_try` valid positions over `colsample_bytree` of the
  features (the default; usually best for speed);
- **`Best`** — greedily take the highest-gain split (better on small data);
- **`TopK`** — sample from the top-`k` candidates (`must_fill_all_k` controls strictness).

A `SplitCandidate` carries the column, position, error reduction, and the left/right
$(u_+, u_-)$ updates. `SplitStrategyState` tracks which positions remain valid; after a
split at position `idx`, the neighborhood is forbidden from further splitting.

### The reducer (`reducer.rs`)

`fitting_reducer` matches on the action and applies it: it inserts the split into the grid,
updates backbone/tilt on the two new sub-intervals, recomputes the affected per-point caches
and the prefix-sum/interval statistics, and (under `evo-logging`) emits log events.

## The two-tensor solver

`solve_two_tensor` (`src/grid_tensor/two_tensor_solver.rs`) is the load-bearing primitive.
It builds the regularized $2\times2$ system from the five sufficient statistics
$S_{11}, S_{22}, S_{12}, t_1, t_2$,

$$
A = \begin{pmatrix} S_{11}+\alpha+\tau & S_{12}-\tau \\ S_{12}-\tau & S_{22}+\alpha+\tau \end{pmatrix},
\qquad t = \begin{pmatrix} t_1 \\ t_2 \end{pmatrix},
$$

solves $\hat{u}=A^{-1}t$ (or an iterative step when the $\ell_1$ coupling $\rho>0$), clamps
$v_\pm = \mathrm{clamp}(1+\hat{u}_\pm, v_{\min}, v_{\max})$ to preserve positivity, and
returns the gain $\mathcal{L}(0,0) - \mathcal{L}(\hat{u})$ used to score the candidate. The
full derivation is in [Fitting → the closed-form solver](../math/fitting.md#the-closed-form-2x2-solver).

### Refinement strategies (`refinement.rs`)

`RefinementStrategy` has two variants that differ only in the weights $w_i$:

- **`L2Refinement`** — $w_i = 1$;
- **`HuberRefinement`** — robust weights $w_i = \min(1, c/|r_i|)$, $c\approx1.345$.

Both expose `alpha`, `tilt_tau` ($\tau$), `tilt_rho` ($\rho$), `prior_sample_size`, and
`update_clamp`. `initialize` precomputes the error reduction of every candidate position.

## Identification

`l2_identify` (`src/grid_tensor/identification.rs`) applies the prediction-preserving
[normalization](../math/model.md#normalization-gauge-fixing) after a fit: it rescales each
axis's backbone to unit weighted $L^2$ (pushing the scale into $\lambda_\pm$) and centers
each tilt by its weighted mean (pushing the offset into $\lambda_+ \mathrel{*}= e^{\sum c_j}$,
$\lambda_- \mathrel{*}= e^{-\sum c_j}$). It does **not** flip orientation (which would change
predictions). The same routine runs before averaging bagged grids
([Bagging & aggregation](../math/bagging-aggregation.md)).

## Histogram binning

`apply_histogram_binning_mask` (`src/grid_tensor/histogram_binning.rs`) is an optional
prologue enabled by `max_bins: Some(u16)`. It computes deterministic quantile bin edges per
feature, forbids non-edge split positions, and replaces per-position prefix sums with
per-bin cumulative sums. This cuts the per-candidate cost from $O(n)$ to $O(\text{bins})$
with little accuracy loss on larger datasets. With `max_bins: None` the exact path
(every position a candidate) is used. Parity between the two paths is tested in
`tests/histogram_binning.rs`.

## Parameters (`params.rs`)

`GridTensorParams` bundles `n_iter` (the split budget), a `SplitStrategyParams`
(`split_try`, `colsample_bytree`, `min_interval_samples`, `min_split_loss`, kind), a
`RefinementStrategyParams` (`alpha`, `tilt_tau`, `tilt_rho`, …, kind), and `max_bins`. All
have fluent `…Builder` types. The Python flat hyperparameters map onto these — see
[Hyperparameters](../guides/hyperparameters.md).
