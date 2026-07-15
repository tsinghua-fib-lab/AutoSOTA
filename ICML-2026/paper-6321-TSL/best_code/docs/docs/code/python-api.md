# Python API

`tsl-py/` wraps the Rust core for Python via PyO3/maturin. The package exposes:

```python
from tensorsl import TSL, GridTensor, StagePredictor, FitResult, TSLRegressor
```

- **`TSLRegressor`** — the scikit-learn estimator and **main entry point** for most users.
- **`TSL`** — the raw PyO3 binding to the boosted model, with the interpretation methods.
- **`GridTensor`**, **`StagePredictor`**, **`FitResult`** — the lower-level pieces.

Diagnostics/plotting live in `tensorsl.plot`, documented on the [Plotting reference](plotting.md)
page.

!!! note "Array contract"
    PyO3 methods expect **`float64`** arrays. `TSLRegressor` coerces the dtype for you.

---

## `TSLRegressor` { #cls-tslregressor }

A scikit-learn–compatible regressor wrapping [`TSL`](#cls-tsl). Constructed with flat
hyperparameters; no fitting happens until [`fit`](#meth-tslregressor-fit). See the
[Hyperparameters](../guides/hyperparameters.md) reference for tuning guidance.

```python
TSLRegressor(epochs=10, n_trees=10, n_iter=10, decay=1.0, split_try=10,
             colsample_bytree=0.8, alpha=0.0, complexity_penalty=0.0,
             min_split_loss=0.0, min_interval_samples=1, refinement_strategy="l2",
             prior_sample_size=0.0, update_clamp=float("inf"), tilt_tau=0.01,
             tilt_rho=0.0, split_strategy="random", top_k=10, must_fill_all_k=True,
             similarity_threshold=0.0, bagged=False, seed=42, verbosity=1, visualdb=None)
```

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `epochs` | `int` | `10` | number of boosting rounds (stages) |
| `n_trees` | `int` | `10` | bagged grid tensors per stage |
| `n_iter` | `int` | `10` | split budget per grid |
| `decay` | `float` | `1.0` | multiply `n_iter` by this after epoch 1 |
| `split_try` | `int` | `10` | candidate split positions per (feature, interval) |
| `colsample_bytree` | `float` | `0.8` | fraction of features sampled per split |
| `alpha` | `float` | `0.0` | ridge regularization on the bin update |
| `complexity_penalty` | `float` | `0.0` | penalty discouraging extra splits |
| `min_split_loss` | `float` | `0.0` | minimum error reduction to accept a split |
| `min_interval_samples` | `int` | `1` | minimum observations either side of a split |
| `refinement_strategy` | `str` | `"l2"` | `"l2"` or `"huber"` |
| `prior_sample_size` | `float` | `0.0` | parent-anchoring strength (advanced; `0.0` = off) |
| `update_clamp` | `float` | `inf` | update-magnitude cap (advanced; `inf` = off) |
| `tilt_tau` | `float` | `0.01` | $\ell_2$ coupling between $u_+$ and $u_-$ |
| `tilt_rho` | `float` | `0.0` | $\ell_1$ coupling on $(u_+ - u_-)$ |
| `split_strategy` | `str` | `"random"` | `"random"`, `"best_split"`, or `"top_k"` |
| `top_k` | `int` | `10` | (for `top_k`) candidate pool size |
| `must_fill_all_k` | `bool` | `True` | (for `top_k`) require all $k$ slots |
| `similarity_threshold` | `float` | `0.0` | bag trim $\xi$ (`0` keeps all) |
| `bagged` | `bool` | `False` | enable the bagged-aggregation path |
| `seed` | `int` | `42` | RNG seed (fits are deterministic) |
| `verbosity` | `int` | `1` | log verbosity |
| `visualdb` | `str | None` | `None` | evo-logging SQLite path |

**Attributes** (set after `fit`)

| Name | Type | Description |
|------|------|-------------|
| `core_estimator_` | `TSL` | the fitted core model |
| `fit_result_` | `FitResult` | training diagnostics |
| `stage_predictors` | `list[StagePredictor]` | the fitted stages |

### `fit` { #meth-tslregressor-fit }

```python
TSLRegressor.fit(X, y) -> TSLRegressor
```

Fit the model. Delegates to [`TSL.fit`](#cmeth-tsl-fit), storing the fitted core model in
`core_estimator_` and training diagnostics in `fit_result_`.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | training features (array or DataFrame) |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |

**Returns**

| Type | Description |
|------|-------------|
| `TSLRegressor` | `self`, fitted (scikit-learn convention) |

### `predict` { #meth-tslregressor-predict }

```python
TSLRegressor.predict(X) -> np.ndarray
```

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | features to predict |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | predictions |

### `score` { #meth-tslregressor-score }

```python
TSLRegressor.score(X, y) -> float
```

The coefficient of determination $R^2$ (via scikit-learn's `r2_score`).

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | features |
| `y` | `ndarray (n_samples,)` | _required_ | true targets |

**Returns**

| Type | Description |
|------|-------------|
| `float` | $R^2$ of `predict(X)` against `y` |

### `save` { #meth-tslregressor-save }

```python
TSLRegressor.save(path) -> None
```

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | destination file (binary) |

**Returns**

| Type | Description |
|------|-------------|
| `None` | nothing; writes the model to `path` (binary). |

### `load` { #cmeth-tslregressor-load }

```python
TSLRegressor.load(path) -> TSLRegressor
```

Load a model saved with [`save`](#meth-tslregressor-save); also reads the legacy MPF `.bin` format.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | model file |

**Returns**

| Type | Description |
|------|-------------|
| `TSLRegressor` | a fitted estimator |

---

## `TSL` { #cls-tsl }

The core boosted model. Use it directly for the interpretation methods below; otherwise
prefer [`TSLRegressor`](#cls-tslregressor). Its `fit` classmethod takes the same flat
hyperparameters as the [`TSLRegressor` constructor](#cls-tslregressor); see the
[Hyperparameters](../guides/hyperparameters.md) reference for tuning guidance.

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `stage_predictors` | `list[StagePredictor]` | the stages of the model |

### `fit` { #cmeth-tsl-fit }

```python
TSL.fit(x, y, epochs, decay, n_trees, n_iter, split_try, colsample_bytree, alpha,
        complexity_penalty, min_split_loss, min_interval_samples, refinement_strategy,
        prior_sample_size, update_clamp, tilt_tau, tilt_rho, split_strategy, top_k,
        must_fill_all_k, similarity_threshold, bagged, seed, verbosity, visualdb=None)
        -> tuple[TSL, FitResult]
```

Fit a boosted TSL model.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | training features (`float64`) |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |
| _hyperparameters_ | — | — | same names/types as the [`TSLRegressor` constructor](#cls-tslregressor) |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[TSL, FitResult]` | the fitted model and its training diagnostics |

### `predict` { #meth-tsl-predict }

```python
TSL.predict(x) -> np.ndarray
```

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features (`float64`) |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | sum of all stage predictions |

### `save` { #meth-tsl-save }

```python
TSL.save(path) -> None
```

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | destination binary file |

**Returns**

| Type | Description |
|------|-------------|
| `None` | nothing; serializes the model to `path`. |

### `load` { #cmeth-tsl-load }

```python
TSL.load(path) -> TSL
```

Reads the native binary format and the legacy MPF `.bin` format.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | model file |

**Returns**

| Type | Description |
|------|-------------|
| `TSL` | a fitted model loaded from `path`. |

### `compute_partial_dependence_function` { #meth-tsl-pd }

```python
TSL.compute_partial_dependence_function(fixed_indices, fixed_values, data_x)
    -> tuple[list, np.ndarray]
```

Model-native partial dependence (see [Partial dependence](../math/partial-dependence.md)),
marginalizing over the **empirical joint** of the non-fixed features.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `fixed_indices` | `list[int]` | _required_ | feature column(s) held fixed |
| `fixed_values` | `ndarray (n_points, len(fixed_indices))` | _required_ | values to evaluate at |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[list, ndarray]` | per-stage $(C_+, C_-)$ constants, and branch curves $[\hat{m}_+^{(0)}, \hat{m}_-^{(0)}, \hat{m}_+^{(1)}, \dots]$ |

### `compute_first_order_partial_dependence_functions` { #meth-tsl-pd1 }

```python
TSL.compute_first_order_partial_dependence_functions(values_x, data_x) -> list
```

1D partial dependence for **every** feature at once.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `values_x` | `ndarray (grid_points, n_features)` | _required_ | evaluation grid (column $j$ supplies $x_j$) |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `list` | one `(constants_per_stage, pd_values)` entry per feature |

### `compute_ice_curves` { #meth-tsl-ice }

```python
TSL.compute_ice_curves(observations, feature_index, x_range, data_x) -> np.ndarray
```

Individual Conditional Expectation curves: sweep one feature for each given observation.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `observations` | `ndarray (n_obs, n_features)` | _required_ | rows to trace |
| `feature_index` | `int` | _required_ | feature to vary |
| `x_range` | `ndarray (n_range,)` | _required_ | values to sweep |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_obs, n_range, 2 * n_stages)` | ICE curves, scaled by `scaling_plus`/`scaling_minus` |

### `compute_per_stage_feature_importance` { #meth-tsl-fi-stage }

```python
TSL.compute_per_stage_feature_importance(data_x) -> tuple[np.ndarray, np.ndarray]
```

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray]` | `(backbone, tilt)` importance, each `(n_stages, n_features)` — $\mathrm{Var}[\log b_j]$ and $\mathrm{Var}[d_j]$ |

### `compute_aggregated_feature_importance` { #meth-tsl-fi-agg }

```python
TSL.compute_aggregated_feature_importance(data_x)
    -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Energy-weighted roll-up of the per-stage importances.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray, ndarray]` | `(global_backbone, global_tilt, stage_weights)`, each 1D |

### `compute_combined_feature_importance` { #meth-tsl-fi-combined }

```python
TSL.compute_combined_feature_importance(data_x, gamma=1.0)
    -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

A single combined importance $I_j = I_j^b + \gamma\, I_j^d$ per feature.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |
| `gamma` | `float` | `1.0` | weight on the tilt component |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray, ndarray]` | `(combined, backbone, tilt)`, each 1D over features |

---

## `GridTensor` { #cls-gridtensor }

One fitted separable component (internals: [GridTensor](grid-tensor.md)).

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `splits` | `list[list[float]]` | split thresholds per axis |
| `intervals` | `list` | interval count/bounds per axis |
| `backbone_values` | `list[list[float]]` | $b_j \ge 0$ per interval per axis |
| `tilt_values` | `list[list[float]]` | $d_j \in \mathbb{R}$ per interval per axis |
| `lambda_plus`, `lambda_minus` | `float` | non-negative branch scalars |
| `mean_factor` / `grid_values` | `list` | per-interval $\prod_j b_j e^{d_j}$ |
| `scaling` | `float` | legacy; ignored in two-tensor mode |

### `fit` { #cmeth-gridtensor-fit }

```python
GridTensor.fit(x, y, n_iter, split_try, colsample_bytree,
               complexity_penalty=0.0, seed=42) -> tuple[GridTensor, FitResult]
```

Fit a single grid tensor (no boosting, no bagging).

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | training features |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |
| `n_iter` | `int` | _required_ | split budget |
| `split_try` | `int` | _required_ | candidate split positions |
| `colsample_bytree` | `float` | _required_ | fraction of features per split |
| `complexity_penalty` | `float` | `0.0` | penalty discouraging extra splits |
| `seed` | `int` | `42` | RNG seed |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[GridTensor, FitResult]` | the fitted component and its training diagnostics. |

### `predict` { #meth-gridtensor-predict }

```python
GridTensor.predict(x) -> np.ndarray
```

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | this component's (unscaled) prediction. |

---

## `StagePredictor` { #cls-stagepredictor }

One boosting stage ([details](stage-predictor.md)).

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `grid_tensors` | `list[GridTensor]` | the bag of fitted components |
| `combined_grid_tensor` | `GridTensor` | the aggregated primary grid |
| `candidate_indices` | `list[int] | None` | bags kept after similarity filtering |
| `scaling_plus`, `scaling_minus` | `float | None` | OLS coefficients for $\tilde{m}_+$ and $-\tilde{m}_-$ |

### `predict` { #meth-stagepredictor-predict }

```python
StagePredictor.predict(x) -> np.ndarray
```

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | the stage's contribution, with OLS scaling applied. |

---

## `FitResult` { #cls-fitresult }

Training diagnostics, returned alongside a fitted model.

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `err` | `float` | final training loss |
| `residuals` | `ndarray (n_samples,)` | training residuals |
| `y_hat` | `ndarray (n_samples,)` | training predictions |
