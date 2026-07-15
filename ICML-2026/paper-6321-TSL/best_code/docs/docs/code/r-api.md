# R API (`tensorsl`)

`tsl-r/` (R package **`tensorsl`**) wraps the Rust core through
[extendr](https://extendr.github.io/), the R analogue of [`tsl-py`](python-api.md). It
exposes a small S3 `fit`/`predict` interface whose hyperparameters mirror the Python
[`TSLRegressor`](python-api.md#cls-tslregressor) — a model fit in R with the same data and
`seed` reproduces the Python results — plus a native [ggplot2](https://ggplot2.tidyverse.org/)
interpretability layer.

```r
library(tensorsl)
```

The package has three layers:

- **Fit & inspect** — [`tsl()`](#fn-tsl), [`predict()`](#meth-predict-tsl),
  [`print()`](#meth-print-tsl), and [`tsl_components()`](#fn-tsl-components) (the fitted
  glass-box structure).
- **Compute layer** — `tsl_*()` functions that reconstruct the interpretability quantities
  (partial dependence, tilt, backbone, importance, local explanations) exactly from the
  fitted components and return **tidy data frames**.
- **Plot layer** — `plot_*()` functions that render those quantities as ggplots in a flat
  theme, plus the one-verb [`autoplot()`](#meth-autoplot-tsl) entry point. They mirror the
  Python `tensorsl.plot` helpers; the [Plotting reference](plotting.md) shows the rendered
  figures.

!!! note "Input contract"
    `tsl()` and `predict()` take a numeric **matrix** (rows = observations, columns =
    features); a data frame is coerced via `as.matrix()`. `NA`/`NaN`/`Inf` are rejected.
    Column names become the feature names used for labelling.

!!! warning "Not portable across sessions"
    A fitted `tsl` object holds an external pointer into Rust, so it cannot be round-tripped
    through `saveRDS()` / `readRDS()`. Refit in the new session (fits are deterministic given
    `seed`). Serialisation is a documented follow-up.

---

## Fit & inspect

### `tsl()` { #fn-tsl }

```r
tsl(x, y,
    epochs = 10L, n_trees = 10L, n_iter = 10L, decay = 1.0,
    split_try = 10L, colsample_bytree = 0.8, alpha = 0.0,
    complexity_penalty = 0.0, min_split_loss = 0.0, min_interval_samples = 1L,
    refinement_strategy = "l2", prior_sample_size = 0.0, update_clamp = Inf,
    tilt_tau = 0.01, tilt_rho = 0.0, split_strategy = "random",
    top_k = 10L, must_fill_all_k = TRUE, similarity_threshold = 0.0,
    bagged = FALSE, seed = 42L, verbosity = 1L)
```

Fit a boosted TSL model. The flat hyperparameters map onto the same Rust builders as the
Python API; see the [Hyperparameters](../guides/hyperparameters.md) reference for tuning
guidance.

**Parameters**

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `matrix` | _required_ | training features (rows = observations, columns = features) |
| `y` | `numeric` | _required_ | training targets, one per row of `x` |
| `epochs` | `integer` | `10` | number of boosting rounds (stages) |
| `n_trees` | `integer` | `10` | bagged grid tensors per stage |
| `n_iter` | `integer` | `10` | split budget per grid |
| `decay` | `numeric` | `1.0` | multiply `n_iter` by this after epoch 1 |
| `split_try` | `integer` | `10` | candidate split positions per feature (`"random"` strategy) |
| `colsample_bytree` | `numeric` | `0.8` | fraction of features sampled per tree |
| `alpha` | `numeric` | `0.0` | refinement regularisation strength |
| `complexity_penalty` | `numeric` | `0.0` | penalty discouraging extra splits |
| `min_split_loss` | `numeric` | `0.0` | minimum loss reduction to accept a split |
| `min_interval_samples` | `integer` | `1` | minimum observations per interval |
| `refinement_strategy` | `character` | `"l2"` | `"l2"` or `"huber"` |
| `prior_sample_size` | `numeric` | `0.0` | parent-anchoring strength (advanced; `0.0` = off) |
| `update_clamp` | `numeric` | `Inf` | update-magnitude cap (advanced; `Inf` = off) |
| `tilt_tau` | `numeric` | `0.01` | $\ell_2$ coupling between the $u_+$ and $u_-$ tilts |
| `tilt_rho` | `numeric` | `0.0` | $\ell_1$ coupling on $(u_+ - u_-)$ |
| `split_strategy` | `character` | `"random"` | `"random"`, `"best_split"`, or `"top_k"` |
| `top_k` | `integer` | `10` | (for `"top_k"`) candidate pool size |
| `must_fill_all_k` | `logical` | `TRUE` | (for `"top_k"`) require all $k$ slots |
| `similarity_threshold` | `numeric` | `0.0` | bag trim $\xi$ (`0` keeps all) |
| `bagged` | `logical` | `FALSE` | accepted for parity with the Python API; has no effect |
| `seed` | `integer` | `42` | RNG seed (fits are deterministic) |
| `verbosity` | `integer` | `1` | log level: `0` off, `1` info, `2` debug, `3` trace |

**Returns**

An object of class `"tsl"`: a list carrying the fitted-model pointer and training
diagnostics.

| Field | Type | Description |
|------|------|-------------|
| `ptr` | `externalptr` | handle to the fitted Rust model |
| `err` | `numeric` | final training error |
| `residuals` | `numeric` | training residuals |
| `y_hat` | `numeric` | training predictions |
| `feature_names` | `character | NULL` | column names of `x` |
| `n_features`, `n_obs` | `integer` | training shape |
| `x_background` | `matrix` | a copy of the training design matrix, retained so the plotting functions can marginalise over it without it being passed again |
| `call` | `call` | the matched call |

### `predict()` { #meth-predict-tsl }

```r
predict(object, newdata, ...)   # S3 method for class "tsl"
```

Predict from a fitted model. `newdata` must be a numeric matrix with the same number of
columns as the training data; the prediction is the sum of all stage predictions.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `object` | `tsl` | _required_ | a fitted model from `tsl()` |
| `newdata` | `matrix` | _required_ | features to predict (same columns as training) |

**Returns** — a `numeric` vector of predictions, one per row of `newdata`.

### `print()` { #meth-print-tsl }

```r
print(x, ...)   # S3 method for class "tsl"
```

A one-line summary of the fitted model: feature count, training rows, and training error.
Returns `x` invisibly.

### `tsl_components()` { #fn-tsl-components }

```r
tsl_components(object) -> list
```

Extract the fitted glass-box structure in **two-tensor form** — one entry per boosting
stage. Each component's prediction is the ordered difference
$\lambda_+ \prod_j b_j e^{d_j} - \lambda_- \prod_j b_j e^{-d_j}$ (see
[backbone and exponential tilt](../math/model.md#backbone-and-exponential-tilt)).

**Returns** — a list with one element per stage; each stage is a list with:

| Field | Type | Description |
|------|------|-------------|
| `scaling_plus`, `scaling_minus` | `numeric` | the stage's OLS coefficients on the $+$ and $-$ branches (the only place scaling is applied) |
| `candidate_indices` | `integer` | 1-based indices of the bagged trees kept in the stage |
| `combined_grid_tensor` | `list` | the aggregated representative component |
| `grid_tensors` | `list` | the per-tree bag it was aggregated from |

Each grid tensor is itself a list of per-feature `splits`, `backbone_values` ($b \ge 0$),
`tilt_values` ($d \in \mathbb{R}$), and `observation_counts`, plus the branch scalars
`lambda_plus`, `lambda_minus` and the legacy `scaling` (ignored in two-tensor mode).

---

## Interpretability: compute layer

These pure functions reconstruct each interpretability quantity from the fitted components
and return tidy data frames; the `plot_*()` functions build on them, and power users can
consume the data directly. Because TSL is separable the reconstructions are **exact**, not
sampled approximations. The model-native PD math is derived in
[Partial dependence](../math/partial-dependence.md).

### Common parameters

Most compute functions share these; the per-function notes below list only the distinctive
ones.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `object` | `tsl` | _required_ | a fitted model from `tsl()` |
| `X` | `matrix | NULL` | `NULL` | background matrix to marginalise over (default: the training data retained by `tsl()`) |
| `features` | `character | integer | NULL` | `NULL` | features to include, by name or 1-based index (default: all) |
| `feature_x`, `feature_y` | `character | integer` | _required_ | the two features for 2D functions |
| `stages` | `integer | NULL` | `NULL` | stages to include, 1-based (default: all) |
| `grid_points` | `integer` | `200`/`100`/`50` | evaluation resolution |
| `scale` | `character` | `"raw"` | `"raw"` (prediction scale) or `"component"` ($\hat{m}$-space shapes) |

### `tsl_pd()` { #fn-tsl-pd }

```r
tsl_pd(object, X = NULL, features = NULL, grid_points = 200L, stages = NULL,
       scale = "raw") -> data.frame
```

First-order partial dependence per feature and stage: the positive branch `pos`, the
signed-negative branch `neg`, and their sum `net`. Summing `net` over stages gives the
total effect.

**Returns** — a data frame with columns `feature`, `feature_idx`, `stage`, `stage_idx`,
`x`, `pos`, `neg`, `net`, `c_plus`, `c_minus`, `backbone`.

### `tsl_pd_2d()` { #fn-tsl-pd-2d }

```r
tsl_pd_2d(object, X = NULL, feature_x, feature_y, grid_points = 50L,
          stages = NULL) -> data.frame
```

Per-stage signed 2D partial dependence over a feature pair, evaluated on a
`grid_points × grid_points` mesh in long form.

**Returns** — a data frame with columns `x`, `y`, `stage`, `value` (the signed 2D PD). The
grid vectors are attached as attributes `x_vals`/`y_vals`, the feature names as
`feature_x`/`feature_y`.

### `tsl_backbone_2d()` { #fn-tsl-backbone-2d }

```r
tsl_backbone_2d(object, X = NULL, feature_x, feature_y, grid_points = 100L,
                stages = NULL) -> data.frame
```

The per-stage [backbone](../math/model.md#backbone-and-exponential-tilt) product
$b_x(x)\,b_y(y)$ (unsigned magnitude) and the signed 2D partial dependence, stacked in one
long frame with a `panel` factor (`"backbone product"` / `"2D partial dependence"`). Grid
vectors and feature names are attached as attributes.

### `tsl_ice()` { #fn-tsl-ice }

```r
tsl_ice(object, X = NULL, feature, n_ice = 50L, grid_points = 100L,
        seed = 0L) -> list
```

Individual Conditional Expectation curves for one feature: each sampled observation's
prediction as the feature is swept over its range, holding the observation's other features
fixed.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `feature` | `character | integer` | _required_ | the feature to vary |
| `n_ice` | `integer` | `50` | observations sampled for the ICE lines |
| `seed` | `integer` | `0` | seed for the observation sample |

**Returns** — a list with `ice` (data frame `ice_id`, `x`, `y`) and `pd` (data frame `x`,
`y`, the average curve).

### `tsl_tilt()` { #fn-tsl-tilt }

```r
tsl_tilt(object, X = NULL, features = NULL, grid_points = 200L,
         stages = NULL) -> data.frame
```

The piecewise-constant [tilt](../math/model.md#backbone-and-exponential-tilt) $d_j(x)$ (the
signed direction) per feature and stage.

**Returns** — a data frame with columns `feature`, `stage`, `x`, `d`.

### `tsl_tilt_2d()` { #fn-tsl-tilt-2d }

```r
tsl_tilt_2d(object, X = NULL, feature_x, feature_y, grid_points = 100L,
            stages = NULL) -> data.frame
```

Per-stage tilt product $d_x(x)\,d_y(y)$ on a mesh, in long form.

**Returns** — a data frame with columns `x`, `y`, `stage`, `value`; grid vectors and feature
names attached as attributes.

### `tsl_tilt_diagnostics()` { #fn-tsl-tilt-diagnostics }

```r
tsl_tilt_diagnostics(object, X = NULL, features = NULL, grid_points = 200L,
                     stages = NULL) -> data.frame
```

Four diagnostic curves per feature and stage — $\tanh d$, $b\tanh d$, $\tanh(d - \bar d)$,
and $b\tanh(d - \bar d)$, where $b$ is the backbone and $d$ the tilt. See
[Backbone–tilt reconstruction](../math/partial-dependence.md#backbonetilt-reconstruction-from-pd).

**Returns** — a data frame with columns `feature`, `stage`, `x`, `curve`, `value`.

### `tsl_importance()` { #fn-tsl-importance }

```r
tsl_importance(object, X = NULL, gamma = 1) -> list
```

Per-stage and aggregated [feature importance](../math/partial-dependence.md#derived-diagnostics).
Backbone importance is $\mathrm{Var}[\log b_j]$ (how strongly feature $j$ gates); tilt
importance is $\mathrm{Var}[d_j]$ (how strongly it steers). Stages are weighted by their
share of prediction energy; the combined score is $I_j = I_j^b + \gamma\, I_j^d$.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `gamma` | `numeric` | `1` | weight on the tilt term in the combined score |

**Returns** — a list with `per_stage` (data frame `feature`, `stage`, `backbone`, `tilt`),
`global` (data frame `feature`, `backbone`, `tilt`, `combined`), `stage_weights` (data frame
`stage`, `weight`), and `gamma`.

### `tsl_local()` { #fn-tsl-local }

```r
tsl_local(object, x) -> list
```

Decompose the prediction for one point into per-stage positive and negative branch
contributions (summing to the prediction), per-feature backbone and tilt values, and the
stage intercepts that absorb the OLS scalings.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `numeric` | _required_ | a vector of length `n_features` (or a one-row matrix) |

**Returns** — a list with `stage_contributions`, `f_plus_contributions`,
`f_minus_contributions`, `backbone_magnitudes`, `tilt_sums`, `feature_backbone` and
`feature_tilt` (matrices, stages × features), `intercept_backbone`, `intercept_tilt`,
`total_prediction`, a tidy `stages` data frame (`stage`, `fpos`, `fneg`, `net`), and
`feature_names`.

### `tsl_plot_data()` { #fn-tsl-plot-data }

```r
tsl_plot_data(p) -> data.frame | list | NULL
```

Recover the data frame a `plot_*()` function was built from (attached as the `"tsl_data"`
attribute), so the plot can be rebuilt or extended. Returns `NULL` if absent.

---

## Interpretability: plot layer

Each `plot_*()` function reconstructs its data through the compute layer and assembles a
ggplot in the flat theme, attaching the computed data as the `"tsl_data"` attribute (recover
it with [`tsl_plot_data()`](#fn-tsl-plot-data)). They share the
[common parameters](#common-parameters) above. The rendered figures are shown on the
[Plotting reference](plotting.md).

### `plot_first_order_pd()` { #fn-plot-first-order-pd }

```r
plot_first_order_pd(object, X = NULL, features = NULL, grid_points = 200L,
                    stages = NULL, scale = "raw", show_backbone_overlay = TRUE,
                    show_data_density = FALSE) -> ggplot
```

The per-stage, per-feature first-order PD as a faceted grid: the positive branch `PD+`
(orange) and negative branch `PD-` (blue), both on the positive scale, with the signed gap
between them shaded. That gap is the net effect $\mathrm{PD}_+ - \mathrm{PD}_-$.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `show_backbone_overlay` | `logical` | `TRUE` | overlay the $\sqrt{C_+ C_-}\,b$ backbone as a dotted line |
| `show_data_density` | `logical` | `FALSE` | add a bottom rug from a sample of the background rows |

### `pd_difference_plot()` { #fn-pd-difference-plot }

```r
pd_difference_plot(object, X = NULL, features = NULL, grid_points = 200L,
                   stages = NULL, show_backbone_overlay = TRUE,
                   show_data_density = FALSE) -> ggplot
```

The same faceted grid as `plot_first_order_pd()`, framed as the signed contribution: the
shaded gap between the `PD+` and `PD-` curves is the stage's net effect, with the per-feature
backbone optionally overlaid as a dotted line. The workhorse 1D interpretation plot.

### `plot_2d_pd()` { #fn-plot-2d-pd }

```r
plot_2d_pd(object, X = NULL, feature_x, feature_y, grid_points = 50L,
           stages = NULL) -> ggplot
```

Per-stage signed 2D partial dependence over a feature pair, drawn as a tiled heatmap with the
blue–orange diverging fill. Each panel is rescaled to its own 98th-percentile range so weaker
stages stay legible.

### `plot_ice()` { #fn-plot-ice }

```r
plot_ice(object, X = NULL, feature, n_ice = 50L, grid_points = 100L,
         seed = 0L) -> ggplot
```

ICE curves for one feature (faint indigo, one per sampled observation) with the
partial-dependence mean overlaid as a bold dark line. See [`tsl_ice()`](#fn-tsl-ice) for the
distinctive parameters.

### `plot_tilt_1d()` { #fn-plot-tilt-1d }

```r
plot_tilt_1d(object, X = NULL, features = NULL, grid_points = 200L,
             stages = NULL) -> ggplot
```

The per-feature, per-stage [tilt](../math/model.md#backbone-and-exponential-tilt) $d_j(x)$ as
step curves, one panel per `(stage, feature)` cell, with the signed fill carrying the sign
(orange positive, blue negative).

### `plot_2d_tilt()` { #fn-plot-2d-tilt }

```r
plot_2d_tilt(object, X = NULL, feature_x, feature_y, grid_points = 100L,
             stages = NULL) -> ggplot
```

The per-stage tilt product $d_x(x)\,d_y(y)$ as a diverging heatmap, each panel rescaled
symmetrically to its own range so the colour is anchored at zero and comparable across
stages.

### `plot_tilt_diagnostics()` { #fn-plot-tilt-diagnostics }

```r
plot_tilt_diagnostics(object, X = NULL, features = NULL, grid_points = 200L,
                      stages = NULL) -> ggplot
```

The four [tilt diagnostic curves](#fn-tsl-tilt-diagnostics) per feature and stage: the curves
run across the columns, features down the rows, one coloured line per stage in each panel.

### `plot_2d_backbone()` { #fn-plot-2d-backbone }

```r
plot_2d_backbone(object, X = NULL, feature_x, feature_y, grid_points = 100L,
                 stages = NULL) -> patchwork
```

Two stacked rows of per-stage heatmaps over a feature pair: the unsigned backbone product
$b_x(x)\,b_y(y)$ on a sequential indigo ramp, and the signed 2D partial dependence on a
diverging blue–orange ramp — the generic "spatial backbone" plot. Composed with
[patchwork](https://patchwork.data-imaginist.com/); without it, the two component ggplots are
returned as a list.

### `plot_feature_importance()` { #fn-plot-feature-importance }

```r
plot_feature_importance(object, X = NULL, gamma = 1) -> patchwork
```

A six-panel report: per-stage backbone and tilt importance (heatmaps), a stage-weight
histogram, and global bars for tilt, combined, and backbone importance. Composed with
patchwork; without it, the six component ggplots are returned as a named list.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `gamma` | `numeric` | `1` | weight on the tilt term in the combined score |

### `plot_local_interpretation()` { #fn-plot-local-interpretation }

```r
plot_local_interpretation(object, points, titles = NULL, top_k_features = 3L) -> patchwork
```

A per-point "backbone × tilt" decomposition: each query point becomes a row of three panels
sharing one stage ordering (by absolute net contribution) — a signed stage-contribution
waterfall, a per-stage backbone-share marimekko, and per-stage signed tilts. Composed with
patchwork; without it, the per-point [`tsl_local()`](#fn-tsl-local) results are returned.

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `points` | `numeric | list | matrix` | _required_ | one point (length `n_features`), a list of such vectors, or a matrix with one point per row |
| `titles` | `character | NULL` | `NULL` | per-point titles (default `"Point 1"`, `"Point 2"`, …) |
| `top_k_features` | `integer` | `3` | tilt axes kept per stage in the signed-tilt panel |

### Component plots

Raw univariate step components of the fitted grid tensors — each feature's curve is the
per-interval mean factor $m = b\cosh(d)$ over the finite interior intervals.

```r
plot_grid_tensor_components(grid_tensor, axis = NULL, feature_names = NULL) -> ggplot
plot_combined_grid_tensors(object, axis = NULL) -> ggplot
plot_epoch_components(object, epoch) -> ggplot
```

| Function | Draws |
|------|-------------|
| `plot_grid_tensor_components` | one grid tensor's components, e.g. `tsl_components(fit)[[1]]$combined_grid_tensor`; pass `axis` to restrict to one feature |
| `plot_combined_grid_tensors` | each stage's combined grid-tensor components, one facet per stage |
| `plot_epoch_components` | every tree in a stage's bag (one facet per feature), shaded by total scale $\lambda_+ + \lambda_-$, with the combined component overlaid in ink. `epoch` is the 1-based stage index |

---

## One-verb entry point

### `autoplot()` { #meth-autoplot-tsl }

```r
autoplot(object, type = "pd", ...)   # S3 method for ggplot2::autoplot, class "tsl"
```

A convenience wrapper dispatching to the `plot_*()` functions by `type`, so every diagnostic
is reachable from one verb. Extra arguments are forwarded to the underlying function (e.g.
`feature_x`/`feature_y` for the 2D plots, `feature` for `"ice"`, `points` for `"local"`).

| Parameter | Type | Default | Description |
|------|------|:--:|-------------|
| `type` | `character` | `"pd"` | `"pd"`, `"pd_difference"`, `"pd_2d"`, `"ice"`, `"tilt"`, `"tilt_2d"`, `"tilt_diagnostics"`, `"backbone_2d"`, `"importance"`, `"local"`, or `"components"` |
| `...` | — | — | forwarded to the dispatched `plot_*()` function |

```r
library(ggplot2)
autoplot(fit, type = "pd", features = "a")
autoplot(fit, type = "tilt_2d", feature_x = "a", feature_y = "b")
```

**Returns** — a ggplot (or a patchwork object for the composite types).

---

## Theme & scales

The flat aesthetic — white panels, hairline borders, a faint grid, muted monospace labels,
and the indigo / blue–orange palette — is exported for custom figures.

### `theme_flat()` { #fn-theme-flat }

```r
theme_flat(base_size = 11) -> theme
```

The minimal ggplot2 theme behind every `tensorsl` diagnostic, composable with `+`. Pair it with
the `scale_*_tsl()` family for the matching colour ramps.

### `scale_*_tsl()` { #fn-scale-tsl }

```r
scale_fill_tsl_backbone(name = "backbone", ...)        # sequential indigo (magnitude)
scale_fill_tsl_tilt(name = "tilt", ...)                # sequential orange (tilt magnitude)
scale_fill_tsl_diverging(name = "value", limits = c(-1, 1), ...)  # blue–orange, anchored at 0
scale_colour_tsl(name = NULL, ...)                     # calm categorical cycle (per stage)
```

Fill and colour scales matching [`theme_flat()`](#fn-theme-flat). `scale_fill_tsl_diverging()`
squishes values outside its symmetric `limits` to the ends.

---

## Installation & development

`tensorsl` lives in the `tsl-r/` subdirectory of the repo and compiles the Rust core as a static
library at build time. The core is pure Rust and links no system numerical libraries, so the
only prerequisite is a Rust toolchain (`rustc >= 1.80`, from [rustup.rs](https://rustup.rs)).

```r
# pak (owner/repo/subdir):
pak::pak("jyliuu/TSL/tsl-r")

# remotes / devtools:
remotes::install_github("jyliuu/TSL", subdir = "tsl-r")
```

The core is pinned as a **git** dependency that cargo fetches during the build, so no
separate checkout is required. For development against the working-tree core and the wrapper
regeneration workflow, see the package
[README](https://github.com/jyliuu/TSL/blob/main/tsl-r/README.md) and the `california`
vignette.
