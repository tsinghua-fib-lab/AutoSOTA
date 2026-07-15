# tensorsl 0.1.0

* First release: R bindings for the `tsl` (Tensor Separation Learning) Rust
  crate via extendr.
* `tsl(x, y, ...)` fits a boosted TSL model; `predict()` and `print()` S3
  methods are provided. Hyperparameters mirror the Python `TSLRegressor`.
* `tsl_components()` extracts the fitted glass-box structure: per stage, the
  OLS scalings and the aggregated and per-tree grid tensors in two-tensor form
  (per-feature backbone, tilt, splits, and the branch scalars).
* Native ggplot2 interpretability layer. A compute layer (`tsl_pd()`,
  `tsl_pd_2d()`, `tsl_ice()`, `tsl_tilt()`, `tsl_tilt_2d()`,
  `tsl_backbone_2d()`, `tsl_tilt_diagnostics()`, `tsl_importance()`,
  `tsl_local()`) reconstructs partial dependence, spatial backbone/tilt
  surfaces, feature importance, and local explanations exactly from the fitted
  components, and a plot layer (`plot_first_order_pd()`, `pd_difference_plot()`,
  `plot_2d_pd()`, `plot_ice()`, `plot_tilt_1d()`, `plot_2d_tilt()`,
  `plot_tilt_diagnostics()`, `plot_2d_backbone()`, `plot_feature_importance()`,
  `plot_local_interpretation()`, and the grid-tensor component plots) renders
  them in a flat theme. `autoplot.tsl()` is a one-verb entry point, and
  `theme_flat()` plus `scale_*_tsl()` are exported for custom figures.
* A `california` vignette walks through these plotting functions on the
  California housing data.
* `tsl()` retains the training matrix (`x_background`) so the plotting
  functions need no further data.
* Installs from GitHub via `pak::pak("jyliuu/TSL/tsl-r")` /
  `remotes::install_github("jyliuu/TSL", subdir = "tsl-r")`. The core is pure
  Rust, so no system numerical libraries are required.

## Known follow-ups

* **CRAN.** The package is structurally CRAN-ready (vendoring placeholders in
  `src/Makevars.in`), but submission still needs the crates vendored offline
  into `src/rust/vendor.tar.xz` (including the `tsl_rust` git dependency) and
  build-time tuning.
* Windows and macOS CI are not yet exercised (CI covers Linux).
* Model serialisation (save/load) is not yet exposed.
