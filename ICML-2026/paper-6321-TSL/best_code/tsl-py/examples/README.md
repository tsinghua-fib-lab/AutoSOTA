# TSL Python — examples

End-to-end scripts that reproduce **every plot from the TSL paper**, using
the generic plotting helpers in `tensorsl.plot` plus a small amount of
per-dataset custom code (local explanations, paper-style split-panel
exports, EBM/XGBoost comparison plots).

## Running

```bash
# from the repo root
python tsl-py/examples/california.py
python tsl-py/examples/bike_sharing.py
python tsl-py/examples/synthetic.py
python tsl-py/examples/synthetic2.py
```

Each script:
- accepts `--data-root` (raw CSVs; defaults to `reproducibility/data`)
- accepts `--out` (defaults to `/tmp/tsl_examples/<dataset>`)
- defaults its model paths to the pretrained binaries shipped in
  `tsl-py/examples/models/<dataset>/` (`mpf_*.bin` for TSL, `ebm_model.pkl`
  for EBM, `xgb_model*.json` for XGBoost). `TSL.load(...)` reads the legacy
  MPF `.bin` format directly. Pass `--refit` to retrain TSL from scratch,
  or set a path to `""` to skip that model (and any figure that depends on it).

## What each example produces

Filenames are human-readable — no `figure_X_Y_` paper-index prefixes.

### California ([`california.py`](california.py))

| Output file | Source |
|---|---|
| `pd_difference_plot_{blackbox,interpretable}.pdf` | `pd_difference_plot` |
| `spatial_backbone_evolution_{blackbox,interpretable}.pdf` | `plot_2d_backbone` (combined 2×2: backbone product + 2D PD per stage) |
| `spatial_tilt_evolution_{blackbox,interpretable}.pdf` | `plot_2d_tilt` (combined 1×2, on the California map) |
| `tilt_1d_{blackbox,interpretable}.pdf` | `plot_tilt_1d` (Latitude, Longitude, MedInc) |
| `tilt_diagnostics_{blackbox,interpretable}.pdf` | `plot_tilt_diagnostics` |
| `feature_importance_{blackbox,interpretable}.pdf` | `plot_feature_importance` |
| `local_explanations_{blackbox,interpretable}.pdf` | verbatim port of `cali_analysis.py::plot_figure_5_local_explanations` |
| `local_interpretation_intercept_{coastal,desert}_{blackbox,interpretable}.pdf` | `plot_local_interpretation` (card grid, intercept broken out) |
| `pd_comparison_{latitude,longitude}_{blackbox,interpretable}.pdf` | 1D PD overlay: TSL (Stage 1) + EBM + XGBoost (blackbox) + XGBoost (interpretable) + SepALS (optional) |

Pass `--variant {blackbox, interpretable}` to switch the TSL model. To
regenerate the paper's full California figure set, run both variants
into the same `--out` directory (the paper uses the *interpretable* TSL
for the spatial-backbone plot, so run that variant last):

```bash
python tsl-py/examples/california.py --variant blackbox      --out tsl-py/examples/figures/california
python tsl-py/examples/california.py --variant interpretable --out tsl-py/examples/figures/california
```

### Bike Sharing ([`bike_sharing.py`](bike_sharing.py))

| Output file | Source |
|---|---|
| `pd_difference_plot.pdf` | `pd_difference_plot` |
| `pd_hour_workingday_tsl.pdf` | `plot_2d_pd` (`kind="lines"`) |
| `pd_hour_workingday_ebm.pdf` | EBM PD via repeated `ebm.predict` (custom) |

### Synthetic PD-cancellation ([`synthetic.py`](synthetic.py))

| Output file | Source |
|---|---|
| `pd_difference_plot.pdf` | `pd_difference_plot` (combined 2×3) |
| `ice_x1_tsl.pdf` | TSL `plot_ice` |
| `ice_x1_ebm.pdf` | EBM ICE (custom) |
| `ice_x1_xgboost.pdf` | XGBoost ICE (custom) |
| `pd_x1_all_models.pdf` | 1D PD overlay: TSL + EBM + XGBoost |
| `pd_x1_x2.pdf` | TSL `plot_2d_pd` (surface) |

### Synthetic2 — backbone bimodality at epoch 0 ([`synthetic2.py`](synthetic2.py))

Reconstructs the figure from Reviewer zSd4's rebuttal: at a single TSL stage,
independently bagged grid tensors on the DGP

    f(x1, x2) = exp(sin(x1) * cos(x2)) + x1,    x1, x2 ~ U[-4, 4],
    y = f(x1, x2),  n = 5000

converge to two distinct backbone representations (clearly bimodal on
Feature 0). The similarity-filtering step (Algorithm 11) recovers a single
coherent set.

| Output file | Description |
|---|---|
| `backbone_bimodal_epoch0.pdf` | 3-row x 2-col panel of backbones `b_j^{(ℓ)}(x_j)` at stage `ℓ = 1`: row 1 splits 17 bottom-λ + 17 top-λ bagged grids by `λ+ + λ-`; row 2 shows the top-`k = ceil((1 - ξ) * n_grids)` similarity-filtered candidates against the `(λ+, λ-)`-anchored reference grid; row 3 shows all `n_grids` bagged backbones colored by `λ+ + λ-`, with the combined backbone overlaid in black. |

The script does a single TSL fit with `similarity_threshold = 1 - ξ` (default
`ξ = 0.9`, giving `|K| = 39` kept candidates out of 389), reads all 389
bagged grids from `stage_predictors[0].grid_tensors`, and computes the
Algorithm 11 reference (closest to the `(λ+, λ-)`-centroid) and combined
similarity scores in Python for the row-2 highlight. Fitting takes a few
seconds on a laptop with rayon enabled. Pass `--seed N` to fix all
stochastic components, or `--n-trees`, `--xi`, `--alpha`, etc. to vary the
configuration.

## Pre-rendered output

[`figures/`](figures/) holds the rendered PDFs from a fresh run on the
pretrained models, mirroring the per-dataset paths above. Re-running the
scripts with default arguments will reproduce them.

## Optional dependencies

The comparison plots need `interpret` (for EBM) and `xgboost`:

```bash
pip install interpret 'xgboost==2.1.3'
```
