# TSL — Tensor Separation Learning

**TSL** is a glass-box regression model: it fits accurately *and* lets you read its learned
structure directly off one-dimensional plots — with no post-hoc surrogate. A fitted model
is a small sum of *stages*, each a separable product of per-feature curves, so every
feature's effect is recoverable **exactly** from a partial-dependence curve.

Everything below is produced by TSL itself (via the `tensorsl.plot` helpers) — these figures
*are* the model, not an approximation of it.

## What TSL shows you

### Where a stage is active — the spatial backbone

<figure markdown="span">
  ![California spatial backbone and per-stage 2D partial dependence](assets/img/california_spatial_backbone.png){ width="100%" }
  <figcaption markdown="span">California housing — two boosting [stages](math/fitting.md), each shown as two panels. <strong>Top row:</strong> the 2D [backbone](math/model.md#backbone-and-exponential-tilt) product $b_\mathrm{lon}(x)\cdot b_\mathrm{lat}(y)$, a non-negative gate whose magnitude encodes where the stage is active (darker blue = larger). <strong>Bottom row:</strong> the 2D [partial dependence](math/partial-dependence.md) $\hat{m}_+ - \hat{m}_-$, the signed prediction each stage contributes at that location (warm orange = positive, cool blue = negative). Stage 1 captures a broad coastal gradient; Stage 2 refines it with a sharper Bay Area and LA correction.</figcaption>
</figure>

### Faithful 1-D partial dependence

<figure markdown="span">
  ![1D partial dependence on latitude and longitude](assets/img/california_pd_comparison.png){ width="100%" }
  <figcaption markdown="span">TSL's [first-order partial dependence](math/partial-dependence.md) for Latitude (left) and Longitude (right) on California housing, overlaid against EBM, XGBoost, and SepALS baselines. TSL (dark blue) shows sharp localized peaks — a spike near latitude 37–38 (San Francisco / Bay Area) and a concentrated coastal band near longitude −122 — that additive-marginalization baselines smooth into nearly monotone slopes. This faithfulness follows from separability: for a product-form model the 1D PD curve recovers the exact factor shape rather than a marginal main effect.</figcaption>
</figure>

### Feature importance — magnitude vs. direction

<figure markdown="span">
  ![Per-stage backbone and tilt feature importance](assets/img/california_feature_importance.png){ width="100%" }
  <figcaption markdown="span">Six-panel [feature importance](code/plotting.md#fn-plot-feature-importance) report on California housing. <strong>Top row (heatmaps):</strong> per-stage backbone importance $\mathrm{Var}[\log b_j]$ (left) and tilt importance $\mathrm{Var}[d_j]$ (right) — rows are stages, columns are features, colour encodes magnitude. <strong>Bottom row (bars):</strong> global tilt importance (left), combined score $I_j = I_j^b + \gamma I_j^d$ (center), global backbone importance (right), with Stage 1 dominating the energy-weight panel. Longitude and Latitude lead in backbone (they gate the spatial stages on/off); Latitude and MedInc lead in tilt (they drive the signed price direction within each stage).</figcaption>
</figure>

### Local, per-observation explanations

<figure markdown="span">
  ![Local explanation for a coastal point](assets/img/california_local_interp_coastal.png){ width="100%" }
  <figcaption markdown="span">[Local explanation](code/plotting.md#fn-plot-local-interpretation) for a coastal home in the San Francisco Bay area — Longitude −122.41, Latitude 37.70, MedInc 2.41, HouseAge 23, TotalRooms 1817, TotalBedrooms 400, Population 1376, Households 382. <strong>Left card:</strong> per-stage contribution bars summing to the total prediction. <strong>Center cards:</strong> per-stage [backbone](math/model.md#backbone-and-exponential-tilt) share, showing which features gate each stage on (Latitude and Longitude dominate Stage 1). <strong>Right card:</strong> signed tilt $d_j$ bars indicating the direction of each feature's effect. The 10-stage blackbox TSL predicts <strong>$173,675</strong>, with Stage 1 contributing +\$172,967 (the coastal premium) and later stages making small corrections.</figcaption>
</figure>

<figure markdown="span">
  ![Local explanation for a desert point](assets/img/california_local_interp_desert.png){ width="100%" }
  <figcaption markdown="span">[Local explanation](code/plotting.md#fn-plot-local-interpretation) for an inland (desert) home near Palm Springs — Longitude −116.50, Latitude 33.81, MedInc 2.54, HouseAge 26, TotalRooms 5032, TotalBedrooms 1229, Population 3086, Households 1183 — shown for contrast with the coastal point above. Stage 1 again dominates (+\$118,103), but the [backbone](math/model.md#backbone-and-exponential-tilt) shares and tilt signs differ: at this inland longitude/latitude the spatial gate is weaker, and Longitude and Latitude contribute a negative tilt rather than the large positive coastal premium. The 10-stage blackbox TSL predicts <strong>$111,364</strong>.</figcaption>
</figure>

!!! tip "Read this first — it is short, and it is the point"
    These plots only mean something once you know what a **backbone**, a **tilt**, and a
    **stage** are. We **strongly recommend** reading the **[Under the hood](math/model.md)**
    section before relying on the figures: start with [The model](math/model.md), then
    [Partial dependence](math/partial-dependence.md). It is what makes TSL interpretable
    rather than just another regressor.

## Examples { #examples }

The figures above come from runnable scripts in
[`tsl-py/examples/`](https://github.com/jyliuu/TSL/tree/main/tsl-py/examples), which
reproduce the paper's plots via the `tensorsl.plot` helpers:

```bash
python tsl-py/examples/california.py
python tsl-py/examples/bike_sharing.py
python tsl-py/examples/synthetic.py
python tsl-py/examples/synthetic2.py
```

| Script | What it shows |
|--------|---------------|
| `california.py` | spatial backbone, 1D PD faithfulness, feature importance, local explanations |
| `bike_sharing.py` | 2D `hour × workingday` interaction PD |
| `synthetic.py` | the masked interaction — signed PD ≈ 0 for every model, yet the backbone recovers the effect |
| `synthetic2.py` | bagging diagnostics (backbone bimodality + similarity filtering) |
| `sepals_synthetic.py` | small synthetic factor-value / SepALS comparison |

Each script accepts `--data-root`, `--out`, `--refit`, and `--variant`; the pretrained
models in `examples/models/` are used by default. See the
[examples README](https://github.com/jyliuu/TSL/blob/main/tsl-py/examples/README.md) for
the full per-script output list and flags.

## How the codebase is organized

The model is a three-level hierarchy, and the `src/` module tree mirrors it exactly:

| Level | Type | What it is | Docs |
|------|------|------------|------|
| 1 | `GridTensor` | one fitted separable component (backbone/tilt + $\lambda_\pm$) | [GridTensor](code/grid-tensor.md) |
| 2 | `StagePredictor` | one boosting stage: a bag of `GridTensor`s + OLS scaling | [StagePredictor](code/stage-predictor.md) |
| 3 | `TSL` | the boosted model: a `Vec<StagePredictor>` summed | [TSL](code/forest.md) |

The core is the Rust crate `tsl_rust` (library name `tsl`). `tsl-py/` wraps it for Python
with a scikit-learn API ([Python API](code/python-api.md)), `tsl-r/` (`tensorsl`) wraps it for R
with an S3 `fit`/`predict` interface and a ggplot2 interpretability layer
([R API](code/r-api.md)), and `tsl-split-evolution-dashboard/` (`tslviz`) visualizes how a
fit was built ([Visualization dashboard](guides/visualizing.md)).

## Where to start

Before anything else, read the **[Under the hood](math/index.md)** material — start with
**[Notation](math/index.md)** and **[The model](math/model.md)** to understand what a
backbone, a tilt, and a stage are. That understanding is what makes TSL interpretable
rather than just another regressor, and it is worth five minutes before you fit your first
model. From there: [Fitting](math/fitting.md) and
[Partial dependence](math/partial-dependence.md) round out the theory.

- **New to TSL?** [Getting started](guides/getting-started.md) — install, fit, predict.
- **Using the model?** The [Python API](code/python-api.md) or [R API](code/r-api.md), the
  [Hyperparameters](guides/hyperparameters.md) reference, then [Examples](#examples).
- **Working on the code?** Start with [Architecture](code/architecture.md) and its two
  critical invariants, then the per-module pages.
