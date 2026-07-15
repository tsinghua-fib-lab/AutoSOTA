# Hyperparameters

All hyperparameters are exposed as flat keyword arguments on `TSLRegressor` (and the
`TSL.fit(...)` classmethod), and mapped onto the nested Rust
[builders](../code/architecture.md#the-builder-pattern):
`TSLBoostedParams` → `StagePredictorParams` → `GridTensorParams`
(`SplitStrategyParams`, `RefinementStrategyParams`). Defaults below are the `TSLRegressor`
constructor defaults.

## Boosting (forest level)

| Param | Default | Meaning |
|-------|:------:|---------|
| `epochs` | `10` | number of boosting rounds (stages $R$); each fits the current residual |
| `decay` | `1.0` | multiply `n_iter` by this after epoch 1 (`<1` makes later stages coarser; `1.0` = off) |
| `seed` | `42` | RNG seed — fits are deterministic given the seed |
| `verbosity` | `1` | log verbosity |
| `visualdb` | `None` | path to an evo-logging SQLite DB; see [Visualization dashboard](visualizing.md) |

## Bag & aggregation (stage level)

| Param | Default | Meaning |
|-------|:------:|---------|
| `n_trees` | `10` | grid tensors fit per stage ($n_{\text{grids}}$); more reduces variance |
| `similarity_threshold` | `0.0` | trim $\xi$ — keep top $\lceil(1-\xi)n_{\text{trees}}\rceil$ bags by similarity (`0` keeps all) |
| `bagged` | `False` | enable the bagged-aggregation path |

The aggregation mode (`Mean` / `GeometricMean` / `Combined`) is set internally; see
[Bagging & aggregation](../math/bagging-aggregation.md).

## Grid refinement (per grid)

| Param | Default | Meaning |
|-------|:------:|---------|
| `n_iter` | `10` | split budget per grid ($T$); larger captures more structure |
| `split_strategy` | `"random"` | `"random"`, `"best_split"`, or `"top_k"` |
| `split_try` | `10` | candidate split positions sampled per (feature, interval) |
| `colsample_bytree` | `0.8` | fraction of features sampled per split proposal |
| `min_interval_samples` | `1` | minimum observations on each side of a split |
| `min_split_loss` | `0.0` | minimum error reduction to accept a split |
| `complexity_penalty` | `0.0` | penalty discouraging additional splits |
| `top_k` | `10` | (for `top_k`) sample from the top-$k$ candidates |
| `must_fill_all_k` | `True` | (for `top_k`) require all $k$ slots filled |

!!! tip "Choosing a split strategy"
    `random` (the default) is usually best for speed and generalizes well; `best_split`
    can help on small datasets where exhaustively picking the top split matters.

## Refinement solver (per node)

| Param | Default | Meaning |
|-------|:------:|---------|
| `refinement_strategy` | `"l2"` | `"l2"` (weights $w_i=1$) or `"huber"` (robust weights) |
| `alpha` | `0.0` | ridge regularization $\alpha$ on the bin update (raise if overfitting) |
| `tilt_tau` | `0.01` | $\ell_2$ coupling $\tau$ between the $u_+$ and $u_-$ updates |
| `tilt_rho` | `0.0` | $\ell_1$ coupling $\rho$ on $(u_+ - u_-)$ |

These feed the [closed-form 2×2 solver](../math/fitting.md#the-closed-form-2x2-solver).

## Advanced / experimental

| Param | Default | Meaning |
|-------|:------:|---------|
| `prior_sample_size` | `0.0` | parent-anchoring strength ($\tau_0$); the default `0.0` disables anchoring |
| `update_clamp` | `inf` | cap on update magnitude; the default `inf` imposes no extra cap |

!!! note
    Leave the two advanced parameters at their defaults unless you are experimenting — the
    defaults shown are the no-op values. The positivity clamp $[v_{\min}, v_{\max}]$ in the
    solver is separate and always active.

## A reasonable starting point

```python
TSLRegressor(
    epochs=5,            # a few stages
    n_trees=16,          # moderate bagging
    n_iter=30,           # enough splits to capture structure
    split_try=16,
    colsample_bytree=0.8,
    alpha=0.01,          # light ridge
    seed=0,
)
```

Tune `epochs`, `n_iter`, and `n_trees` first; reach for `alpha` (and `min_split_loss`) if
the model overfits, and `similarity_threshold` if bagged components disagree.
